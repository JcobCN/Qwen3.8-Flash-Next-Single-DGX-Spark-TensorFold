# TensorFold 接入 LMCache 跨节点 KV Cache 复用改造指南

本文档记录了在 NVIDIA DGX Spark 集群（GB10 架构）上，针对 `Qwen3.8-Flash-Next` 混合架构（Attention + Gated DeltaNet 线性注意力 + MTP），将单机 TensorFold 镜像改造成支持 **LMCache 跨进程 / 跨节点分布式 KV Cache 复用**的完整方案。

后续若原基础镜像 `hello/qwen3.8-flash-next-single-dgx-spark-tensorfold:lastest` 发生版本升级，遵循本文档可在数分钟内快速重现改造。

---

## 1. 核心架构与改造原理

### 1.1 架构难点与突破
`Qwen3.8-Flash-Next` 并非单纯的 Transformer 模型，包含：
- **12 层 Full Attention**：需要缓存 Keys、Values 以及 Indexer keys（`ikc`）和 Pooled keys（`pooled`），在 `--kv-dtype int8` 下还包含 fp16 的 scales。
- **36 层 Gated DeltaNet（线性注意力）**：包含循环隐状态 `rec`（`[2, n, nv, dv, dk]`，float32）和卷积隐状态 `conv`（`[n, kernel-1, dim]`，bfloat16）。
- **MTP（多 Token 预测）Head**：其自身包含独立的注意力缓存与已草稿位置记录。

**传统仅迁移 KV Cache 的方案会导致模型输出灾难性错乱。本方案通过桥接层完整持久化上述全量状态，实现跨节点 100% 精确还原。**

### 1.2 改造拓扑
```text
  Client / API Gateway
       │               │
       ▼ (Request A)   ▼ (Request B，相同前缀)
┌──────────────┐     ┌──────────────┐
│  Node 19     │     │  Node 20     │
│  TensorFold  │     │  TensorFold  │
└──────┬───────┘     └──────┬───────┘
       │ (1. 异步入库)       │ (2. 探针命中 & 拉取)
       ▼                    ▼
┌───────────────────────────────────┐
│     LMCache Server (0.0.0.0:8998) │
│     (部署在 19 或独立的集中节点)    │
└───────────────────────────────────┘
```

---

## 2. 依赖环境安装（可在新容器内执行）

LMCache 官方提供 aarch64 Python 3.12 预编译 Wheel 包：

```bash
# 在容器内安装 lmcache 及核心依赖
pip install lmcache
```

> **注意**：若目标机器无法直连外网，可通过内网从已安装节点同步 `/usr/local/lib/python3.12/dist-packages/` 下与 `lmcache` 相关的依赖包，或使用离线 wheel 缓存。

---

## 3. 代码修改清单

改造仅涉及新增 **1 个桥接模块** 和修改 **1 个调度文件**。

### 3.1 新增桥接适配器：`tensorfold/lmcache_bridge.py`
创建文件：`/usr/local/lib/python3.12/dist-packages/tensorfold/lmcache_bridge.py`

```python
"""LMCache bridge for TensorFold: cross-node KV and recurrent state reuse across DGX Spark nodes."""

from __future__ import annotations

import io
import os
import time
import socket
import struct
import hashlib
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import torch
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.memory_management import MemoryFormat
from lmcache.utils import CacheEngineKey

logger = logging.getLogger("tensorfold.lmcache")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [LMCacheBridge] %(message)s")

MODEL_NAME = os.environ.get("LMCACHE_MODEL_NAME", "Qwen3.8-Flash-Next")
SERVER_HOST = os.environ.get("LMCACHE_SERVER_HOST", "192.168.1.19")
SERVER_PORT = int(os.environ.get("LMCACHE_SERVER_PORT", "8998"))
MIN_TOKENS = int(os.environ.get("LMCACHE_MIN_TOKENS", "256"))
ENABLED = os.environ.get("TENSORFOLD_LMCACHE", "1") in ("1", "true", "True")


def is_enabled() -> bool:
    return ENABLED and bool(SERVER_HOST)


def hash_tokens(tokens: Sequence[int]) -> int:
    h = hashlib.blake2b(struct.pack(f"{len(tokens)}q", *tokens), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=True)


def make_key(tokens: Sequence[int]) -> CacheEngineKey:
    return CacheEngineKey(
        model_name=MODEL_NAME,
        world_size=1,
        worker_id=0,
        chunk_hash=hash_tokens(tokens),
        dtype=torch.uint8,
    )


class LMCacheClient:
    def __init__(self, host: str, port: int, timeout: float = 3.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None

    def _get_socket(self) -> socket.socket | None:
        if self._sock is not None:
            return self._sock
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            self._sock = sock
            return sock
        except Exception as e:
            logger.debug(f"Failed to connect to LMCache server at {self.host}:{self.port}: {e}")
            return None

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def exists(self, key: CacheEngineKey) -> bool:
        with self._lock:
            sock = self._get_socket()
            if sock is None:
                return False
            try:
                meta = ClientMetaMessage(
                    ClientCommand.EXIST, key, 0, MemoryFormat(1), torch.uint8, torch.Size([0])
                )
                sock.sendall(meta.serialize())
                header = sock.recv(ServerMetaMessage.packlength())
                if len(header) < ServerMetaMessage.packlength():
                    self._close()
                    return False
                res = ServerMetaMessage.deserialize(header)
                return res.code == ServerReturnCode.SUCCESS
            except Exception as e:
                logger.debug(f"LMCache exists check failed: {e}")
                self._close()
                return False

    def get(self, key: CacheEngineKey) -> dict | None:
        with self._lock:
            sock = self._get_socket()
            if sock is None:
                return None
            try:
                meta = ClientMetaMessage(
                    ClientCommand.GET, key, 0, MemoryFormat(1), torch.uint8, torch.Size([0])
                )
                sock.sendall(meta.serialize())
                header = sock.recv(ServerMetaMessage.packlength())
                if len(header) < ServerMetaMessage.packlength():
                    self._close()
                    return None
                res = ServerMetaMessage.deserialize(header)
                if res.code != ServerReturnCode.SUCCESS:
                    return None
                total = res.length
                received = bytearray()
                while len(received) < total:
                    packet = sock.recv(min(total - len(received), 4 * 1024 * 1024))
                    if not packet:
                        self._close()
                        return None
                    received.extend(packet)
                return torch.load(io.BytesIO(received), weights_only=False)
            except Exception as e:
                logger.warning(f"LMCache GET failed: {e}")
                self._close()
                return None

    def put(self, key: CacheEngineKey, payload: dict) -> bool:
        buf = io.BytesIO()
        torch.save(payload, buf)
        data = buf.getvalue()
        with self._lock:
            sock = self._get_socket()
            if sock is None:
                return False
            try:
                meta = ClientMetaMessage(
                    ClientCommand.PUT,
                    key,
                    len(data),
                    MemoryFormat(1),
                    torch.uint8,
                    torch.Size([len(data)]),
                )
                sock.sendall(meta.serialize())
                sock.sendall(data)
                return True
            except Exception as e:
                logger.warning(f"LMCache PUT failed: {e}")
                self._close()
                return False


_CLIENT: LMCacheClient | None = None
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lmcache_io")


def get_client() -> LMCacheClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LMCacheClient(SERVER_HOST, SERVER_PORT)
    return _CLIENT


def retrieve(owner, prompt: list[int]) -> tuple[Any, dict, int] | None:
    """尝试从远程 LMCache 获取最长匹配的前缀状态"""
    if not is_enabled():
        return None
    client = get_client()

    candidates: list[int] = []
    if getattr(owner, "points", None) is not None:
        try:
            candidates.extend(owner.points(prompt))
        except Exception:
            pass
    candidates = sorted({p for p in candidates if MIN_TOKENS <= p < len(prompt)}, reverse=True)
    if not candidates:
        return None

    matched_pos = None
    matched_key = None
    for p in candidates:
        key = make_key(prompt[:p])
        if client.exists(key):
            matched_pos = p
            matched_key = key
            break

    if matched_pos is None or matched_key is None:
        return None

    t0 = time.perf_counter()
    data = client.get(matched_key)
    if data is None:
        return None

    busy = owner._busy()
    st = None
    if owner.free:
        st = owner.free.pop()
    else:
        idle = next((k[1] for k in owner.kept if id(k[1]) not in busy), None)
        if idle is not None:
            owner._drop_kept(idle)
            st = idle

    if st is None:
        return None

    try:
        if not owner._grow(st, len(prompt) + owner.depth + 2):
            owner.free.append(st)
            return None

        p = matched_pos
        device = st.rec.device
        for i, kc_item in enumerate(data["kc"]):
            st.kc[i].k[:p].copy_(kc_item["k"].to(device, non_blocking=True))
            st.kc[i].v[:p].copy_(kc_item["v"].to(device, non_blocking=True))
            if st.kc[i].quantized and "ks" in kc_item:
                st.kc[i].ks[:p].copy_(kc_item["ks"].to(device, non_blocking=True))
                st.kc[i].vs[:p].copy_(kc_item["vs"].to(device, non_blocking=True))
            st.ikc[i][:p].copy_(data["ikc"][i].to(device, non_blocking=True))
            st.pooled[i][:p // st.ratio].copy_(data["pooled"][i].to(device, non_blocking=True))

        mtp_len = data.get("mtp_len", 0)
        if hasattr(st, "mtp_kc") and mtp_len > 0 and "mtp_kc" in data:
            st.mtp_kc.k[:mtp_len].copy_(data["mtp_kc"]["k"].to(device, non_blocking=True))
            st.mtp_kc.v[:mtp_len].copy_(data["mtp_kc"]["v"].to(device, non_blocking=True))
            if st.mtp_kc.quantized and "ks" in data["mtp_kc"]:
                st.mtp_kc.ks[:mtp_len].copy_(data["mtp_kc"]["ks"].to(device, non_blocking=True))
                st.mtp_kc.vs[:mtp_len].copy_(data["mtp_kc"]["vs"].to(device, non_blocking=True))
            st.mtp_ikc[:mtp_len].copy_(data["mtp_ikc"].to(device, non_blocking=True))
            st.mtp_pooled[:mtp_len // st.ratio].copy_(data["mtp_pooled"].to(device, non_blocking=True))

        snap = data["snap"]
        snap_dev = {
            "pos": snap["pos"],
            "rec": snap["rec"].to(device),
            "conv": snap["conv"].to(device),
            "ple_tail": snap["ple_tail"].to(device),
            "ple_history": None if snap.get("ple_history") is None else snap["ple_history"].copy(),
            "mtp_len": snap.get("mtp_len", 0),
        }
        st.restore(snap_dev)

        tail = data.get("tail")
        if tail is not None:
            tail = tail.to(device)

        dt = (time.perf_counter() - t0) * 1000
        logger.info(f"LMCache HIT: restored prefix of {p} tokens in {dt:.1f}ms")
        return st, {"state": snap_dev, "tail": tail}, p
    except Exception as e:
        logger.exception(f"Failed to restore LMCache prefix: {e}")
        owner.free.append(st)
        return None


def async_store(ids: list[int], st, snap: dict, tail: torch.Tensor | None):
    """异步将已 Prefill 完成的前缀推送到 LMCache"""
    if not is_enabled() or len(ids) < MIN_TOKENS:
        return
    pos = len(ids)

    try:
        kc_list = []
        for kc in st.kc:
            item = {"k": kc.k[:pos].cpu(), "v": kc.v[:pos].cpu()}
            if kc.quantized:
                item["ks"] = kc.ks[:pos].cpu()
                item["vs"] = kc.vs[:pos].cpu()
            kc_list.append(item)
        ikc_list = [t[:pos].cpu() for t in st.ikc]
        pooled_list = [t[:pos // st.ratio].cpu() for t in st.pooled]

        mtp_len = snap.get("mtp_len", 0)
        mtp_kc_dict = None
        mtp_ikc = None
        mtp_pooled = None
        if hasattr(st, "mtp_kc") and mtp_len > 0:
            mtp_kc_dict = {"k": st.mtp_kc.k[:mtp_len].cpu(), "v": st.mtp_kc.v[:mtp_len].cpu()}
            if st.mtp_kc.quantized:
                mtp_kc_dict["ks"] = st.mtp_kc.ks[:mtp_len].cpu()
                mtp_kc_dict["vs"] = st.mtp_kc.vs[:mtp_len].cpu()
            mtp_ikc = st.mtp_ikc[:mtp_len].cpu()
            mtp_pooled = st.mtp_pooled[:mtp_len // st.ratio].cpu()

        snap_cpu = {
            "pos": snap["pos"],
            "rec": snap["rec"].cpu(),
            "conv": snap["conv"].cpu(),
            "ple_tail": snap["ple_tail"].cpu(),
            "ple_history": snap.get("ple_history"),
            "mtp_len": mtp_len,
        }
        tail_cpu = tail.cpu() if tail is not None else None

        payload = {
            "pos": pos,
            "kc": kc_list,
            "ikc": ikc_list,
            "pooled": pooled_list,
            "snap": snap_cpu,
            "tail": tail_cpu,
            "mtp_len": mtp_len,
        }
        if mtp_kc_dict is not None:
            payload["mtp_kc"] = mtp_kc_dict
            payload["mtp_ikc"] = mtp_ikc
            payload["mtp_pooled"] = mtp_pooled

        key = make_key(ids)
    except Exception as e:
        logger.warning(f"Failed to prepare LMCache payload: {e}")
        return

    def _worker():
        t0 = time.perf_counter()
        if get_client().put(key, payload):
            dt = (time.perf_counter() - t0) * 1000
            logger.info(f"LMCache STORE: persisted prefix of {pos} tokens in {dt:.1f}ms")

    _EXECUTOR.submit(_worker)
```

---

### 3.2 修改调度逻辑：`prefixes.py`
编辑文件：`/usr/local/lib/python3.12/dist-packages/tensorfold/families/qwen4_exp/cuda/prefixes.py`

也可直接应用补丁文件 `patches/0003-lmcache-cross-node.patch`：

```diff
--- prefixes.py.bak
+++ prefixes.py
@@ -1,5 +1,7 @@
 """Select and retain prompt prefixes without consuming a longer chain when a spare slot can hold a copy."""
 
+from tensorfold import lmcache_bridge
+
 
 def _best(kept, prompt, busy=()):
     return max((k for k in kept if id(k[1]) not in busy and len(k[0]) < len(prompt)
@@ -36,6 +38,13 @@
         owner.kept = [k for k in owner.kept if k[1] is not best[1]
                       or len(k[0]) <= n and best[0][:len(k[0])] == k[0]]
         return best[1], {"state": best[2], "tail": best[3]}, n
+
+    # Remote LMCache lookup if local in-memory kept did not hit
+    if reuse:
+        remote = lmcache_bridge.retrieve(owner, prompt)
+        if remote is not None:
+            return remote
+
     if not owner.free:
         idle = next((k[1] for k in owner.kept if id(k[1]) not in busy), None)
         if idle is None:
@@ -57,3 +66,9 @@
         if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                 all(f is not old for f in owner.free):
             owner.free.append(old)
+
+    # Remote LMCache async store
+    try:
+        lmcache_bridge.async_store(ids, st, snap, tail)
+    except Exception:
+        pass
```

---

## 4. 镜像与持久化部署（升级重做 SOP）

当官方或实验室提供了新的基础镜像（例如新版 `hello/...:latest`）时，按以下步骤快速重新部署：

### 步骤 1：安装依赖并提交新镜像
在 19 和 20 节点上拉取新镜像后，启动临时容器安装 `lmcache` 并提交：
```bash
docker run -d --name tf-temp <NEW_IMAGE> tail -f /dev/null
docker exec tf-temp pip install lmcache
docker commit tf-temp hello/qwen3.8-flash-next-single-dgx-spark-tensorfold:lmcache
docker rm -f tf-temp
```

### 步骤 2：维护宿主机持久化文件与挂载
确保宿主机的部署目录（例如 `/home/clouder/Qwen3.8-Flash-Single/`）存放了修改后的源码：
- `lmcache_bridge.py`
- `prefixes.py`

在 `compose.yaml` 中配置 Volume 映射（双保险机制）：
```yaml
services:
  # 19 号节点额外运行此常驻服务
  lmcache-server:
    image: hello/qwen3.8-flash-next-single-dgx-spark-tensorfold:lmcache
    container_name: lmcache-server
    restart: unless-stopped
    network_mode: host
    command: python3 /usr/local/bin/lmcache_server 0.0.0.0 8998 cpu

  qwen38-flash-next-tf:
    image: hello/qwen3.8-flash-next-single-dgx-spark-tensorfold:lmcache
    container_name: qwen38-flash-next-tf
    restart: unless-stopped
    ipc: host
    network_mode: host
    environment:
      # 开启 LMCache 配置
      - TENSORFOLD_LMCACHE=1
      - LMCACHE_SERVER_HOST=192.168.1.19
      - LMCACHE_SERVER_PORT=8998
      - LMCACHE_MIN_TOKENS=256
      # 其余原有环境变量保持不变...
    volumes:
      - /home/clouder/model-gguf/Qwen3.8-Flash-Next-MLX-4bit-MTP:/model:ro
      - /home/clouder/.cache/tensorfold-qwen38:/cache
      - /home/clouder/.cache/huggingface:/root/.cache/huggingface
      # 持久化注入核心桥接代码
      - /home/clouder/Qwen3.8-Flash-Single/lmcache_bridge.py:/usr/local/lib/python3.12/dist-packages/tensorfold/lmcache_bridge.py:ro
      - /home/clouder/Qwen3.8-Flash-Single/prefixes.py:/usr/local/lib/python3.12/dist-packages/tensorfold/families/qwen4_exp/cuda/prefixes.py:ro
    command: >
      tensorfold serve /model ...
```

### 步骤 3：启动服务
```bash
# 在 19 节点
cd /home/clouder/Qwen3.8-Flash-Single
docker compose up -d

# 在 20 节点
cd /home/clouder/Qwen3.8-Flash-Single
docker compose up -d qwen38-flash-next-tf
```

---

## 5. 验证跨节点复用

通过一段包含系统提示词与长上下文的请求进行端到端验证：

1. **请求打给 19 节点**：
   观察 19 节点日志，确认执行了全量 Prefill 并完成异步存入：
   ```text
   [LMCacheBridge] LMCache STORE: persisted prefix of 2721 tokens in 249.6ms
   ```

2. **相同前缀请求打给 20 节点**：
   观察 20 节点日志与 API 返回，确认跨机命中并直接跳过已缓存 Token：
   ```text
   [LMCacheBridge] LMCache HIT: restored prefix of 2721 tokens in 304.7ms
   ```
   API 返回值中的 `usage.prompt_tokens_details.cached_tokens` 会准确显示命中的 Token 数量（如 `2721`），首字延迟大幅缩短，生成的回复文本语义完全一致。
