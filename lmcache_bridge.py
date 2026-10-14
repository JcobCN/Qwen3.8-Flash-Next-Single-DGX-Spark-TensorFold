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
