from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import cupy as cp
import numpy as np
import torch

from media.media import FShape


pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

if torch.cuda.is_available():
    import cupy as cp

    def set_cupy_cuda_device(device: str = "cuda:0") -> None:
        if not torch.cuda.is_available():
            print(f"[E] No cuda device found, cannot set {device}")
            return
        device_no: int = 0
        try:
            device_no = int(device.split(":")[1])
        except:
            pass
        # print(f"[I] Use cuda device {device_no}")
        cp.cuda.runtime.setDevice(device_no)
else:
    def set_cupy_cuda_device(device: str = "cuda:0") -> None:
        pass


@dataclass
class HostDeviceMemory:
    host: Any
    device: Any
    nbytes: int



def allocate_memory(
    shape: FShape,
    dtype: np.dtype,
    stream: Any | None = None
) -> HostDeviceMemory:
    nbytes = math.prod(shape) * np.dtype(dtype).itemsize

    if stream is not None:
        with stream:
            h_mem = cp.cuda.alloc_pinned_memory(nbytes)
            # d_mem = cp.cuda.MemoryPointer(cp.cuda.Memory(nbytes), 0)
            d_mem = cp.empty((math.prod(shape),), dtype)
    else:
        h_mem = cp.cuda.alloc_pinned_memory(nbytes)
        # d_mem = cp.cuda.MemoryPointer(cp.cuda.Memory(nbytes), 0)
        d_mem = cp.empty((math.prod(shape),), dtype)

    # print(f"[V] allocated: host={h_mem}, device={d_mem}, {nbytes}")
    return HostDeviceMemory(
        host=h_mem,
        device=d_mem,
        nbytes=nbytes
    )

