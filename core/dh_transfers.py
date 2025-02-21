import time
import numpy as np
import torch
from torch import Tensor
from utils.p_print import red


def htod_transfer(
    host_mem: Tensor,
    img_buffer: Tensor,
    img_dtype: torch.dtype,
    img_shape: tuple[int, int, int],
    cuda_stream: torch.cuda.Stream,
    device: str = "cuda:0",
) -> Tensor:
    host_mem.copy_(img_buffer, non_blocking=True)
    d_tensor: Tensor = host_mem.to(device=device)

    time.sleep(0.0001)
    cuda_stream.synchronize()

    if not d_tensor.is_contiguous():
        raise ValueError(red("[E] Tensor is not contiguous cannot use view"))
    return d_tensor.view(dtype=img_dtype).view(img_shape)



def dtoh_transfer(
    host_mem: Tensor,
    d_img: Tensor,
    cuda_stream: torch.cuda.Stream,
) -> np.ndarray:
    out_shape: tuple[int, int, int] = d_img.shape
    host_mem.copy_(d_img.ravel(), non_blocking=True)

    time.sleep(0.0001)
    cuda_stream.synchronize()

    return host_mem.view(out_shape).numpy()



def dtoh_transfer_sync(
    host_mem: Tensor,
    d_img: Tensor,
) -> np.ndarray:
    """Synchronous transfer from GPU to CPU.
    """
    out_shape: tuple[int, int, int] = d_img.shape
    host_mem.copy_(d_img.ravel(), non_blocking=False)
    return host_mem.view(out_shape).numpy()



def htod_transfer_sync(
    host_mem: Tensor,
    h_img: Tensor,
    device: str = "cuda:0",
) -> Tensor:
    host_mem.copy_(
        np.ascontiguousarray(h_img), non_blocking=False
    )
    d_img: Tensor = host_mem.to(device=device)
    return d_img
