from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor

from core import NnFrame
from utils.p_print import *



def gpu_resize_(
    frame: NnFrame,
    scale_factor: Optional[float | int | list[float]] = None,
    out_size: Optional[list[int, int]] = None,
    interpolation_method: str = "bicubic",
) -> NnFrame:
    in_x: Tensor = frame.tensor
    if isinstance(scale_factor, float):
        out_size: list[int, int] = list(
            [int(((x * scale_factor) // 2 ) * 2) for x in in_x.shape[2:]]
        )

    in_x = in_x.to(dtype=torch.float16)
    out_x: Tensor = F.interpolate(
        input=in_x,
        size=out_size,
        mode=interpolation_method,
        align_corners=False,
        antialias=bool(interpolation_method in ('bilinear', 'bicubic'))
    )
    frame.tensor = torch.clamp(out_x.contiguous(), 0, 1.0)

    return frame



def gpu_resize_to_(
    frame: NnFrame,
    out_size: tuple[int, int],
    interpolation_method: str = "bicubic",
) -> NnFrame:
    """outsize: (h, w)"""
    return gpu_resize_(
        frame,
        out_size=out_size,
        interpolation_method=interpolation_method
    )
