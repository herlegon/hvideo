from dataclasses import dataclass
import numpy as np
import torch

from utils.time_conversions import FrameRate
from .media import ChannelOrder, FShape


@dataclass
class DecoderSeek:
    start: str = ""
    to: str = ""
    duration: str = ""

    f_no: int = 0
    count: int = 0


@dataclass
class VideoPipeInfo:
    """This structure contains all info about frames coming from a decode process
    or sent to an encoder process.
    It's used to generate a processe's command
    """
    filepath: str
    dtype: np.dtype | torch.dtype
    c_order: ChannelOrder
    shape: FShape
    nbytes: int
    pix_fmt: str

    start: str = ""
    to: str = ""
    duration: str = ""

    f_no: int = 0
    nframes: int = 0



def clean_fcomplex(line: str) -> str:
    cleaned: str = line
    for c in ('\\', '\"', ' ', '\r', '\n'):
        cleaned = cleaned.replace(c, '')
    return cleaned.strip()


