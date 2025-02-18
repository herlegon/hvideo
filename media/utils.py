from dataclasses import dataclass, field
import numpy as np
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
    It's used to generate the processes' commands
    """
    filepath: str
    dtype: np.dtype
    c_order: ChannelOrder
    shape: FShape
    nbytes: int
    pix_fmt: str

    start: str = ""
    to: str = ""
    duration: str = ""

    f_no: int = 0
    nframes: int = 0


