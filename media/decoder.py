
from argparse import Namespace
from dataclasses import dataclass, field
import math
from pprint import pprint
import re
import subprocess
import sys

import numpy as np

from utils.time_conversions import (
    FrameRate,
    frame_to_sexagesimal,
    sexagesimal_to_frame,
)

from .media import (
    ChannelOrder,
    FShape,
    VideoInfo,
)
from utils.p_print import *
from utils.tools import ffmpeg_exe



@dataclass
class VideoStreamInfo:
    filepath: str
    dtype: np.dtype
    c_order: ChannelOrder
    shape: FShape
    nbytes: int
    pix_fmt: str
    sar: tuple[int, int]

    start: str = ""
    to: str = ""
    duration: str = ""
    nframes: int = 0

    metadata: dict[str, str] = field(default_factory=dict)



def _decoder_frame_prop(
    vi: VideoInfo,
    fp32: bool = False
) -> tuple[FShape, np.dtype, ChannelOrder, int] | None:
    """Returns the shape, dtype, channel order and size in bytes
        of a decoded frame
        fp32: force dtype to float32
    """

    out_c_order: ChannelOrder = (
        'rgb' if vi['c_order'] == 'yuv' else vi['c_order']
    )
    out_c_order = 'bgr' if 'bgr' in out_c_order or fp32 else 'rgb'

    in_bpp = vi['bpp']
    if in_bpp > 16:
        in_pix_fmt = vi['pix_fmt']
        sys.exit(f"[E] {in_pix_fmt} is not a supported pixel format (bpp>16)")
        return None

    out_dtype: np.dtype = np.float32
    if not fp32:
        out_dtype = np.uint16 if in_bpp > 8 else np.uint8

    return (
        vi['shape'],
        out_dtype,
        out_c_order,
        math.prod(vi['shape']) * np.dtype(out_dtype).itemsize
    )



def video_stream_info(
    in_vi: VideoInfo,
    args: Namespace,
    debug: bool = False
) -> VideoStreamInfo:

    if not in_vi['is_frame_rate_fixed']:
        sys.exit("[E] variable frame rate is not supported yet")

    count: int = in_vi['frame_count']
    start: int = 0
    frame_rate: FrameRate = in_vi['frame_rate_r']

    # Seek
    start_hms: str = ""
    duration_hms: str = ""
    to_hms: str = ""

    seek_start: str = args.ss
    if seek_start:
        if result := re.search(re.compile(r"^(\d+)f$"), seek_start):
            start = int(result.group(1))
            start_hms = frame_to_sexagesimal(start, frame_rate)
        else:
            start = sexagesimal_to_frame(seek_start)
            start_hms = seek_start

    seek_duration: str = args.t
    seek_end: str = args.to
    if seek_duration != '':
        if result := re.search(re.compile(r"^(\d+)f$"), seek_duration):
            count = int(result.group(1))
            duration_hms = frame_to_sexagesimal(count, frame_rate)
        else:
            count = sexagesimal_to_frame(seek_duration)
            duration_hms = seek_duration

    elif seek_end != '':
        if result := re.search(re.compile(r"^(\d+)f$"), seek_end):
            to = int(result.group(1))
            to_hms = frame_to_sexagesimal(to, frame_rate)
        else:
            to_hms = seek_end
            to = sexagesimal_to_frame(seek_end)
        count = to - start


    frame_shape, dtype, _, nbytes = _decoder_frame_prop(in_vi)
    vsi: VideoStreamInfo = VideoStreamInfo(
        filepath=in_vi['filepath'],
        dtype=dtype,
        c_order='rgb',
        shape=frame_shape,
        nbytes=nbytes,
        pix_fmt='rgb24' if in_vi["bpp"] < 8 else 'rgb48',
        sar=in_vi['sar'],

        start=start_hms,
        to=to_hms,
        duration=duration_hms,
        nframes=count,
    )

    if debug:
        print(lightcyan("Decoder pipe stream:"))
        pprint(vsi)

        print(f"  shape: {vsi.shape}")
        print(f"  pixel format: {vsi.pix_fmt}")
        print(f"  dtype: {vsi.dtype}")
        print(f"  nbytes: {vsi.nbytes}")
        print(f"  channel order: {vsi.c_order}")
        print(f"  sar: {vsi.sar}")
        print(f"  start: {vsi.start}")
        print(f"  to: {vsi.to}")
        print(f"  duration: {vsi.duration}")
        print(f"  frames: {vsi.nframes}")



def decoder_subprocess(
    vsi: VideoStreamInfo,
    debug: bool = False
) -> subprocess.Popen:

    d_command: list[str] = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel", "warning",
        "-nostats",
    ]
    if vsi.start:
        d_command.extend(["-ss", vsi.start])

    d_command.extend(["-i", vsi.filepath])

    if vsi.to:
        d_command.extend(["-to", vsi.to])
    elif vsi.duration:
        d_command.extend(["-t", vsi.duration])

    d_command.extend([
        "-f", "image2pipe",
        "-pix_fmt", vsi.pix_fmt,
        "-vcodec", "rawvideo",
        "-"
    ])

    if debug:
        print(lightgreen(f"[V][D] FFmpeg command:"), ' '.join(d_command))

    # Open subprocess
    sub_process: subprocess.Popen
    try:
        sub_process = subprocess.Popen(
            d_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        raise ValueError(f"[V][D] Unexpected error: {type(e)}", flush=True)

    return sub_process
