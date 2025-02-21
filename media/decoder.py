
from argparse import Namespace
import math
import numpy as np
from pprint import pprint
import re
import subprocess
import sys
from warnings import warn

from utils.time_conversions import (
    FrameRate,
    frame_to_sexagesimal,
    sexagesimal_to_frame,
)
from utils.p_print import *
from utils.tools import ffmpeg_exe

from .media import (
    ChannelOrder,
    FShape,
    VideoInfo,
)
from .utils import DecoderSeek, VideoPipeInfo



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


def get_seek(
    in_vi: VideoInfo,
    args: Namespace,
) -> DecoderSeek:

    if not in_vi['is_frame_rate_fixed']:
        sys.exit("[E] variable frame rate is not supported yet")

    count: int = in_vi['frame_count']
    start: int = 0
    to: int = in_vi['frame_count']
    frame_rate: FrameRate = in_vi['frame_rate_r']

    # Seek
    start_hms: str = ""
    duration_hms: str = ""
    to_hms: str = ""

    seek_start: str = args.ss
    if seek_start:
        if result := re.search(re.compile(r"^(\d+)f$"), seek_start):
            start = int(result.group(1))
        else:
            start = sexagesimal_to_frame(seek_start, frame_rate)
        start_hms = frame_to_sexagesimal(start, frame_rate)
        count = to - start

    if start >= in_vi['frame_count']:
        raise ValueError(red(f"Erroneous seek start: {start} >= {in_vi['frame_count']}"))

    seek_duration: str = args.t
    seek_end: str = args.to
    if seek_duration != '':
        if result := re.search(re.compile(r"^(\d+)f$"), seek_duration):
            count = int(result.group(1))
        else:
            count = sexagesimal_to_frame(seek_duration, frame_rate)
        duration_hms = frame_to_sexagesimal(count, frame_rate)

    elif seek_end != '':
        if result := re.search(re.compile(r"^(\d+)f$"), seek_end):
            to = int(result.group(1))
        else:
            to = sexagesimal_to_frame(seek_end, frame_rate)
        to_hms = frame_to_sexagesimal(to, frame_rate)
        count = to - start

    if start + count > in_vi['frame_count']:
        warn(yellow(f"Erroneous seek, reducing <- improve this message"))
        count = in_vi['frame_count'] - start
        to_hms = ""
        duration_hms = ""

    return DecoderSeek(
        start=start_hms,
        to=to_hms,
        duration=duration_hms,
        f_no=start,
        count=count,
    )



def video_decoder_pipe_info(
    in_vi: VideoInfo,
    seek: DecoderSeek,
    debug: bool = False
) -> VideoPipeInfo:

    frame_shape, dtype, _, nbytes = _decoder_frame_prop(in_vi)
    vpi: VideoPipeInfo = VideoPipeInfo(
        filepath=in_vi['filepath'],
        dtype=dtype,
        c_order='rgb',
        shape=frame_shape,
        nbytes=nbytes,
        pix_fmt='rgb48' if in_vi['bpp'] > 8 else 'rgb24',

        start=seek.start,
        to=seek.to,
        duration=seek.duration,
        f_no=seek.f_no,
        nframes=seek.count,
    )

    if debug:
        print(lightcyan("Decoder pipe:"))
        pprint(vpi)
        print(f"  shape: {vpi.shape}")
        print(f"  pixel format: {vpi.pix_fmt}")
        print(f"  dtype: {vpi.dtype}")
        print(f"  nbytes: {vpi.nbytes}")
        print(f"  channel order: {vpi.c_order}")
        print(f"  start: {vpi.start}")
        print(f"  to: {vpi.to}")
        print(f"  duration: {vpi.duration}")
        print(f"  frames: {vpi.nframes}")

    return vpi



def decoder_subprocess(
    vpi: VideoPipeInfo,
    debug: bool = False
) -> subprocess.Popen:

    d_command: list[str] = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel", "warning",
        "-nostats",
    ]
    if vpi.start:
        d_command.extend(["-ss", vpi.start])

    d_command.extend(["-i", vpi.filepath])

    if vpi.to:
        d_command.extend(["-to", vpi.to])
    elif vpi.duration:
        d_command.extend(["-t", vpi.duration])

    d_command.extend([
        "-f", "image2pipe",
        "-pix_fmt", vpi.pix_fmt,
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
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        raise ValueError(f"[V][D] Unexpected error: {type(e)}", flush=True)

    return sub_process
