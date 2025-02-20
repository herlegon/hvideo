

from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
from pprint import pprint
import re
import signal
import sys
from warnings import warn

import torch

from core.cp_utils import set_cupy_cuda_device
from core.t_decoder import DecoderThread
from media.decoder import VideoPipeInfo, get_seek, video_decoder_pipe_info
from media.utils import DecoderSeek
from utils.arg_parse import args_parse, check_args
from media.encoder import (
    EncoderSettings,
    args_to_encoder_settings,
    video_encoder_pipe_info,
)
from utils.logger import (
    logger,
    set_logger_settings,
)
from media.media import (
    open_media_file,
    VideoInfo,
)
from utils.p_print import *
from utils.time_conversions import current_datetime_str


# Keep here to fasten modifications
DEFAULT_H265_PARAMS: str = """
    -preset slow
    -crf 16
    -profile:v main422-10
    -x265-params sao=0
"""

DEFAULT_HVEC_NVENC_PARAMS: str = """
    -profile:v main
    -b_ref_mode disabled
    -tag:v hvc1
    -g 30
    -preset p7
    -tune hq
    -rc constqp
    -qp 17
    -rc-lookahead 20
    -spatial_aq 1
    -aq-strength 15
    -b:v 0
"""


def main():
    # Parse arguments
    #-------------------------------------------------------------------------
    arguments: Namespace = args_parse()
    debug: bool = arguments.debug


    # Add some default encoder parameters
    #-------------------------------------------------------------------------
    if not arguments.ffmpeg:
        if arguments.vcodec == 'h265':
            arguments.ffmpeg = DEFAULT_H265_PARAMS
        elif arguments.vcodec == 'hevc_nvenc':
            arguments.ffmpeg = DEFAULT_HVEC_NVENC_PARAMS


    # Check arguments and get filepaths
    #-------------------------------------------------------------------------
    vi_fp, vo_fp = check_args(arguments)


    # Logger
    #-------------------------------------------------------------------------
    set_logger_settings(args=arguments, out_media_fp=vi_fp)
    logger.debug(f"Python executable dir: {sys.executable}")
    logger.debug(f"arguments: {sys.argv}")
    print(lightcyan(f"Input video file:"), f"{vi_fp}")
    logger.debug(f"input: {vi_fp}")


    # Open media file, create the input info
    #-------------------------------------------------------------------------
    in_vi: VideoInfo = open_media_file(
        vi_fp, verbose=True, debug=arguments.debug
    )['video']
    seek: DecoderSeek = get_seek(in_vi=in_vi, args=arguments)


    # Video info of the filter
    #-------------------------------------------------------------------------
    f_vi: VideoInfo = deepcopy(in_vi)
    f_vi['frame_count'] = seek.count

    #   Update output video size if needed when SAR is not 1:1
    do_resize: bool = False
    h, w, c = f_vi['shape']
    if arguments.fsar and arguments.fsar_h:
        sys.exit(red("[E] fsar and fsar_h cannot be used together"))
    do_resize_with_fsar: bool = arguments.fsar or arguments.fsar_h
    if do_resize_with_fsar:
        # resize weidth
        if result := re.search(re.compile(r"^(\d+)\/(\d+)$"), arguments.fsar):
            f_vi['shape'] = (
                h,
                int(w * int(result.group(2)) / int(result.group(1)) + 0.5),
                c
            )
        # resize height
        if result := re.search(re.compile(r"^(\d+)\/(\d+)$"), arguments.fsar_h):
            f_vi['shape'] = (
                int(h * int(result.group(1)) / int(result.group(2)) + 0.5),
                w,
                c
            )

    #   resize with sar. Default: resize height
    if not do_resize_with_fsar and any([x != 1 for x in f_vi['sar']]):
        do_resize = True
        f_vi['shape'] = (int(h * f_vi['sar'][0] / f_vi['sar'][1]), w, c)

    if do_resize:
        warn(yellow(f"[W] TODO: validate resize: {in_vi['shape']} -> {f_vi['shape']}"))


    #   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #   The scale of the filter MUST BE 1x. If not,
    #   the size has to be modified here
    f_scale: float = 1.
    if f_scale != 1:
        f_vi['shape'] = (int(f_scale * h), int(f_scale * w), c)



    # Output video info and encoder settings
    #-------------------------------------------------------------------------
    out_vi: VideoInfo = deepcopy(f_vi)
    out_vi['filepath'] = vo_fp
    e_settings: EncoderSettings = args_to_encoder_settings(
        args=arguments, vi=out_vi
    )
    # filepath's extension may have been patched depending on the codec
    out_vi['filepath'] = e_settings.filepath
    out_fp: str = out_vi['filepath']
    print(lightcyan(f"Output video file:"), f"{out_fp}")
    logger.debug(f"output: {out_fp}")
    if debug:
        print(lightcyan("Encoder settings:"))
        pprint(e_settings)
    # optional, may be customized
    out_vi.update({
        'metadata': {
            'pytc': current_datetime_str()
        }
    })


    # Decoder thread
    #-------------------------------------------------------------------------
    #   create a list of functions
    functions: list[Callable] = []
    # if do_resize:
    #     functions.append(gpu_resize_)
    d_vpi: VideoPipeInfo = video_decoder_pipe_info(
        in_vi, seek=seek, debug=arguments.debug
    )

    if debug:
        print(lightcyan("Decoder Video pipe"))
        pprint(d_vpi)


    d_thread: DecoderThread = DecoderThread(
        video_pipe_info=d_vpi,
        tensor_dtype=torch.float16
    )

    # Encoder thread
    #-------------------------------------------------------------------------
    e_vpi: VideoPipeInfo = video_encoder_pipe_info(
        out_vi, e_settings=e_settings, debug=arguments.debug
    )
    if debug:
        print(lightcyan("Encoder Video pipe"))
        pprint(e_vpi)


    # Image Writer thread
    #-------------------------------------------------------------------------
    # Create an image writer

    # Create a progress bar

    #



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()

