from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
import os
from pprint import pprint
import re
import signal
import sys
import torch
from warnings import warn

from core import (
    DecoderThread,
    EncoderThread,
    ProgressThread,
    run_threads,
)

from core.t_inference import InferenceThread
from media import (
    args_to_encoder_settings,
    DecoderSeek,
    EncoderSettings,
    get_seek,
    MediaInfo,
    open_media_file,
    VideoInfo,
    VideoPipeInfo,
    video_decoder_pipe_info,
    video_encoder_pipe_info,
)
from pynnlib import (
    Idtype,
)
from utils.arg_parse import args_parse, check_args
from utils.logger import logger, set_logger_settings
from utils.p_print import *
from utils.path_utils import absolute_path
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
    in_media_info: MediaInfo = open_media_file(
        vi_fp, verbose=True, debug=arguments.debug
    )
    in_vi: VideoInfo = in_media_info['video']
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



    # Resize before inference
    #-------------------------------------------------------------------------
    if arguments.resize != 1:
        do_resize = True
        resize_factor = arguments.resize
        f_vi['shape'] = (int(resize_factor * h), int(resize_factor * w), c)



    # Resize before inference
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

    total_frames: int = out_vi['frame_count']



    # Decoder thread
    #-------------------------------------------------------------------------
    d_vpi: VideoPipeInfo = video_decoder_pipe_info(
        in_vi, seek=seek, debug=arguments.debug
    )
    if debug:
        print(lightcyan("Decoder Video pipe"))
        pprint(d_vpi)

    d_thread: DecoderThread = DecoderThread(
        name="decoder",
        video_pipe_info=d_vpi,
        tensor_dtype=torch.float16
    )



    # Encoder thread
    #-------------------------------------------------------------------------
    e_vpi: VideoPipeInfo = video_encoder_pipe_info(
        out_vi=out_vi, e_settings=e_settings, debug=arguments.debug
    )
    if debug:
        print(lightcyan("Encoder Video pipe"))
        pprint(e_vpi)
    e_thread: EncoderThread = EncoderThread(
        name="encoder",
        video_info=out_vi,
        video_pipe_info=e_vpi,
        e_settings=e_settings,
        in_media_info=in_media_info,
        debug=arguments.debug
    )



    # Image Writer thread
    #-------------------------------------------------------------------------
    # ...



    # Tensor inference thread
    #-------------------------------------------------------------------------
    i_thread: InferenceThread | None= None
    if arguments.model:
        model_filepath: str = absolute_path(arguments.model)
        i_thread = InferenceThread(name="trt_inference")
        i_dtype: Idtype = 'fp16'
        if arguments.fp32:
            i_dtype = 'fp32'
        elif arguments.fp16:
            i_dtype = 'fp16'
        elif arguments.bf16:
            i_dtype = 'bf16'

        i_thread.initialize(
            filepath=model_filepath,
            device="cuda:0",
            dtype=i_dtype,
            prescale=f_vi['shape'] if do_resize else None
        )
        d_thread.set_consumer(i_thread)
        e_thread.set_producer(i_thread)

        i_thread.set_producer(d_thread)
        i_thread.set_consumer(e_thread)



    # Filters
    #-------------------------------------------------------------------------
    # Not filters if TRT inference before
    f_thread = None
    if arguments.model:
        # ...


        d_thread.set_consumer(f_thread)
        e_thread.set_producer(f_thread)
        # f_thread.set_producer(d_thread)
        # f_thread.set_consumer(e_thread)



    # Progress bar
    #-------------------------------------------------------------------------
    progress_thread: ProgressThread = ProgressThread(total=total_frames)
    e_thread.set_progress_thread(progress_thread)


    # Main loop
    #-------------------------------------------------------------------------
    run_threads(
        d_thread=d_thread,
        e_thread=e_thread,
        i_threads=[i_thread, f_thread],
        progress_thread=progress_thread
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()

