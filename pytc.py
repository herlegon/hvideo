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
    CudaTemporalInferenceThread,
    CudaSegInferenceThread,
)

from core.t_trt_inference import InferenceThread
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
    NnModel,
    NnFrameworkType,
    nnlib,
)
from utils.arg_parse import args_parse, check_args
from utils.logger import logger, set_logger_settings
from utils.p_print import *
from utils.path_utils import absolute_path
from utils.time_conversions import current_datetime_str
try:
    import winsound
except ImportError:
    pass



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

    step_no: int = 1

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
        # resize width
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
        print(f" {step_no}. resize to {f_vi['shape'][1]}x{f_vi['shape'][0]}")
        step_no += 1

    #   resize with sar. Default: resize height
    if not do_resize_with_fsar and any([x != 1 for x in f_vi['sar']]):
        do_resize = True
        f_vi['shape'] = (int(h * f_vi['sar'][0] / f_vi['sar'][1]), w, c)
        print(f" {step_no}. resize to {f_vi['shape'][1]}x{f_vi['shape'][0]}")
        step_no += 1

    if do_resize:
        warn(yellow(f"[W] TODO: validate resize: {in_vi['shape']} -> {f_vi['shape']}"))


    # Resize before inference
    #-------------------------------------------------------------------------
    if arguments.resize != 1:
        do_resize = True
        resize_factor = arguments.resize
        h, w, c = f_vi['shape']
        f_vi['shape'] = (int(resize_factor * h + 0.5), int(resize_factor * w + 0.5), c)

    elif arguments.resize_to:
        do_resize = True
        w, h = arguments.resize_to.split("x")
        f_vi['shape'] = (int(h), int(w), f_vi['shape'][-1])

    pre_resize_shape: list | None = None
    if do_resize:
        pre_resize_shape = f_vi['shape']
        print(f" {step_no}. resize to {f_vi['shape'][1]}x{f_vi['shape'][0]}")
        step_no += 1


    # Scale factor of the filters
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #-------------------------------------------------------------------------
    #   The scale of the filter MUST BE 1x. If not,
    #   the size has to be modified here
    f_scale: float = 1.
    if f_scale != 1:
        f_vi['shape'] = (int(f_scale * h), int(f_scale * w), c)


    # TensorRT inference
    #-------------------------------------------------------------------------
    i_dtype = 'fp32'
    if arguments.model:
        model_filepath: str = absolute_path(arguments.model)
        print(lightcyan("Engine:"), os.path.basename(model_filepath))
        trt_model: NnModel = nnlib.open(model_filepath, device="cuda:0")
        if trt_model.framework.type != NnFrameworkType.TENSORRT:
            raise ValueError(red(f"[E] {model_filepath} is not a TensorRT engine"))
        h, w, c = f_vi['shape']
        f_vi['shape'] = (int(trt_model.scale * h), int(trt_model.scale * w), c)
        if arguments.debug:
            print(trt_model)

        i_dtype: Idtype = 'fp16'
        if arguments.fp32:
            i_dtype = 'fp32'
        elif arguments.fp16:
            i_dtype = 'fp16'
        elif arguments.bf16:
            i_dtype = 'bf16'

        if i_dtype not in trt_model.dtypes:
            i_dtype = 'fp32'

        print(f" {step_no}. TRT inference, scale: {trt_model.scale}, dtype: {i_dtype}")
        step_no += 1

    h, w, c = f_vi['shape']

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
    if arguments.model:
        out_vi.update({
            'metadata': {
                'model': os.path.basename(model_filepath)
            }
        })

    # known issue: wrong info is codec/pixfmt in custom params
    print(
        lightcyan("Encoder:"),
        f"{out_vi['shape'][1]}x{out_vi['shape'][0]}, {e_settings.vcodec.value}, {e_settings.pix_fmt}"
    )
    step_no += 1


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


    # Segmentation
    #-------------------------------------------------------------------------
    s_thread = None
    if not arguments.model:
        i_dtype = 'fp16'
        s_thread = CudaSegInferenceThread(
            name="segmentation",
            debug=arguments.debug,
        )
        s_thread.initialize(
            video_info=f_vi,
            model=None,
            device="cuda:0",
            dtype=i_dtype,
            prescale=pre_resize_shape,
        )

        e_thread.set_producer(s_thread)
        s_thread.set_consumer(e_thread)


    # Tensor inference thread
    #-------------------------------------------------------------------------
    i_thread: InferenceThread | None= None
    if arguments.model:
        model_filepath: str = absolute_path(arguments.model)
        trt_model: NnModel = nnlib.open(model_filepath, device="cuda:0")
        if trt_model.framework.type != NnFrameworkType.TENSORRT:
            raise ValueError(red(f"[E] {model_filepath} is not a TensorRT engine"))

        i_thread = InferenceThread(name="trt_inference", debug=arguments.debug)
        i_thread.initialize(
            model=trt_model,
            device="cuda:0",
            dtype=i_dtype,
            prescale=pre_resize_shape
        )
        i_thread.set_consumer(e_thread)


    # Decoder thread
    #-------------------------------------------------------------------------
    d_vpi: VideoPipeInfo = video_decoder_pipe_info(
        in_vi, seek=seek, debug=arguments.debug
    )
    if debug:
        print(lightcyan("Decoder Video pipe"))
        pprint(d_vpi)

    # Use the dtype of the inference task to avoid a useless conversion
    d_thread: DecoderThread = DecoderThread(
        name="decoder",
        video_pipe_info=d_vpi,
        tensor_dtype=i_dtype
    )

    if i_thread is not None:
        d_thread.set_consumer(i_thread)
        # i_thread.set_producer(d_thread)
        e_thread.set_producer(d_thread)

    elif s_thread is not None:
        d_thread.set_consumer(s_thread)
        s_thread.set_producer(d_thread)


    # Progress bar
    #-------------------------------------------------------------------------
    total_frames: int = out_vi['frame_count']
    progress_thread: ProgressThread = ProgressThread(total=total_frames)
    e_thread.set_progress_thread(progress_thread)


    # Main loop
    #-------------------------------------------------------------------------
    elapsed = run_threads(
        d_thread=d_thread,
        e_thread=e_thread,
        i_threads=(i_thread, s_thread),
        progress_thread=progress_thread,
        total_frames=out_vi['frame_count'],
        verbose=arguments.debug
    )

    print(f"elapsed: {elapsed:.02f}s ({total_frames/elapsed:.02f}fps)")

    torch.cuda.empty_cache()

    if sys.platform == "win32":
        winsound.Beep(frequency=440, duration=200)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()

