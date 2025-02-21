from argparse import Namespace
import os
from pprint import pprint
import signal
import sys
import torch

from core import (
    ImgReaderThread,
    ImgWriterThread,
    InferenceThread,
    ProgressThread,
    run_threads,
)

from pynnlib import (
    Idtype,
)
from utils.arg_parse import args_parse
from utils.logger import logger, set_logger_settings
from utils.p_print import *
from utils.path_utils import absolute_path, is_access_granted, path_split
try:
    import winsound
except ImportError:
    pass



def main():
    # Parse arguments
    #-------------------------------------------------------------------------
    arguments: Namespace = args_parse()
    debug: bool = arguments.debug


    # Check arguments and get filepaths
    #-------------------------------------------------------------------------
    if not arguments.model:
        raise ValueError(red(f"[E] missing model filepath"))

    in_dir: str = absolute_path(arguments.input)
    if not os.path.isdir(in_dir):
        sys.exit(red(f"Error: not a valid folder {in_dir}"))

    if arguments.output:
        out_dir: str = absolute_path(arguments.output)
        os.makedirs(out_dir, exist_ok=True)
        if not is_access_granted(out_dir, 'w'):
            sys.exit(red(f"Error: no write access to {out_dir}"))
    else:
        out_dir = in_dir


    # Logger
    #-------------------------------------------------------------------------
    set_logger_settings(args=arguments, out_media_fp=in_dir)
    logger.debug(f"Python executable dir: {sys.executable}")
    logger.debug(f"arguments: {sys.argv}")
    print(lightcyan(f"Input directory:"), f"{in_dir}")
    logger.debug(f"input: {in_dir}")


    # Get all png filepath in directory
    #-------------------------------------------------------------------------
    # in a far future, create a generator rather than listing files
    in_frames: tuple[str] = tuple(
        [
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if f.endswith(".png")
        ]
    )
    total_frames: int = len(in_frames)


    # Resize before inference
    #-------------------------------------------------------------------------
    resize_factor: float = 1
    do_resize: bool = False
    if arguments.resize != 1:
        do_resize = True
        resize_factor = arguments.resize


    # Image Writer thread
    #-------------------------------------------------------------------------
    out_frames: list[str] = []
    if out_dir != in_dir and not arguments.suffix:
        out_frames = list(
            [os.path.join(out_dir, os.path.basename(fp)) for fp in in_frames]
        )
    else:
        suffix = arguments.suffix
        if not suffix:
            suffix = "_lr"
        for fp in in_frames:
            _, basename, ext = path_split(fp)
            out_frames.append(
                os.path.join(out_dir, f"{basename}{suffix}{ext}")
            )

    if arguments.debug:
        for in_fp, out_fp in zip(in_frames, out_frames):
            print(f"{in_fp} -> {out_fp}")

    e_thread: ImgWriterThread = ImgWriterThread(
        name="img_writer",
        filepaths=out_frames,
    )


    # Tensor inference thread
    #-------------------------------------------------------------------------
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
        prescale=resize_factor if do_resize else None
    )
    e_thread.set_producer(i_thread)
    i_thread.set_consumer(e_thread)



    # Image Reader thread
    #-------------------------------------------------------------------------
    d_thread: ImgReaderThread = ImgReaderThread(
        name="img_writer",
        filepaths=in_frames,
        tensor_dtype=i_dtype
    )
    d_thread.set_consumer(i_thread)
    i_thread.set_producer(d_thread)



    # Progress bar
    #-------------------------------------------------------------------------
    progress_thread: ProgressThread = ProgressThread(total=total_frames)
    e_thread.set_progress_thread(progress_thread)



    # Main loop
    #-------------------------------------------------------------------------
    run_threads(
        d_thread=d_thread,
        e_thread=e_thread,
        i_threads=(i_thread,),
        progress_thread=progress_thread
    )

    torch.cuda.empty_cache()

    if sys.platform == "win32":
        winsound.Beep(frequency=440, duration=200)



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()

