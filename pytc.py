

from argparse import Namespace
from copy import deepcopy
from pprint import pprint
import signal
import sys

from threads.cupy_utils import set_cupy_cuda_device
from utils.arg_parse import args_parse, check_args
from media.encoder import (
    EncoderSettings,
    args_to_encoder_settings,
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





def main():
    # Parse arguments
    arguments: Namespace = args_parse()
    debug: bool = arguments.debug

    # Check arguments and get filepaths
    vi_fp, vo_fp = check_args(arguments)

    # Logger
    set_logger_settings(args=arguments, out_media_fp=vi_fp)
    logger.debug(f"Python executable dir: {sys.executable}")
    logger.debug(f"arguments: {sys.argv}")
    print(lightcyan(f"Input video file:"), f"{vi_fp}")
    logger.debug(f"input: {vi_fp}")

    # Open media file
    in_vi: VideoInfo = open_media_file(
        vi_fp, verbose=True, debug=arguments.debug
    )['video']

    # Generate an output video info and encoder settings
    out_vi: VideoInfo = deepcopy(in_vi)
    out_vi['filepath'] = vo_fp
    e_settings: EncoderSettings = args_to_encoder_settings(
        args=arguments, video_info=out_vi
    )
    # filepath's extension may have been patched
    # depending on the codec
    out_vi['filepath'] = e_settings.filepath
    out_fp: str = out_vi['filepath']
    print(lightcyan(f"Output video file:"), f"{out_fp}")
    logger.debug(f"output: {out_fp}")
    if debug:
        print(lightcyan("Encoder settings:"))
        pprint(e_settings)

    # update output vi if needed
    out_vi.update({
        'metadata': {
            'pytc': current_datetime_str()
        }
    })

    set_cupy_cuda_device()
    # Create a decoder thread

    # Create an encoder thread
    # Create an image writer

    # Create a progress bar

    #



    pass




if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()

