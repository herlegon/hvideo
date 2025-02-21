from __future__ import annotations

from argparse import (
    ArgumentParser,
    ArgumentTypeError,
    Namespace,
    RawTextHelpFormatter,
)
import os
import sys

from media.codecs import str_to_video_codec
from utils.p_print import red
from .path_utils import (
    absolute_path,
    is_access_granted,
    path_split,
)


class BoundedInteger:
    def __init__(self, min_value: int, max_value: int) -> None:
        self.min_value: int = min_value
        self.max_value: int = max_value

    def __call__(self, value) -> int:
        try:
            value = int(value)
        except:
            raise ArgumentTypeError(f"Value is not a valid argument.")
        if self.min_value <= value <= self.max_value:
            return value
        raise ArgumentTypeError(
            f"Value is not a valid argument, allowed range: {self.min_value}..{self.max_value}"
        )



def args_parse() -> Namespace:
    parser = ArgumentParser(
        description="Python wrapper for vs_temporal_fix script",
        formatter_class=RawTextHelpFormatter
    )
    parser = ArgumentParser(
        description="Python wrapper of vs_temporal_fix",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="""Input video file.
"""
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="""Output video file. If not specified, it will append a suffix to the input filename.
"""
    )

    parser.add_argument(
        "-suffix",
        "--suffix",
        type=str,
        default="_pytc",
        required=False,
        help="""Suffix used when no output filename is specified.
"""
    )

    # Specific for DNxHR, SAR may be not present though it should be -> resize has to be done
    parser.add_argument(
        "-fsar",
        "--fsar",
        type=str,
        default="",
        help="""SAR metadata may be not present in input video.
Use this one to resize the video and overwrite the one specified in metadata
Keep the height, modify the width.
It should normally never be used as it's really very specific.
format: 4/3 (or 16/9, etc.)
\n"""
    )

    parser.add_argument(
        "-fsar_h",
        "--fsar_h",
        type=str,
        default="",
        help="""SAR metadata may be not present in input video.
Use this one to resize the video and overwrite the one specified in metadata.
Keep the width, modify the height.
It should normally never be used as it's really very specific.
format: 4/3 (or 16/9, etc.)
\n"""
    )

    parser.add_argument(
        "-resize",
        "--resize",
        type=float,
        default=1.,
        help="""Resize the video before applying any filter.
\n"""
    )

    # Seeking
    parser.add_argument(
        "-ss",
        "--ss",
        type=str,
        required=False,
        default='',
        help="""Seeks in this input file to position.
HOURS:MM:SS.MILLISECONDS
Refer to https://ffmpeg.org//ffmpeg.html#Main-options
and https://ffmpeg.org//ffmpeg-utils.html#time-duration-syntax
\n"""
    )
    parser.add_argument(
        "-t",
        "--t",
        type=str,
        required=False,
        default='',
        help="""Limit the duration of data read from the input file.
HOURS:MM:SS.MILLISECONDS
Refer to https://ffmpeg.org//ffmpeg.html#Main-options
\n"""
    )
    parser.add_argument(
        "-to",
        "--to",
        type=str,
        required=False,
        default="",
        help="""Stop reading the input at position.
HOURS:MM:SS.MILLISECONDS
--to and --t are mutually exclusive and --t has priority.
Refer to https://ffmpeg.org//ffmpeg.html#Main-options"
\n"""
    )

    # TensorRT infernece
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="",
        help="""Trt engine filepath
\n"""
    )
    parser.add_argument(
        "-fp32",
        "--fp32",
        action="store_true",
        required=False,
        help="""Inference with fp32 datatype
\n"""
    )
    parser.add_argument(
        "-fp16",
        "--fp16",
        action="store_true",
        required=False,
        help="""Inference with fp16 datatype
\n"""
    )
    parser.add_argument(
        "-bf16",
        "--bf16",
        action="store_true",
        required=False,
        help="""Inference with bf16 datatype
\n"""
    )

    # Encoder
    parser.add_argument(
        "-vcodec",
        "--vcodec",
        choices=list(str_to_video_codec.keys()),
        default='h265',
        required=False,
        help="""Video encoder
\n"""
    )
    parser.add_argument(
        "-pix_fmt",
        "--pix_fmt",
        default="yuv420p10le",
        required=False,
        help="""FFMpeg pix_fmt. rgb/yuv only.
recommended: yuv420p, yuv420p10le, yuv420p12le
\n"""
    )

    parser.add_argument(
        "-preset",
        "--preset",
        choices=[
            'ultrafast',
            'superfast',
            'veryfast',
            'faster',
            'fast',
            'medium',
            'slow',
            'slower',
            'veryslow',
        ],
        default='slow',
        required=False,
        help="""FFmpeg video preset
\n"""
    )

    parser.add_argument(
        "-crf",
        "--crf",
        type=int,
        default=16,
        required=False,
        help="""FFmpeg CRF
\n"""
    )

    parser.add_argument(
        "-tune",
        "--tune",
        choices=[
            'film',
            'animation',
            'grain',
            'stillimage',
            'fastdecode',
            'zerolatency',
        ],
        default='',
        help="""FFmpeg tune setting
\n"""
    )

    parser.add_argument(
        "-ffmpeg",
        "--ffmpeg",
        type=str,
        default="",
        help="""FFmpeg custom options.
\n"""
    )

    # Logger
    parser.add_argument(
        "--log",
        action="store_true",
        required=False,
        default=False,
        help="""(DEV) log in ./logs
\n"""
    )

    # Just for debug
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        default=False,
        help="""(DEV) display additionnal info
\n"""
    )

    arguments: Namespace = parser.parse_args()
    return arguments



def check_args(
    args: Namespace,
    add_suffix: str = ""
) -> tuple[str, str]:
    """Check that input/output media files can be opened/written

        returns absolute path of each media files
    """

    in_media_path: str = absolute_path(args.input)
    if not os.path.isfile(in_media_path):
        sys.exit(red(f"Error: missing input file {in_media_path}"))

    # Use output filepath before verification because it uses the
    #   output directory to store the log file
    out_media_path: str = absolute_path(args.output)
    if not args.output:
        dirname, basename, extension = path_split(in_media_path)
        out_media_path: str = os.path.join(
            dirname, f"{basename}{args.suffix}{add_suffix}{extension}"
        )
    if out_media_path == in_media_path:
        sys.exit(red(f"Error: output file must be different from input file: {out_media_path}"))

    # Verify that the file can be saved
    out_dir: str = path_split(out_media_path)[0]
    if not os.path.isdir(out_dir):
        out_dir_parent = absolute_path(os.path.join(out_dir, os.pardir))
        try:
            os.makedirs(out_dir, exist_ok=True)
        except:
            sys.exit(red(f"Error: no write access to {out_dir_parent}"))

    if not is_access_granted(out_dir, 'w'):
        sys.exit(red(f"Error: no write access to {out_dir}"))

    return in_media_path, out_media_path
