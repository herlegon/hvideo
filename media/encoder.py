from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import math
from pprint import pprint
import re
import subprocess
import sys

import numpy as np

from utils.p_print import lightgreen, red
from utils.path_utils import get_extension
from utils.tools import ffmpeg_exe
from .codecs import (
    vcodec_to_extension,
    VideoCodec,
    str_to_video_codec,
)
from .media import (
    MediaInfo,
    VideoInfo,
)
from .utils import VideoPipeInfo, clean_fcomplex
from .pxl_fmt import PIXEL_FORMAT



def clean_ffmpeg_params(data: str) -> str:
    for c in ['\"', '\r', '\n', '\t']:
        data = data.replace(c, '')
    # avoid too many spaces when debugging
    for _ in range(4):
        data = data.replace('  ', ' ')
    return data



@dataclass
class FFv1Settings:
    level: int = 1
    coder: int = 1
    context: int = 1
    g: int = 1
    threads: int = 8


@dataclass
class H264Settings:
    pass


@dataclass
class H265Settings:
    pass


@dataclass
class DNxHRSettings:
    profile: str | None = None

CodecSettings = FFv1Settings | H264Settings | H265Settings | DNxHRSettings

@dataclass
class ColorSettings:
    colorspace: str | None = 'bt709'
    color_primaries: str | None = 'bt709'
    color_trc: str | None = 'bt709'
    color_range: str | None = 'tv'


@dataclass(slots=True)
class EncoderSettings:
    filepath: str
    # Complex filters
    keep_sar: bool = False
    size: tuple[int, int] | None = None
    resize_algo: str = ''
    add_borders: bool = False
    # Encoder
    vcodec: VideoCodec = VideoCodec.H264
    pix_fmt: str | None = 'yuv420p'
    preset: str | None = 'medium'
    tune: str | None = None
    crf: int | None = None
    overwrite: bool = True
    codec_settings: CodecSettings | None = None
    color_settings: ColorSettings | None = None
    custom_params: str = ''
    # Audio
    copy_audio: bool = False
    # Debug
    benchmark: bool = False
    verbose: bool = False



def args_to_encoder_settings(
    args: Namespace,
    vi: VideoInfo,
) -> EncoderSettings:
    """Parse the command line to set the encoder parameters
        inplace modifications of the video info
    """
    # Copy from input
    params: EncoderSettings = EncoderSettings(
        filepath='',
        vcodec=str_to_video_codec[vi['codec']],
        keep_sar=True,
        pix_fmt=vi['pix_fmt'],
    )

    # Encoder: codec, settings
    if args.vcodec:
        params.vcodec = str_to_video_codec[args.vcodec]

    if args.preset:
        params.preset = args.preset
    if args.tune:
        params.tune = args.tune
    if args.crf:
        params.crf = args.crf

    vcodec: VideoCodec = params.vcodec
    if vcodec == VideoCodec.FFV1:
        params.codec_settings = FFv1Settings()

    elif vcodec == VideoCodec.DNXHD:
        params.codec_settings = DNxHRSettings(
            profile=vi['profile']
        )
        params.preset = params.tune = params.crf = None

    # Extract pix_fmt from ffmpeg_args
    if (
        not args.pix_fmt
        and (re_match := re.search(re.compile(r"-pix_fmt\s([a-y0-9]+)"), args.ffmpeg_args))
    ):
        params.pix_fmt = re_match.group(1)

    elif args.pix_fmt:
        params.pix_fmt = args.pix_fmt

    if params.pix_fmt not in PIXEL_FORMAT.keys():
        sys.exit(red(f"Error: pixel format \"{params.pix_fmt}\" is not supported"))

    # Colorspace
    params.color_settings = ColorSettings(
        colorspace=vi.get('color_space', None),
        color_primaries=vi.get('color_primaries', None),
        color_trc=vi.get('color_transfer', None),
        color_range=vi.get('color_range', None),
    )

    # Set the output extension depending on the codec
    out_fp: str = vi['filepath']
    if get_extension(out_fp) == '.$$$':
        out_fp = out_fp.replace('.$$$', vcodec_to_extension[vcodec])
    params.filepath = out_fp

    # Modify the encoder settings used by the encoder node
    params.custom_params = clean_ffmpeg_params(args.ffmpeg)

    # Copy audio stream if no video clipping
    if (
        args.ss == ''
        and args.to == ''
        and args.t == ''
    ):
        params.copy_audio = True

    return params



def video_encoder_pipe_info(
    out_vi: VideoInfo,
    e_settings: EncoderSettings,
    debug: bool = False
) -> VideoPipeInfo:
    # Force pipe to rgb format
    c_order: str = 'rgb'
    # dtype is choosen depending on the encoding pixel format
    bpp = PIXEL_FORMAT[e_settings.pix_fmt]['bpp']
    pipe_pix_fmt: str = f"{c_order}{48 if bpp > 8 else 24}"
    dtype: np.dtype = np.uint16 if bpp > 8 else np.uint8
    nbytes: int = math.prod(out_vi['shape']) * np.dtype(dtype).itemsize

    vpi: VideoPipeInfo = VideoPipeInfo(
        filepath=out_vi['filepath'],
        dtype=dtype,
        c_order=c_order,
        shape=out_vi['shape'],
        nbytes=nbytes,
        nframes=out_vi['frame_count'],
        pix_fmt=pipe_pix_fmt,
    )
    return vpi



def generate_ffmpeg_encoder_cmd(
    video_pipe_info: VideoPipeInfo,
    video_info: VideoInfo,
    e_settings: EncoderSettings,
    in_media_info: MediaInfo
) -> list[str]:
    """Generate a FFmpeg command line
        - vpi: encoder pipe info
        - e_settings: encoder settings
        - video_info: encoder video info: output frame_rate and metadata
        - in_media_info: input media, use to add args to copy audio/subtitles
    """
    in_vi: VideoInfo = in_media_info['video']
    frame_rate: str = ""

    f_rate = video_info['frame_rate_r']
    if isinstance(f_rate, tuple | list):
        frame_rate = ":".join(map(str, f_rate))
    else:
        frame_rate = str(f_rate)

    h, w = video_pipe_info.shape[:2]

    ffmpeg_command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        '-f', 'rawvideo',
        '-pixel_format', video_pipe_info.pix_fmt,
        '-video_size', f"{w}x{h}",
        "-r", frame_rate,
        '-i', 'pipe:0'
    ]

    if e_settings.copy_audio and in_media_info['audio']['nstreams'] > 0:
        ffmpeg_command.extend(['-i', in_vi['filepath']])

    if e_settings.benchmark:
        ffmpeg_command.extend(["-benchmark", "-f", "null", "-"])
        return ffmpeg_command

    # Aspect ratio
    sar_dar: list[str] = []
    if 'sar' in in_vi:
        sar : str = '/'.join(map(str, in_vi['sar']))
        if sar != "1/1":
            sar_dar.append(f"setsar={sar}")

    if 'dar' in in_vi:
        dar : str = '/'.join(map(str, in_vi['dar']))
        if dar != "1/1":
            sar_dar.append(f"setdar={dar}")

    if sar_dar:
        ffmpeg_command.extend(["-vf", ','.join(sar_dar)])


    # Color space
    color_settings: ColorSettings = e_settings.color_settings
    _tmp_array: list[str] = []
    for k, v in color_settings.__dict__.items():
        if k == 'color_range':
            continue
        if (
            k not in e_settings.custom_params
            and v is not None
        ):
            _tmp_array.append(f"{k}={v}")
    if _tmp_array:
        ffmpeg_command.extend([
            "-vf",  f"setparams={':'.join(_tmp_array)}"
        ])

    else:
        # default to rec709
        fcomplex_str = clean_fcomplex("""
            setparams=colorspace=bt709
                :color_primaries=bt709
                :color_trc=bt709
        """)
        ffmpeg_command.extend([
            "-vf", fcomplex_str
        ])


    # Encoder
    if (
        "-vcodec" not in e_settings.custom_params
        and "-c:v" not in e_settings.custom_params
    ):
        ffmpeg_command.extend(["-vcodec", f"{e_settings.vcodec.value}"])

    if "-pix_fmt" not in e_settings.custom_params:
        ffmpeg_command.extend(["-pix_fmt", f"{e_settings.pix_fmt}"])

    # Settings
    if "-preset" not in e_settings.custom_params and e_settings.preset:
        ffmpeg_command.extend(["-preset", f"{e_settings.preset}"])

    if "-tune" not in e_settings.custom_params and e_settings.tune:
        ffmpeg_command.extend(["-tune", f"{e_settings.tune}"])

    if (
        "-crf" not in e_settings.custom_params
        and e_settings.crf is not None
        and e_settings.crf > 0
    ):
        ffmpeg_command.extend(["-crf", f"{e_settings.crf}"])

    if e_settings.codec_settings is not None:
        for k, v in e_settings.codec_settings.__dict__.items():
            ffmpeg_command.extend([f"-{k}", v])


    k, v = 'color_range', color_settings.color_range
    if (
        k not in e_settings.custom_params
        and v is not None
        and v.lower() not in ("unknown", "unspecified")
    ):
        limited: tuple[str] = ("tv", "mpeg", "limited")
        # full: tuple[str] = ("pc", "jpeg", "full")
        ffmpeg_command.extend([f"-{k}", "limited" if v.lower() in limited else "full"])

    # Audio/subtitles
    if e_settings.copy_audio and True:
        if in_media_info['audio']['nstreams'] > 0:
            ffmpeg_command.extend([
                "-map", "1:a", "-acodec", "copy"
            ])
        if in_media_info['subtitles']['nstreams'] > 0:
            ffmpeg_command.extend([
                "-map", "2:s", "-scodec", "copy"
            ])

    # Custom params
    codec_params: list[str] = e_settings.custom_params.split(" ")
    ffmpeg_command.extend([x for x in codec_params if x])

    # Add metadata
    if get_extension(e_settings.filepath) == ".mkv":
        ffmpeg_command.extend(["-movflags", "use_metadata_tags"])
        metadata: dict[str, str]
        for metadata in (video_info['metadata'], in_vi['metadata']):
            if metadata is not None and len(metadata.keys()):
                for k, meta in metadata.items():
                    ffmpeg_command.extend(["-metadata:s:v:0", f"{k}={meta}"])

    # Output filepath
    ffmpeg_command.append(e_settings.filepath)
    if e_settings.overwrite:
        ffmpeg_command.append('-y')


    return ffmpeg_command



def encoder_subprocess(
    video_pipe_info: VideoPipeInfo,
    video_info: VideoInfo,
    e_settings: EncoderSettings,
    in_media_info: MediaInfo,
    debug: bool = False
) -> subprocess.Popen:

    e_command: list[str] = generate_ffmpeg_encoder_cmd(
        video_pipe_info=video_pipe_info,
        e_settings=e_settings,
        video_info=video_info,
        in_media_info=in_media_info,
    )

    if debug:
        print(lightgreen(f"[V][E] FFmpeg command:"), ' '.join(e_command))
        # pprint(e_command)

    # Open subprocess
    sub_process: subprocess.Popen
    try:
        sub_process = subprocess.Popen(
            e_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        raise ValueError(f"[V][E] Unexpected error: {type(e)}", flush=True)

    return sub_process
