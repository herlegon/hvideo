from __future__ import annotations

from enum import Enum
import json
from pprint import pformat, pprint
import sys
import numpy as np
import subprocess
from typing import Any, Literal, TypedDict

from ..utils.logger import logger
from ..utils.path_utils import absolute_path
from ..utils.p_print import *
from ..utils.pxl_fmt import PIXEL_FORMAT
from ..utils.time_conversions import (
    FrameRate,
    frame_rate_to_str,
)
from ..utils.tools import ffprobe_exe


ChannelOrder = Literal['rgb', 'bgr']
FShape = tuple[int, int, int]


class VideoCodec(Enum):
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    FFV1 = "ffv1"
    DNXHD = "dnxhd"
    HEVC_NVENC = "hevc_nvenc"


str_to_video_codec: dict[str, VideoCodec] = {
    'h264': VideoCodec.H264,
    'h265': VideoCodec.H265,
    'ffv1': VideoCodec.FFV1,
    'vp9': VideoCodec.VP9,
    'dnxhd': VideoCodec.DNXHD,
    "hevc_nvenc": VideoCodec.HEVC_NVENC,
}


# Limit the containers
vcodec_to_extension: dict[VideoCodec, str] = {
    VideoCodec.H264: '.mkv',
    VideoCodec.H265: '.mkv',
    VideoCodec.FFV1: '.mkv',
    VideoCodec.VP9: '.mkv',
    VideoCodec.DNXHD: '.mxf',
    VideoCodec.HEVC_NVENC: '.mkv',
}


class FieldOrder(Enum):
    PROGRESSIVE = 'progressive' # Progressive video
    TOP_FIELD_FIRST = 'tt'      # Interlaced video, top field coded and displayed first
    BOTTOM_FIELD_FIRST = 'bb'   # Interlaced video, bottom field coded and displayed first
    TOP_FIELD_BOTTOM = 'tb'     # Interlaced video, top coded first, bottom displayed first
    BOTTOM_FIELD_TOP = 'bt'     # Interlaced video, bottom coded first, top displayed first


class AudioInfo(TypedDict):
    nstreams: int = 0


class SubtitleInfo(TypedDict):
    nstreams: int = 0


class VideoInfo(TypedDict):
    # Note: keep TypeDict until works enough to change to class
    shape: tuple[int, int, int]
    dtype: np.dtype
    bpp: int

    c_order: Literal['bgr', 'rgb', 'yuv']

    sar: tuple[int]
    dar: tuple[int]
    is_interlaced: bool
    field_order: FieldOrder
    frame_rate_r: FrameRate
    frame_rate_avg: FrameRate
    is_frame_rate_fixed: bool
    codec: str
    pix_fmt: str
    color_range: str
    color_space: str
    color_transfer: str
    color_primaries: str
    frame_count: int
    duration: float
    metadata: Any
    filepath: str


class MediaInfo(TypedDict):
    audio: AudioInfo
    video: VideoInfo
    subtitles: SubtitleInfo


def frame_rate_dec(frame_rate: FrameRate) -> int | float:
    if isinstance(frame_rate, int | float):
        return frame_rate
    if frame_rate[1] == 1:
        return frame_rate[0]
    return float(frame_rate[0]) / frame_rate[1]


def get_video_resolution(video_filepath):
    """Get video resolution, (i.e. width and height) of the 1st video stream"""
    ffprobe_command = ([
        ffprobe_exe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_filepath])
    result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE)
    dimensions_str = result.stdout.decode('utf-8').split('x')
    return int(dimensions_str[0]), int(dimensions_str[1])



def get_media_info(media_filepath: str):
    ffprobe_command = [
        ffprobe_exe,
        "-v", "error",
        '-show_format',
        '-show_streams',
        '-of','json',
        media_filepath
    ]
    process = subprocess.run(ffprobe_command, stdout=subprocess.PIPE)
    return json.loads(process.stdout.decode('utf-8'))



def extract_media_info(media_filepath: str) -> MediaInfo:
    media_info = get_media_info(media_filepath)
    duration_s = float(media_info['format']['duration'])

    # Use the first video track
    v_stream: dict = [
        stream for stream in media_info['streams'] if stream['codec_type'] == 'video'
    ][0]
    audio_info: AudioInfo = AudioInfo(
        nstreams=len([
            stream for stream in media_info['streams'] if stream['codec_type'] == 'audio'
        ])
    )
    subs_info: SubtitleInfo = SubtitleInfo(
        nstreams=len([
            stream for stream in media_info['streams'] if stream['codec_type'] == 'subtitle'
        ])
    )

    # Video stream
    # Only first stream is used
    field_order = FieldOrder._value2member_map_[v_stream.get('field_order', 'progressive')]
    video_info: VideoInfo = {
        'filepath': media_filepath,
        'shape': (
            v_stream['height'],
            v_stream['width'],
            0
        ),
        'sar': [int(v) for v in v_stream.get('sample_aspect_ratio', '1:1').split(':')],
        'dar': [int(v) for v in v_stream.get('display_aspect_ratio', '1:1').split(':')],

        'field_order': field_order,

        'frame_rate_r': [int(v) for v in v_stream.get('r_frame_rate').split('/')],
        'avg_frame_rate': [int(v) for v in v_stream.get('avg_frame_rate').split('/')],

        'codec': v_stream['codec_name'],
        'pix_fmt': v_stream.get('pix_fmt', None),
        # Colors
        'color_space': v_stream.get('color_space', None),
        'color_matrix': v_stream.get('color_matrix', None),
        'color_transfer': v_stream.get('color_transfer', None),
        'color_primaries': v_stream.get('color_primaries', None),
        'color_range': v_stream.get('color_range', None),

        'duration': duration_s,
        'metadata': v_stream.get('tags', None),
    }

    if isinstance(video_info['metadata'], dict):
        tags_to_discard: tuple[str] = (
            'duration',
            'encoder',
            'creation_time',
            'handler_name',
            'vendor_id',
            'file_package_umid'
        )
        tag_name: str
        for tag_name in list(video_info['metadata'].keys()).copy():
            if tag_name.lower() in tags_to_discard:
                try:
                    del video_info['metadata'][tag_name]
                except:
                    pass

    # Tags for DNxHD / DNxHR are stored in format struct
    if video_info['codec'] == 'dnxhd':
        tags_to_discard: tuple[str] = (
            'application_platform',
            'company_name',
            'generation_uid',
            'material_package_umid',
            'operational_pattern_ul',
            'product_name',
            'product_uid',
            'product_version',
            'product_version_num',
            'timecode',
            'toolkit_version_num',
            'uid',
        )

        tags: dict[str, str]
        if tags := media_info['format']['tags']:
            tag_name: str
            for tag_name in list(tags.keys()).copy():
                if tag_name.lower() in tags_to_discard:
                    try:
                        del video_info['metadata'][tag_name]
                    except:
                        pass
                    continue
                video_info['metadata'][tag_name] = tags[tag_name]

    # Is interlaced?
    if (fo := v_stream.get('field_order', None)):
        video_info['is_interlaced'] = bool(fo != FieldOrder.PROGRESSIVE.value)
    else:
        video_info['is_interlaced'] = False
    video_info['is_frame_rate_fixed'] = bool(video_info['frame_rate_r'] == video_info['avg_frame_rate'])


    # Determine nb of channels and bpp
    is_supported = False
    try:
        v = PIXEL_FORMAT[video_info['pix_fmt']]
        is_supported = v['supported']
        video_info['shape'] = (
            v_stream['height'],
            v_stream['width'],
            v['c']
        )
    except:
        pass

    if not is_supported:
        raise ValueError(f"{video_info['pix_fmt']} is not supported")

    c_order = v['c_order']
    if c_order not in ['rgb', 'bgr', 'yuv']:
        raise ValueError(f"{v['c_order']} is not supported")
    video_info['c_order'] = c_order


    video_info['bpp'] = v['bpp']
    video_info['frame_count'] = int(
        (video_info['duration'] * float(video_info['frame_rate_r'][0]))
        / video_info['frame_rate_r'][1]
        + 0.5
    )

    tags = v_stream.get('tags', None)
    if tags is not None:
        tag_frame_count = tags.get('NUMBER_OF_FRAMES', None)
        if tag_frame_count is not None:
            tag_frame_count = int(tag_frame_count)
            if video_info['frame_count'] != tag_frame_count:
                video_info['frame_count'] = tag_frame_count
                video_info['frame_rate_r'] = tag_frame_count / video_info['duration']
                video_info['avg_frame_rate'] = video_info['frame_rate_r']

    return MediaInfo(
        video=video_info,
        audio=audio_info,
        subtitles=subs_info
    )




def open_media_file(
    fp: str,
    verbose: bool = False,
    debug: bool = False,
) -> MediaInfo:
    in_media_path: str = absolute_path(fp)
    in_media_info: MediaInfo | None = None

    # Try opening media
    try:
        in_media_info = extract_media_info(in_media_path)
    except:
        sys.exit(f"[E] {in_media_path} is not a valid input media file")
    if verbose:
        print(lightcyan("FFmpeg media info:"))
        pprint(get_media_info(in_media_path))
        print(lightcyan("Input media info:"))
        pprint(in_media_info)
    logger.debug(f"FFmpeg media info:\n{pformat(get_media_info(in_media_path))}")
    logger.debug(f"Input media info:\n{pformat(in_media_info)}")

    # Update video info
    in_media_info['video']['filepath'] = in_media_path

    if debug:
        in_video_info: VideoInfo = in_media_info['video']
        frame_count_str: str = f"    {in_video_info['frame_count']} frames"
        h, w = in_video_info['shape'][:2]
        dim_str: str = f", {w}x{h}"
        frame_rate_str: str = f", {frame_rate_to_str(in_video_info['frame_rate_r'])} fps"
        pix_fmt_str: str = f", {in_video_info['pix_fmt']}"
        _sar: tuple[int] = in_video_info['sar']
        sar_str: str = f", SAR {':'.join(map(str, _sar))}" if _sar[0] / _sar[1] != 1 else ""
        _dar: tuple[int] = in_video_info['dar']
        dar_str: str = f", DAR {':'.join(map(str, _dar))}" if _dar[0] / _dar[1] != 1 else ""
        in_vi_str: str = "".join((frame_count_str, dim_str, frame_rate_str, pix_fmt_str, sar_str, dar_str))

    if verbose:
        print(in_vi_str)
        logger.debug(f"input video format: {in_vi_str}")

    return in_media_info
