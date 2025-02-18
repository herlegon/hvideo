
from enum import Enum


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
