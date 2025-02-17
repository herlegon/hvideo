from datetime import (
    datetime,
    timedelta,
)
import math
from typing import TypeAlias


FrameRate: TypeAlias = float | int | tuple[int, int]


def frame_rate_to_str(frame_rate: FrameRate) -> str:
    if (
        isinstance(frame_rate, int)
        or isinstance(frame_rate, float) and int(frame_rate) == frame_rate
    ):
        return f"{int(frame_rate)}"

    return (
        f"{frame_rate[0]/frame_rate[1]:.02f}"
        if frame_rate[1] > 1
        else f"{frame_rate[0]}"
    )



def frame_to_s(no: int, frame_rate: FrameRate) -> int:
    if isinstance(frame_rate, tuple | list):
        return float(no * frame_rate[1]) / float(frame_rate[0])
    return float(no) / float(frame_rate)



def frame_to_ms(no: int, frame_rate: FrameRate) -> int:
    return 1000. * frame_to_s(no, frame_rate)



def frame_to_sexagesimal(no: int, frame_rate: FrameRate) -> str:
    """This function returns an approximate segaxesimal value.
        the ms are rounded to the near integer value.
        FFmpeg '-ss' option uses rounded ms
    """
    s: float
    if isinstance(frame_rate, tuple | list):
        s = (float(no * frame_rate[1])) / float(frame_rate[0])
    else:
        s = float(no) / float(frame_rate)
    frac, s = math.modf(s)
    return f"{timedelta(seconds=int(s))}.{int(1000 * frac + 0.5):03}"



def ms_to_frame(ms: float, frame_rate: FrameRate) -> int:
    if isinstance(frame_rate, tuple | list):
        return int((ms * frame_rate[0]) / (1000. * frame_rate[1]))
    return int((ms * frame_rate) / 1000.)



def sexagesimal_to_frame(hms: str, frame_rate: FrameRate) -> int:
    h_m_s = hms.split(':')
    h_m_s_len: int = len(h_m_s)
    if not hms or not 1 <= h_m_s_len <= 3:
        raise ValueError(f"[{hms}] is not valid sexagesimal value")
    h, m, s = 0., 0., float(h_m_s[-1])
    m: float = float(h_m_s[-2]) if h_m_s_len > 1 else 0.
    h: float = float(h_m_s[-3]) if h_m_s_len > 2 else 0.
    ms: int = int(1000 * (h * 3600 + m * 60 + s))
    return ms_to_frame(ms, frame_rate)



def s_to_sexagesimal(s: float) -> int:
    frac, s = math.modf(s)
    return f"{timedelta(seconds=int(s))}.{int(1000 * frac):03}"



def current_datetime_str() -> str:
    return datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")



def reformat_datetime(date_str: str) -> str | None:
    """Returns the datetime to a string which can be used as a filename"""
    date_format = "%a, %d %b %Y %H:%M:%S GMT"
    try:
        d = datetime.strptime(date_str, date_format)
    except:
        d_str: str = date_str
        for c in (' ', ',', ':', ','):
            d_str = d_str.replace(c, '_')
        return d_str
    return d.strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    frame_rate = (25,1)
    hms_array = (
        ".20",
        "04.20",
        "02:04.20",
        "124.2",
        "01:02:04.20",
    )
    for in_hms in hms_array:
        f_no = sexagesimal_to_frame(in_hms, frame_rate)
        hms = frame_to_sexagesimal(f_no, frame_rate)
        print(f"{in_hms}-> {f_no} -> {hms}")
