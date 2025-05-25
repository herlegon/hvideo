# pytc


# Installation
- `git clone --recurse-submodules https://github.com/JepEtau/pytc.git`
- download and extract (FFmpeg and FFprobe)[https://ffmpeg.org/download.html#build-windows] to `external/ffmpeg` folder
- install (PyTorch)[https://pytorch.org/]
- `pip install -r requirements`

# Usage
`python.exe .\pytc.py --help`

## args
```
-i <INPUT>
[-o <OUTPUT>] if not specified, uses the input filepath and suffix
-suffix <SUFFIX> default `_pytc`
-resize <RESIZE>> scale applied before the filtering/model
-resize <RESIZE>> scale applied before the filtering/model

(-fsar <FSAR>) you should never need this
(-fsar_h <FSAR_H>) you should never need this

-ss <SS> FFmpeg format or frame no. followed by `f` (example: -ss 1234f)
-t <T> FFmpeg format or frame no. followed by `f`
-to <TO> FFmpeg format or frame no. followed by `f`
```

## Args passed to FFmpeg
```
-vcodec {h264, h265, ffv1, vp9, dnxhd, hevc_nvenc}: default: `h265`
-pix_fmt <PIX_FMT> {yuv420p, yuv422p10le, ...}, default: `yuv420p10le`
-preset {ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow}
-crf <CRF>
-tune {film, animation, grain, stillimage, fastdecode, zerolatency}
-ffmpeg <FFMPEG> must be betwen `"`
```

## Debug
```
--debug
--log   experimental, not tested
```

# Changing default values:
- arguments: `utils/arg_parse.py`
- H265/HVEC_NVEC encoder options: `pytc.py`

