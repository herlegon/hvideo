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


# Torch to TensorRT engine

## Conversion
```
cd pynnlib
python -m scripts.convert_model -trt -m A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k.pth -opt 640x480 -fixed -bf16 -f
```

Note:
- `-f`: force and overwrite previous engine if exists
- The conversion is really slow, it seems that it's stuck after having printed `[I] [TRT] Compiler backend is used during engine build.` but it's not.
- One of the latest printed line is `[I] TensorRT engine saved as A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k_cc8.9_op20_fp32_bf16_640x480_640x480_640x480_10.9.0.34.engine`


## Test
```
python -m scripts.img_infer -m A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k_cc8.9_op20_fp32_bf16_640x480_640x480_640x480_10.9.0.34.engine -i A:\imgs\img_640x480.png
```


## Video inference
```
python .\pytc.py -i .\ep01_episode_008_j_lr.mxf -resize_to 640x480 -m A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k_cc8.9_op20_fp32_bf16_640x480_640x480_640x480_10.9.0.34.engine -bf16
```

Note:
- `-resize_to 640x480` is used to resize the video before running the inference
