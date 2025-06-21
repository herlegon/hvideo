# pytc


# Installation
- create a conda env: `conda create -n pytc python=3.12.9`
- activate: `conda activate pytc`
- `git clone --recurse-submodules https://github.com/JepEtau/pytc.git`
- download and extract (FFmpeg and FFprobe)[https://ffmpeg.org/download.html#build-windows] to `external/ffmpeg` folder
- install (PyTorch)[https://pytorch.org/]: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
- Install python packages: `pip install -r requirements`

# Usage
`python.exe .\pytc.py --help`

## args
```
-i <INPUT>
[-o <OUTPUT>] if not specified, uses the input filepath and suffix
-suffix <SUFFIX> default `_pytc`
-resize_to RESIZE_TO resize to specified dimension before the inference
-resize <RESIZE> scale applied before the filtering/model

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

# Changing default values:
- arguments: `utils/arg_parse.py`
- H265/HVEC_NVEC encoder options: `pytc.py`


# Torch to TensorRT engine

## Conversion
```
python -m pynnlib.scripts.convert_model -trt -m A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k.pth -fixed -opt 640x480 -bf16 -f
```

Note:
- `-fixed`: fixed size for DAT2, oom with static
- `-opt`: fixed tensorRT size
- `-f`: force and overwrite previous engine if exists
- The conversion is really slow, it seems that it's stuck after having printed `[I] [TRT] Compiler backend is used during engine build.` but it's not.
- One of the latest printed line is `[I] TensorRT engine saved as A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k_cc8.9_op20_fp32_bf16_640x480_640x480_640x480_10.11.0.33.engine`


## Test
```
python -m pynnlib.scripts.img_infer -m A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k_cc8.9_op20_fp32_bf16_640x480_640x480_640x480_10.11.0.33.engine -i A:\imgs\img_640x480.png
```


## Video inference
```
python .\pytc.py -i A:\tmp\ep01_episode_008_j_lr.mxf -resize_to 640x480 -m A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k_cc8.9_op20_fp32_bf16_640x480_640x480_640x480_10.11.0.33.engine
```

Note:
- `-resize_to 640x480` is used to resize the video before performing the inference. Not needed when input size is consistent with the model.
