from __future__ import annotations
from pprint import pprint
from queue import Queue
import time
import numpy as np
import torch
from torch import nn
from torch import Tensor
from typing import TYPE_CHECKING, Any

from media import VideoInfo
from pynnlib import (
    nnlib,
    PyTorchModel,
    Idtype,
)
from gpu_filters.gpu_resize import gpu_resize_, gpu_resize_to_
from segmentation.cache import SegmentationFrameCache
from utils.p_print import *
if TYPE_CHECKING:
    from core.types import NnFrame
    from core.t_cuda_seg_inference import CudaSegInferenceThread
    from core.types import NnFrame
    from core.t_decoder import DecoderThread
    from core.t_encoder import EncoderThread

try:
    from pynnlib.nn_pytorch.archs.EfficientTAM.efficient_track_anything.build_efficienttam import (
        build_efficienttam_video_predictor,
    )
    from pynnlib.nn_pytorch.archs.EfficientTAM.efficient_track_anything.efficienttam_video_predictor import (
        EfficientTAMVideoPredictor,
    )
except:
    pass

TORCH_LOGS="+dynamo"
TORCHDYNAMO_VERBOSE=1

def load_points(predictor: EfficientTAMVideoPredictor, inference_state: dict[str, Any]) -> None:

    # hold all the clicks we add for visualization
    prompts: dict = {}

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = (
        2  # give a unique id to each object we interact with (it can be any integers)
    )

    # Let's add a positive click at (x, y) = (200, 300) to get started on the first object
    points = np.array([[320, 240]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # # add the first object
    # ann_frame_idx = 0  # the frame index we interact with
    # ann_obj_id = (
    #     2  # give a unique id to each object we interact with (it can be any integers)
    # )

    # # Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
    # # sending all clicks (and their labels) to `add_new_points_or_box`
    # points = np.array([[200, 300], [275, 175]], dtype=np.float32)
    # # for labels, `1` means positive click and `0` means negative click
    # labels = np.array([1, 0], np.int32)
    # prompts[ann_obj_id] = points, labels
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=ann_obj_id,
    #     points=points,
    #     labels=labels,
    # )




def initialize_inference(
    self: CudaSegInferenceThread,
    video_info: VideoInfo,
    model: PyTorchModel | None,
    device: str = "cuda:0",
    dtype: Idtype = 'fp32',
) -> None:

    # settings: remove holes, dots, morphologic transformations


    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # autocast
    # torch.autocast(device, dtype=torch.bfloat16).__enter__()
    # print("CUDA Compute Capability: ", torch.cuda.get_device_capability())

    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # if torch.cuda.get_device_properties(0).major >= 8:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    # parse model
    model: PyTorchModel = nnlib.load_model('efficienttam_s_512x512')

    self.infer_stream = torch.cuda.Stream(device)
    torch.cuda.empty_cache()
    with torch.cuda.stream(self.infer_stream):
        predictor: EfficientTAMVideoPredictor = build_efficienttam_video_predictor(
            config_file=model.config_fp,
            ckpt_path=model.filepath,
            device=torch.device(device),
        )
    self.predictor = predictor
    self.vi = video_info




# ask consumer for all frames


@torch.inference_mode()
def perform_inference(
    self: CudaSegInferenceThread,
    verbose: bool = False
) -> None:

    frame_count = self.vi['frame_count']
    video_shape = self.vi['shape']

    cache: SegmentationFrameCache = SegmentationFrameCache(
        window_size=frame_count
    )

    if verbose:
        print(cyan(f"[V][I][FILTER] Cuda InferenceThread"))
    in_queue: Queue = self.in_queue

    d_thread: DecoderThread = self.producer
    e_thread: EncoderThread = self.consumer

    # seg_session: PyTorchSegSession = self.seg_session
    cuda_stream = self.infer_stream

    predictor: EfficientTAMVideoPredictor = self.predictor
    i = 0
    with torch.cuda.stream(cuda_stream):
        while not self._stop_event.is_set():
            if verbose:
                print(cyan("[V][I][SEG] waiting"))

            frame = None
            if not cache.processing():
                # receive frame from consumer
                input = in_queue.get(block=True)
                if self._stop_event.is_set():
                    print(red("asked  to stop"))
                if input is None:
                    print(red("input is none"))
                    break
                frame: NnFrame = input
                i+=1
                # print(cyan(f"[V][I][SEG] received: {type(frame)} {i}/{self.frame_count}"))

                # # Resize before inference
                # if self.prescale is not None:
                #     if isinstance(self.prescale, list | tuple):
                #         gpu_resize_to_(
                #             frame=frame,
                #             out_size=self.prescale,
                #             interpolation_method="bicubic"
                #         )
                #     else:
                #         gpu_resize_(
                #             frame=frame,
                #             scale_factor=self.prescale,
                #             interpolation_method="bicubic"
                #         )

                # resize to 512x512 (EfficientTAMBase)
                gpu_resize_to_(
                    frame=frame,
                    out_size=(512, 512),
                    interpolation_method="bicubic"
                )

                # TODO try inplace operation
                frame.tensor = frame.tensor.float().squeeze(0)

                # Append fram to cache
                # if frame.last:
                #     print(yellow("last frame"))
                cache.append(frame=frame)

                # Do not process if cache is not ready
                if not cache.is_ready():
                    # print("  ask for frame")
                    d_thread.set_produce_flag()
                    # print("continue")
                    continue

            # All frames loaded
            all_tensors: list[Tensor] = cache.get_all_tensors()
            inference_state = predictor.init_state(all_tensors, video_shape=video_shape)
            # print(red("inited"))
            predictor.reset_state(inference_state)

            load_points(predictor=predictor, inference_state=inference_state)

            # print("points loaded")
            # video_segments contains the per-frame segmentation results
            video_segments = {}
            # frame_idx, obj_ids, video_res_masks
            start_time = time.time()
            color = torch.cat([torch.rand(3), torch.tensor([0.6])], dim=0)

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0) # .cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                # print(f"returned: {out_frame_idx}")
                mask: Tensor
                # print(video_shape)
                h, w, c = video_shape
                # out_tensor = torch.zeros((1, c, h, w), device="cuda")
                # print(out_tensor.shape)
                out_mask = torch.ones((1, h, w), dtype=torch.float16, device="cuda")
                for out_obj_id, mask in video_segments[out_frame_idx].items():
                    # print(f"{out_obj_id} -> {mask.shape}")
                    out_mask = out_mask * mask.to(dtype=torch.float16)

                # print(f"{out_obj_id} -> {out_mask.shape}, {out_mask.dtype}")
                out_mask = out_mask.repeat(3, 1, 1)
                    # mask_image = mask.view(h, w, 3) * color.view(1, 1, -1)
                out_mask = out_mask.unsqueeze(0)
                # print(f"  out_mask={out_mask.shape}")
                    # print(f"  {mask_image.device}")
                    # print(torch.min(mask_image))
                    # print(torch.max(mask_image))


                #     h, w = mask.shape[-2:]
                #     mask = torch.rand(0., 1., (1, 256, 256))

                #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

                #     # Reshape mask to (height, width, 1) and multiply by color
                #     mask_image = mask.view(h, w, 1) * color.view(1, 1, -1)

                #     # Convert to numpy for imshow (matplotlib expects numpy arrays)
                #     ax.imshow(mask_image.numpy())

                # # Example usage
                # mask = torch.randint(0, 2, (1, 256, 256))  # Example binary mask
                # fig, ax = plt.subplots()
                # show_mask(mask, ax, obj_id=2)  # Display the mask
                # plt.show()


                #     show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                # pprint(out_obj_ids)
                # pprint(out_mask_logits)
                # pprint(video_segments[out_frame_idx])
                # for k, v in video_segments[out_frame_idx].items():
                #     print(f"{k} -> {v.shape}")

                out_frame = cache.current_frame()
                out_frame.tensor = out_mask.clone()
                # print(f"send frame: {out_frame.f_no}, {out_frame.tensor.shape}, {out_frame.tensor.dtype}")


                time.sleep(0.0001)
                cuda_stream.synchronize()
                e_thread.put_frame(out_frame)
                # print(yellow(f"output:"), out_frame.f_no)
            # elapsed = time.time() - start_time
            # print(elapsed)
            # print(f"{self.frame_count/elapsed:02f}fps")
            break

            if cache.is_empty():
                print("empty")
                break
