import torch
from pynnlib import (
    nnlib
)
# settings: remove holes, dots, morphologic transformations


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# autocast
# torch.autocast(device, dtype=torch.bfloat16).__enter__()
# print("CUDA Compute Capability: ", torch.cuda.get_device_capability())

# parse model
nnlib.load_model('efficienttam_s_512x512')

# Load video
[
    first_frame,  # first_frame_path
    gr.State([]),  # tracking_points
    gr.State([]),  # trackings_input_label
    first_frame,  # input_first_frame_image
    first_frame,  # points_map
    extracted_frames_output_dir,  # video_frames_dir
    scanned_frames,  # scanned_frames
    None,  # stored_inference_state
    None,  # stored_frame_names
    gr.update(open=False),  # video_in_drawer
]


outputs=[
    first_frame_path,
    tracking_points,  # update Tracking Points in the gr.State([]) object
    trackings_input_label,  # update Tracking Labels in the gr.State([]) object
    input_first_frame_image,  # hidden component used as ref when clearing points
    points_map,  # Image component where we add new tracking points
    video_frames_dir,  # Array where frames from video_in are deep stored
    scanned_frames,  # Scanned frames by EfficientTAM
    stored_inference_state,  # EfficientTAM inference state
    stored_frame_names,  #
    video_in_drawer,  # Accordion to hide uploaded video player
],



inference_state = predictor.init_state(video_path=frames_dir, async_loading_frames=True, offload_video_to_cpu=True)


# ask consumer for all frames

