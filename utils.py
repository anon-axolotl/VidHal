import random
import string
import argparse
import numpy as np
import torch
from decord import VideoReader
import io
from tqdm import tqdm
from collections import OrderedDict

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--annotations_path", type=str, required=True)
    parser.add_argument("--videos_path", type=str, required=True)
    parser.add_argument("--options_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=4)

    # Inference parameters
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_captions", type=int, default=3)

    # Model parameters
    parser.add_argument("--model", type=str, default="random")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--load_8bit", action="store_true")
    # LLaVA-NeXT-Video parameters
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--mm_pooling_position", type=str, default="after")
    # VideoChat2 parameters
    parser.add_argument("--config_path", type=str, default=None)
    # MovieChat parameters
    parser.add_argument("--fragment_video_path", type=str, default=None)
    # Proprietary model parameters
    parser.add_argument("--api_key", type=str, default=None)

    # Evaluation parameters
    parser.add_argument("--predictions_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)

    # TODO: Add more parameters if needed
    args = parser.parse_args()

    return args

OPTION_DISPLAY_ORDER = None
def generate_display_order(dataset):
    """
    Generates a random option display order if none is provided.
    Maintains a global display order to be used in multiple components, if generating a random order is required
    and one hasn't been generated yet.
    """
    global OPTION_DISPLAY_ORDER
    if OPTION_DISPLAY_ORDER is None:
        OPTION_DISPLAY_ORDER = {}
        for i in tqdm(range(len(dataset))):
            example = dataset[i]
            video_id, captions = example["video_id"], example["captions"]
            caption_keys = list(captions.keys())
            random.shuffle(caption_keys)

            # Assign permuted options (A, B, C) to actual order (1, 2, 3)
            option_prefix = list(string.ascii_uppercase)
            option_to_rank = OrderedDict(
                {option_prefix[i]: option for i, option in enumerate(caption_keys)}
            )

            OPTION_DISPLAY_ORDER[video_id] = option_to_rank

    return OPTION_DISPLAY_ORDER

"""
Adapted from VideoChat2: https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/dataset/video_utils.py
"""
def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, fps=1, max_num_frames=-1, clip=None):
    if sample in ["rand", "middle"]: # Uniform sampling
        acc_samples = min(num_frames, vlen)
        # Split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            if clip:
                start_idx, end_idx = round(clip[0] * fps), min(round(clip[1] * fps), max_num_frames)
            else:
                if max_num_frames < 0:
                    max_num_frames = vlen - 1
                start_idx, end_idx  = 0, max_num_frames
            
            seg_size = float(end_idx - start_idx) / num_frames
            frame_indices = np.array([
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_frames)
            ])
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # Sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / fps
        delta = 1 / output_fps  # Gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices
    
def read_video(
    video_path, num_frames=None, sample='rand', fix_start=None, client=None, clip=None
):
    if video_path.startswith('s3') or video_path.startswith('p2'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if num_frames is None:
        num_frames = round(duration)

    # Hack: To get all frames
    if num_frames == -1:
        num_frames = vlen

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start, fps=fps, max_num_frames=vlen - 1, clip=clip
    )
    frames = video_reader.get_batch(frame_indices)
    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames.asnumpy())
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    
    return frames, frame_indices, float(fps)