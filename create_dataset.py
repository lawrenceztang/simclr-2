import mmcv
import os
from random import randint

directory = "/Users/lawrencetang/Documents/deep-learning-class/simclr-2/kinetics-dataset/k400_targz/train"
save_directory = "paired_dataset"

for filename in os.listdir(directory):
    # Check if the file is a video (e.g., .mp4, .avi)
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Load the video
        video = mmcv.VideoReader(file_path)

        # Obtain a pair of frames from the video less than 1 second apart
        #Save frames to save_directory with filename as prefix and either 0 or 1 as suffix
        if len(video) <= int(video.fps) + 1:
            continue
        i1 = randint(0, len(video) - int(video.fps) - 1)
        i2 = i1 + int(video.fps)
        mmcv.imwrite(video[i1], os.path.join(save_directory, filename[:-4] + "_0.jpg"))
        mmcv.imwrite(video[i2], os.path.join(save_directory, filename[:-4] + "_1.jpg"))
