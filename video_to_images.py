"""Converts video files to images for for training Gaussian Splatting models.

Usage:

    video_to_images.py -h

Note: This script requires FFmpeg (https://www.ffmpeg.org/).
"""

import os
import subprocess
import cv2
import argparse
import shutil
from pathlib import Path

def extract_frames(video_path: Path, output_dir: Path, num_frames: int, resolution: tuple[int, int] = None) -> None:
    """
    Extracts a specified number of frames from a video, optionally scaling them.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frrames.
        num_frames (int): Number of frames to extract.
        resolution (tuple, optional): Desired resolution as (width, height).
    """

    # Ensure that the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the total number of frames in the video
    cap = cv2.VideoCapture(video_path) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate the frame intervals
    interval = max(1, total_frames // num_frames)

    # Prepare FFMPEG command
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf",
        f"select='not(mod(n\\,{interval}))'" # for sampling evenly based on the calculated interval.
    ]

    # Add resolution scaling if specified
    if resolution:
        width, height = resolution
        ffmpeg_cmd[-1] += f",scale={width}:{height}" # adjust for for resolution if specified
    
    ffmpeg_cmd += [
        "-vsync", "vfr",
        f"{output_dir}/%d.jpg",
        "-hide_banner",
        "-loglevel", "error"
    ]

    # Execute the command
    print("Running FFmpeg command...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Frames extracted to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_file", required=True, type=str, help="video file path")
    parser.add_argument("-o", "--output_directory", required=True, type=str, help="output directory")
    parser.add_argument("-f", "--num_frames", default=200, type=int, help="number of frames to extract")
    parser.add_argument("-r", "--resolution", type=str, help="desired resolution, in the form 'height,width'") #default="1440,1920" is what the metafood captured at.
    args = parser.parse_args()

    if args.resolution != None:
        args.resolution = args.resolution.split(",")
        res_height = int(args.resolution[0])
        res_width = int(args.resolution[1])
        args.resolution = [res_width, res_height]

    # If input video is of type .mov (from IPhone), convert to mp4.
    # ffmpeg -i {in-video}.mov -vcodec h264 -acodec aac {out-video}.mp4
    if args.video_file.lower().endswith(".mov"):
        print("converting file from .mov to .mp4")
        filepath_without_extension = args.video_file[:-4] # Remove the last 4 characters
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", f"{args.video_file}",
            "-vcodec", "h264",
            "-acodec", "aac",
            f"{filepath_without_extension}.mp4",
            "-loglevel", "error"
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        args.video_file = filepath_without_extension + ".mp4"

    extract_frames(args.video_file, args.output_directory, args.num_frames, args.resolution)
