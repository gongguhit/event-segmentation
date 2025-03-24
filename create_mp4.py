#!/usr/bin/env python3
"""
Create MP4 videos from PNG frame sequences

This script converts a sequence of PNG frames to an MP4 video using FFmpeg.
"""

import os
import argparse
import subprocess
import glob
from tqdm import tqdm

def create_mp4_from_frames(input_pattern, output_file, framerate=30, crf=20):
    """
    Create an MP4 video from a sequence of image frames using FFmpeg.
    
    Args:
        input_pattern: Glob pattern for input images (e.g., "./frames/frame_*.png")
        output_file: Path to output MP4 file
        framerate: Frame rate of the output video (frames per second)
        crf: Constant Rate Factor (0-51, lower means better quality, 18-28 is typical)
    """
    # Get the frames from the pattern
    frames = sorted(glob.glob(input_pattern))
    
    if not frames:
        print(f"No frames found matching pattern: {input_pattern}")
        return False
    
    print(f"Found {len(frames)} frames for {output_file}")
    
    # Create a temporary file list
    temp_list_file = "frames.txt"
    with open(temp_list_file, "w") as f:
        for frame in frames:
            f.write(f"file '{frame}'\n")
    
    # Build the FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f", "concat",  # Use concat demuxer
        "-safe", "0",    # Don't restrict filenames
        "-i", temp_list_file,
        "-framerate", str(framerate),
        "-c:v", "libx264",
        "-profile:v", "high",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_file
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the FFmpeg command
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video saved: {output_file}")
        # Clean up temp file
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        # Clean up temp file
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)
        return False

def main():
    parser = argparse.ArgumentParser(description="Create MP4 videos from PNG frame sequences")
    parser.add_argument("--input_dir", type=str, default="./visualization_results",
                        help="Directory containing PNG frames")
    parser.add_argument("--output_dir", type=str, default="./visualization_results",
                        help="Directory to save MP4 videos")
    parser.add_argument("--framerate", type=int, default=30,
                        help="Frame rate of the output video (frames per second)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create MP4 for accumulator frames
    accumulator_pattern = os.path.join(args.input_dir, "accumulator_*.png")
    accumulator_output = os.path.join(args.output_dir, "accumulator_30fps.mp4")
    create_mp4_from_frames(accumulator_pattern, accumulator_output, args.framerate)
    
    # Create MP4 for time surface frames
    ts_pattern = os.path.join(args.input_dir, "time_surface_*.png")
    ts_output = os.path.join(args.output_dir, "time_surface_30fps.mp4")
    create_mp4_from_frames(ts_pattern, ts_output, args.framerate)
    
    print("Done!")

if __name__ == "__main__":
    main() 