#!/usr/bin/env python3
"""
Event Batch Visualizer

This script processes events in batches of a specified size (e.g., 20000 events) 
and generates an accumulated image and time surface map for each batch.
"""

import dv_processing as dv
import cv2
import numpy as np
import os
import argparse
import glob
import subprocess
from tqdm import tqdm

def visualize_aedat4_by_event_count(input_file, output_dir, batch_size=20000, max_frames=300):
    """
    Process events in batches of specified size and generate visualizations.
    
    Args:
        input_file: Path to the AEDAT4 file
        output_dir: Directory to save visualizations
        batch_size: Number of events per batch
        max_frames: Maximum number of frames to generate (None for unlimited)
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading AEDAT4 file: {input_file}")
    
    # Open the AEDAT4 file
    try:
        reader = dv.io.MonoCameraRecording(input_file)
        print(f"Opened AEDAT4 file from camera: {reader.getCameraName()}")
    except Exception as e:
        print(f"Error opening AEDAT4 file: {e}")
        return
    
    # Check for event stream
    if not reader.isEventStreamAvailable():
        print("No event stream available in this file!")
        return
    
    # Get dimensions
    dimensions = reader.getEventResolution()
    if isinstance(dimensions, tuple):
        width, height = dimensions
    else:
        width, height = dimensions.width, dimensions.height
    
    print(f"Event dimensions: {width}x{height}")
    
    # Initialize visualizers
    tuple_dimensions = (width, height)
    
    # Initialize accumulator
    accumulator = dv.Accumulator(tuple_dimensions)
    accumulator.setMinPotential(0.0)
    accumulator.setMaxPotential(1.0)
    accumulator.setNeutralPotential(0.5)
    accumulator.setEventContribution(0.15)
    accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
    accumulator.setDecayParam(1e+6)
    accumulator.setIgnorePolarity(False)
    accumulator.setSynchronousDecay(True)  # Synchronous to see clear transitions
    
    # Initialize time surface
    time_surface = dv.TimeSurface(tuple_dimensions)
    
    print(f"Processing events in batches of {batch_size} events...")
    
    # Keep track of frames
    frame_count = 0
    total_events = 0
    event_count_in_current_batch = 0
    
    # Create subdirectories for different visualizations
    acc_dir = os.path.join(output_dir, "accumulator")
    ts_dir = os.path.join(output_dir, "time_surface")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(ts_dir, exist_ok=True)
    
    # Process events
    while True:
        # Get next batch of events
        events = reader.getNextEventBatch()
        
        # Check if end of file
        if events is None or len(events) == 0:
            print("End of file reached")
            break
        
        # Process this batch of events
        event_count_in_current_batch += len(events)
        
        # Pass to accumulators
        accumulator.accept(events)
        time_surface.accept(events)
        
        # If we've collected enough events, generate a frame
        if event_count_in_current_batch >= batch_size:
            # Generate and save frames
            acc_frame = accumulator.generateFrame()
            ts_frame = time_surface.generateFrame()
            
            acc_path = os.path.join(acc_dir, f"frame_{frame_count:06d}.png")
            ts_path = os.path.join(ts_dir, f"frame_{frame_count:06d}.png")
            
            cv2.imwrite(acc_path, acc_frame.image)
            cv2.imwrite(ts_path, ts_frame.image)
            
            # Update counters
            total_events += event_count_in_current_batch
            frame_count += 1
            event_count_in_current_batch = 0  # Reset counter
            
            # Show progress
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames, {total_events} events")
            
            # Check if we've reached the maximum number of frames
            if max_frames is not None and frame_count >= max_frames:
                print(f"Reached maximum number of frames ({max_frames})")
                break
    
    print(f"Processing complete! Generated {frame_count} frames from {total_events} events")
    
    # Create videos
    if frame_count > 0:
        create_video(acc_dir, os.path.join(output_dir, "accumulator_video.mp4"), 60)
        create_video(ts_dir, os.path.join(output_dir, "time_surface_video.mp4"), 60)

def create_video(input_dir, output_file, framerate=60, crf=20):
    """
    Create a video with 1:1 frame mapping at specified framerate.
    Each input frame becomes exactly one frame in the output video.
    
    Args:
        input_dir: Directory containing input frames
        output_file: Output video file path
        framerate: Frame rate (fps) of output video (default: 60)
        crf: Constant Rate Factor (quality, lower is better)
    """
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not frames:
        print(f"No frames found in {input_dir}")
        return False
    
    # Calculate duration based on framerate
    duration_sec = len(frames) / framerate
    
    print(f"Creating video with {len(frames)} frames at {framerate} fps...")
    print(f"Video duration will be {duration_sec:.2f} seconds (1:1 frame mapping)")
    
    # Set the pattern for input files
    pattern = os.path.join(input_dir, "frame_%06d.png")
    
    # FFmpeg command with direct framerate
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(framerate),  # Input framerate (same as output)
        "-i", pattern,                 # Input pattern
        "-c:v", "libx264",             # Video codec
        "-pix_fmt", "yuv420p",         # Pixel format
        "-crf", str(crf),              # Quality
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video saved: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process events in batches and generate visualizations")
    parser.add_argument("--input", type=str, default="bottle.aedat4",
                        help="Input AEDAT4 file")
    parser.add_argument("--output_dir", type=str, default="./batch_visualization",
                        help="Output directory for visualizations")
    parser.add_argument("--batch_size", type=int, default=20000,
                        help="Number of events per batch")
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Maximum number of frames to generate")
    
    args = parser.parse_args()
    
    visualize_aedat4_by_event_count(args.input, args.output_dir, args.batch_size, args.max_frames)

if __name__ == "__main__":
    main() 