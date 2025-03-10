# MVSEC Dataset

This directory contains utilities for loading and preprocessing the MVSEC (Multi-Vehicle Stereo Event Camera) dataset.

## Dataset Overview

The MVSEC dataset is a collection of sequences recorded with event cameras in various driving scenarios. It includes:
- Day and night driving sequences
- Indoor sequences
- Stereo event camera data
- Ground truth from LIDAR and GPS/IMU systems

## Download Instructions

1. Visit the official MVSEC website: https://daniilidis-group.github.io/mvsec/

2. Download the following sequences (recommended for this implementation):
   - outdoor_day1 
   - outdoor_day2
   - outdoor_night1
   - outdoor_night2
   - indoor_flying1

3. Create a directory structure as follows:
   ```
   mvsec/
   ├── outdoor_day1/
   │   ├── data.h5
   │   └── ...
   ├── outdoor_day2/
   │   ├── data.h5
   │   └── ...
   ├── outdoor_night1/
   │   ├── data.h5
   │   └── ...
   └── ...
   ```

4. Update the `data_path` in your configuration file to point to this directory.

## Dataset Format

The MVSEC dataset is stored in HDF5 files with the following structure:
- `events/left/data`: Contains event data for the left camera
  - `x`: x-coordinates of events
  - `y`: y-coordinates of events
  - `timestamp`: timestamps of events in microseconds
  - `polarity`: polarity of events (0 or 1)
- `davis/left/image_raw`: Contains grayscale images from the left camera
  - `data`: image data
  - `timestamp`: image timestamps in microseconds

## Preprocessing

The dataset loader in `mvsec_dataset.py` handles the following preprocessing steps:
1. Loading events between consecutive frames
2. Converting events to voxel grids or event frames
3. Preparing synchronized image and event data for training

## Additional Resources

- MVSEC Paper: https://arxiv.org/abs/1711.06396
- GitHub Repository: https://github.com/daniilidis-group/mvsec 