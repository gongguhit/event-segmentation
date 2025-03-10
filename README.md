# Event-Based Segmentation with Token Transition

This project implements an event-based segmentation method using token transition through self-attention layers. The implementation uses the MVSEC (Multi-Vehicle Stereo Event Camera) dataset and can optionally leverage Azure OpenAI GPT-4o for enhanced features.

## Architecture Overview

The model architecture features:
- Token embeddings with self-attention layers
- Token transition between layers using a Markov Chain approach
- Correlation-aware weighting of pivotal token embeddings
- Multi-layer training strategy

![Model Architecture](https://github.com/yourusername/event-segmentation/raw/main/docs/model_architecture.png)

## Dataset

This implementation uses the [MVSEC dataset](https://daniilidis-group.github.io/mvsec/), which contains sequences recorded with event cameras in various driving scenarios:
- Day and night driving sequences
- Indoor sequences
- Stereo event camera data
- Ground truth from LIDAR and GPS/IMU systems

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/event-segmentation.git
cd event-segmentation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the MVSEC dataset from the [official website](https://daniilidis-group.github.io/mvsec/) and place it in the `data/mvsec` directory

## Validating Dataset

To check if your dataset is correctly set up and the model can read it properly:

```bash
python check_dataset.py --data_path ./data/mvsec
```

This will visualize a few random samples and save them to the `dataset_check` directory.

## Usage

1. Train the model:
```bash
python main.py train --config config/default.yaml --data_path ./data/mvsec
```

2. Evaluate the model:
```bash
python main.py evaluate --checkpoint checkpoints/best_model.pth --visualize
```

3. Use another model checkpoint:
```bash
python main.py train --config config/default.yaml --data_path ./data/mvsec --resume checkpoints/checkpoint_epoch_10.pth
```

## Configuration

The model and training parameters can be adjusted in the `config/default.yaml` file. Key configuration options include:

- `data`: Dataset parameters (batch size, workers, etc.)
- `model`: Model architecture parameters 
- `training`: Training hyperparameters
- `logging`: Checkpoint and logging settings

## Results

Sample visualization of event-based segmentation:

![Sample Results](https://github.com/yourusername/event-segmentation/raw/main/docs/sample_results.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MVSEC dataset by [Daniilidis Group](https://daniilidis-group.github.io/mvsec/)
- The original paper that inspired this implementation 