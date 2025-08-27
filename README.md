# Prostate Cancer Classification

Multi-task learning pipeline for prostate cancer classification using the SICAPv2 dataset.

## Features
- Multi-label classification for mixed Gleason patterns
- Multi-resolution processing (224x224 + 384x384)
- Transformer-based feature fusion
- Specialized cribriform detection
- Stain normalization

## Quick Start

1. **Setup environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Run training:**
```bash
python main.py
```

The script will automatically download the SICAPv2 dataset and start training.

## Results

Results and model checkpoints will be saved to `SICAPv2_results/` directory.
