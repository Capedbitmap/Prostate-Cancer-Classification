# Prostate Cancer Classification

Multi-task learning pipeline for prostate cancer classification using the SICAPv2 dataset.

## Features
- Multi-label classification for mixed Gleason patterns
- Multi-resolution processing (224x224 + 384x384)
- Transformer-based feature fusion
- Specialized cribriform detection
- Stain normalization

## Data Structure

The SICAPv2 dataset is located in `data/SICAPv2/` with the following structure:
```
data/SICAPv2/
├── images/           # 18,783 histology patch images (512x512, 10X magnification)
├── masks/            # Corresponding annotation masks
├── partition/
│   ├── Test/        # Train/test split files
│   └── Validation/  # Cross-validation folds (Val1-Val4)
├── wsi_labels.xlsx  # WSI-level Gleason scores
└── readme.txt       # Dataset documentation
```

## Quick Start

1. **Setup virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Test the setup:**
```bash
python test_imports.py
```

3. **Run training:**
```bash
python main.py
```

## Data Information

- **Total samples**: 18,783 patches (512×512 pixels)
- **Magnification**: 10X
- **Labels**: Multi-label classification (NC, G3, G4, G5, G4C)
- **Cribriform detection**: Specialized detection for G4C patterns
- **Cross-validation**: 4-fold patient-based splits available

## Results

Results and model checkpoints will be saved to `SICAPv2_results/` directory.
