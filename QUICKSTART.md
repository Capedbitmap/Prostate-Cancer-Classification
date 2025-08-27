# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.8+ installed
- Git installed
- ~8GB free disk space (for dataset)
- CUDA-compatible GPU (recommended)

### Option 1: Automated Setup
```bash
# Run the setup script
./setup.sh

# Activate environment
source venv/bin/activate

# Run the pipeline
python main.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python main.py
```

### Option 3: Using Make
```bash
# See all available commands
make help

# Complete setup
make setup-env

# Activate environment and run
source venv/bin/activate
make run
```

## 📂 Project Structure After Setup

```
Prostate/
├── 📄 main.py                 # Main training script
├── 📋 requirements.txt        # Dependencies
├── 📖 README.md              # Full documentation
├── 🚀 setup.sh               # Setup script
├── 🔧 Makefile               # Development commands
├── 🙈 .gitignore             # Git ignore rules
├── 📁 src/                   # Source code modules
├── ⚙️  configs/              # Configuration files
├── 🧪 experiments/           # Experiment logs
├── 🤖 models/                # Saved models
├── 📊 data/                  # Dataset (created after download)
└── 📈 results/               # Training results
```

## 🎯 What Happens When You Run

1. **Dataset Download**: Automatically downloads SICAPv2 dataset from Google Drive
2. **Preprocessing**: Applies stain normalization and creates multi-label targets
3. **Training**: Trains multi-task model with transformer attention
4. **Specialist Training**: Trains specialized cribriform detector
5. **Evaluation**: Tests ensemble model and generates metrics
6. **Visualization**: Creates prediction samples and error analysis plots

## 📊 Expected Results

- **Multi-label Classification**: Handles mixed Gleason patterns
- **Cribriform Detection**: Specialized G4C pattern detection
- **Model Files**: Saved in `results/` directory
- **Metrics**: Comprehensive evaluation report
- **Visualizations**: Sample predictions and error analysis

## 🔧 Customization

Edit `configs/config.py` to modify:
- Model architecture
- Training hyperparameters
- Data augmentation
- Loss function weights

## ❓ Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch_size in config
2. **Download fails**: Check internet connection
3. **Import errors**: Ensure virtual environment is activated
4. **Slow training**: Verify CUDA is available with `torch.cuda.is_available()`

### Getting Help:
- Check the full README.md for detailed documentation
- Review logs in the terminal output
- Examine error messages in `results/` directory

## 🎉 Success!

If everything works, you should see:
- Training progress bars
- Model evaluation metrics
- Saved model checkpoints
- Generated visualizations

The complete results will be saved in the `SICAPv2_results/` directory.
