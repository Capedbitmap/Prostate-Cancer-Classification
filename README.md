# Prostate Cancer Classification with Multi-Task Learning

A comprehensive deep learning pipeline for prostate cancer classification using the SICAPv2 dataset, featuring novel multi-label classification, cribriform pattern detection, and multi-resolution processing.

## 🚀 Novel Features

- **Multi-label Classification**: Handles mixed Gleason patterns in tissue samples
- **Multi-resolution Processing**: Dual-scale input (224x224 + 384x384) for enhanced feature extraction
- **Transformer-based Fusion**: Multi-head attention mechanism for feature integration
- **Specialized Cribriform Detection**: Two-stage approach with dedicated cribriform specialist
- **Multi-task Learning**: Joint optimization of classification, segmentation, and cribriform detection
- **Advanced Preprocessing**: Macenko stain normalization for consistent appearance
- **Mask Integration**: Tissue mask percentages as additional features

## 📊 Dataset

This project uses the **SICAPv2** dataset for prostate cancer classification:
- High-resolution histopathology images
- Gleason grading annotations (NC, G3, G4, G4C, G5)
- Segmentation masks for tissue regions
- Cribriform pattern annotations

## 🏗️ Architecture

### Multi-Task Multi-Resolution Model
- **Backbone**: Dual EfficientNet-B0 encoders for different resolutions
- **Feature Fusion**: Transformer-based attention mechanism
- **Multi-task Heads**: 
  - Multi-label classifier (5 classes)
  - Cribriform detector (binary)
  - Segmentation head (4 tissue types)

### Specialized Ensemble
- Base multi-task model for general classification
- Cribriform specialist for enhanced G4C detection
- Dynamic routing based on G4 predictions

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Prostate
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Prostate/
├── main.py                 # Main training and evaluation pipeline
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
├── Docs/                  # Documentation and benchmarks
│   └── Prostate Dataset Benchmark.xlsx
├── src/                   # Source code modules (to be created)
├── configs/               # Configuration files (to be created)
├── experiments/           # Experiment logs and results
└── models/                # Saved model checkpoints
```

## 🔧 Usage

### Quick Start

1. **Run the complete pipeline**:
```bash
python main.py
```

The script will automatically:
- Download the SICAPv2 dataset
- Preprocess images with stain normalization
- Train the multi-task model
- Train the cribriform specialist
- Evaluate the ensemble model
- Generate visualizations and reports

### Key Components

- **Data Loading**: Automatic dataset download and preprocessing
- **Multi-label Targets**: Generated from segmentation masks
- **Training**: Multi-task learning with adaptive loss weighting
- **Evaluation**: Comprehensive metrics including cross-validation
- **Visualization**: Prediction samples and error analysis

## 📈 Results

The model achieves state-of-the-art performance on:
- **Multi-label Classification**: Handles mixed Gleason patterns
- **Cribriform Detection**: Specialized detection of G4C patterns
- **Cross-validation**: Robust performance across validation folds

Results are automatically saved to `SICAPv2_results/` including:
- Model checkpoints (`*.pth`)
- Performance metrics (`enhanced_results_summary.txt`)
- Visualizations (`*.png`)

## 🧪 Experiments

To run specific experiments or modify the pipeline:

1. **Adjust hyperparameters** in `main.py`
2. **Modify architecture** in the model definition
3. **Change data augmentation** in transform definitions
4. **Experiment with loss weights** in `MultiTaskLoss`

## 📊 Evaluation Metrics

- **Multi-label Metrics**: Exact match accuracy, Hamming distance
- **Cribriform Metrics**: Precision, Recall, F1-score, Confusion matrix
- **Cross-validation**: K-fold validation across dataset partitions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SICAPv2 dataset creators
- PyTorch and timm library contributors
- Medical imaging research community

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{prostate_multitask_2025,
  title={Multi-Task Multi-Label Prostate Cancer Classification with Specialized Cribriform Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/prostate-classification}
}
```
