#!/usr/bin/env python3
"""
Test script to verify the environment setup and key imports.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("🔧 Testing package imports...")
    
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import cv2
        import PIL
        import sklearn
        import matplotlib
        import seaborn
        import albumentations
        import timm
        import gdown
        print("✅ All core packages imported successfully!")
        
        # Test PyTorch functionality
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"🎮 CUDA device: {torch.cuda.get_device_name()}")
        else:
            print("💻 Running on CPU")
            
        # Test tensor operations
        x = torch.rand(2, 3)
        y = torch.rand(3, 2)
        z = torch.mm(x, y)
        print(f"🧮 Tensor multiplication test: {z.shape}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_file_structure():
    """Test that project structure is correct."""
    print("\n📁 Testing project structure...")
    
    import os
    
    required_dirs = [
        "src",
        "configs", 
        "experiments",
        "models",
        ".vscode"
    ]
    
    required_files = [
        "main.py",
        "README.md",
        "requirements.txt",
        ".gitignore",
        "Makefile",
        "setup.sh"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ directory exists")
        else:
            print(f"❌ {directory}/ directory missing")
            
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            
    return True


def test_config():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        import sys
        sys.path.append('configs')
        
        from configs.config import MODEL_CONFIG, TRAINING_CONFIG
        print("✅ Configuration loaded successfully")
        print(f"📊 Model classes: {MODEL_CONFIG['num_classes']}")
        print(f"🏃 Training epochs: {TRAINING_CONFIG['epochs']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Prostate Cancer Classification - Environment Test")
    print("=" * 50)
    
    success = True
    
    success &= test_imports()
    success &= test_file_structure() 
    success &= test_config()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Environment is ready.")
        print("🚀 You can now run: python main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("💡 Try: pip install -r requirements.txt")
        
    print("=" * 50)
