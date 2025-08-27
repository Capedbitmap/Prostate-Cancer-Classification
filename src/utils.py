"""
Utility functions for the Prostate Cancer Classification project.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch


def check_file_exists(path):
    """Check if a file exists at the given path."""
    return os.path.exists(path)


def explore_excel(path):
    """Load and return an Excel file as a DataFrame."""
    if not check_file_exists(path):
        return None
    try:
        df = pd.read_excel(path)
        return df
    except Exception as e:
        print(f"Error loading Excel file {path}: {e}")
        return None


def create_directories(directories):
    """Create multiple directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_device():
    """Get the appropriate device (CUDA/CPU) for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_info(model, save_path):
    """Save model architecture and parameter information."""
    with open(save_path, 'w') as f:
        f.write(f"Model Architecture:\n")
        f.write(f"{model}\n\n")
        f.write(f"Total Parameters: {count_parameters(model):,}\n")


def load_classification_df(xlsx_path, crib_xlsx_path=None):
    """
    Load classification DataFrame from Excel files.
    
    Args:
        xlsx_path: Path to main Excel file
        crib_xlsx_path: Optional path to cribriform Excel file
        
    Returns:
        DataFrame with filename and label columns
    """
    if not check_file_exists(xlsx_path):
        return pd.DataFrame(columns=["filename", "label"])

    df = pd.read_excel(xlsx_path).copy()

    # Handle multi-column label format
    label_columns = ['NC', 'G3', 'G4', 'G5', 'G4C']
    if all(col in df.columns for col in label_columns):
        def get_label(row):
            for label in reversed(label_columns):
                if row[label] == 1:
                    return label.lower()
            return "nc"

        df["label"] = df.apply(get_label, axis=1)
        df["filename"] = df["image_name"]
        df = df[["filename", "label"]]

    # Handle cribriform labels
    if crib_xlsx_path and check_file_exists(crib_xlsx_path):
        try:
            crib_df = pd.read_excel(crib_xlsx_path)
            crib_df["filename"] = crib_df["image_name"]
            g4c_files = set(crib_df["filename"].tolist())
            df.loc[(df["label"] == "g4") & (df["filename"].isin(g4c_files)), "label"] = "g4c"
        except Exception as e:
            print(f"Error processing cribriform file: {e}")

    return df


def create_multilabel_targets(df, masks_folder, threshold=0.05):
    """
    Create multi-label targets from segmentation masks.
    
    Args:
        df: DataFrame with image information
        masks_folder: Path to segmentation masks
        threshold: Minimum percentage for class inclusion
        
    Returns:
        Enhanced DataFrame with multilabels column
    """
    enhanced_df = df.copy()
    multilabels_list = []

    for idx, row in df.iterrows():
        img_filename = row["image_name"] if "image_name" in row else row["filename"]
        mask_filename = os.path.splitext(img_filename)[0] + ".png"
        mask_path = os.path.join(masks_folder, mask_filename)

        multilabels = []

        if os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert("L")
                mask_array = np.array(mask)
                unique_values = np.unique(mask_array)
                total_pixels = mask_array.size

                for grade in range(4):
                    if grade in unique_values:
                        percentage = np.sum(mask_array == grade) / total_pixels
                        if percentage >= threshold:
                            if grade == 0:
                                multilabels.append(0)  # NC
                            elif grade == 1:
                                multilabels.append(1)  # G3
                            elif grade == 2:
                                if "g4c" in row["label"]:
                                    multilabels.append(3)  # G4C
                                else:
                                    multilabels.append(2)  # G4
                            elif grade == 3:
                                multilabels.append(4)  # G5

                if not multilabels:
                    # Fallback to original label
                    label_map = {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4}
                    multilabels.append(label_map.get(row["label"], 0))

            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")
                label_map = {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4}
                multilabels.append(label_map.get(row["label"], 0))
        else:
            # No mask available, use original label
            label_map = {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4}
            multilabels.append(label_map.get(row["label"], 0))

        multilabels_list.append(multilabels)

    enhanced_df["multilabels"] = multilabels_list
    return enhanced_df


def create_cribriform_labels(df):
    """
    Create binary cribriform labels from DataFrame.
    
    Args:
        df: DataFrame with label or multilabels information
        
    Returns:
        Tensor of cribriform labels (0/1)
    """
    cribriform_labels = []
    for _, row in df.iterrows():
        if "g4c" in str(row["label"]) or (isinstance(row["multilabels"], list) and 3 in row["multilabels"]):
            cribriform_labels.append(1.0)
        else:
            cribriform_labels.append(0.0)
    return torch.tensor(cribriform_labels, dtype=torch.float32)
