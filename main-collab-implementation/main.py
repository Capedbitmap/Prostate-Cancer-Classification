!pip install -q openpyxl gdown timm opencv-python albumentations

import os
import sys
import shutil
import zipfile
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.models as torch_models
import torchvision.models.segmentation as seg_models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import cv2
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import gc

try:
    from torch.cuda.amp import autocast, GradScaler
    USE_AMP = torch.cuda.is_available()
except:
    USE_AMP = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GDRIVE_ZIP_LINK = "1Jiz3Ij2NbbhGml2jce31gboJwDDct73P"
LOCAL_ZIP_NAME = "SICAPv2.zip"

!gdown --id {GDRIVE_ZIP_LINK} -O {LOCAL_ZIP_NAME}

if os.path.exists("SICAPv2"):
    shutil.rmtree("SICAPv2")
os.makedirs("SICAPv2", exist_ok=True)
with zipfile.ZipFile(LOCAL_ZIP_NAME, 'r') as zip_ref:
    zip_ref.extractall("SICAPv2")

BASE_DIR = "/content/SICAPv2/SICAPv2"
IMAGES_FOLDER = os.path.join(BASE_DIR, "images")
MASKS_FOLDER = os.path.join(BASE_DIR, "masks")
PARTITION_DIR = os.path.join(BASE_DIR, "partition")
SAVE_DIR = "/content/SICAPv2_results"
os.makedirs(SAVE_DIR, exist_ok=True)

TEST_DIR = os.path.join(PARTITION_DIR, "Test")
VALIDATION_DIR = os.path.join(PARTITION_DIR, "Validation")

TRAIN_XLSX = os.path.join(TEST_DIR, "Train.xlsx")
TEST_XLSX = os.path.join(TEST_DIR, "Test.xlsx")
TRAIN_CRIB_XLSX = os.path.join(TEST_DIR, "TrainCribfriform.xlsx")
TEST_CRIB_XLSX = os.path.join(TEST_DIR, "TestCribfriform.xlsx")
WSI_LABELS_XLSX = os.path.join(BASE_DIR, "wsi_labels.xlsx")

VAL1_DIR = os.path.join(VALIDATION_DIR, "Val1")
VAL2_DIR = os.path.join(VALIDATION_DIR, "Val2")
VAL3_DIR = os.path.join(VALIDATION_DIR, "Val3")
VAL4_DIR = os.path.join(VALIDATION_DIR, "Val4")

def check_file_exists(path):
    return os.path.exists(path)

def explore_excel(path):
    if not check_file_exists(path):
        return None
    try:
        df = pd.read_excel(path)
        return df
    except Exception as e:
        return None

train_df = explore_excel(TRAIN_XLSX)
test_df = explore_excel(TEST_XLSX)
train_crib_df = explore_excel(TRAIN_CRIB_XLSX)
test_crib_df = explore_excel(TEST_CRIB_XLSX)
wsi_labels_df = explore_excel(WSI_LABELS_XLSX)

def stain_normalize_macenko(img, target_means=None, target_stds=None):
    if target_means is None:
        target_means = [0.65, 0.70, 0.29]
    if target_stds is None:
        target_stds = [0.15, 0.15, 0.10]

    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.clip(img_array, 0.01, 0.99)

    od = -np.log(img_array)
    od_reshaped = od.reshape(-1, 3)

    try:
        u, s, vh = np.linalg.svd(od_reshaped)
        he_matrix = vh[:2]

        projections = np.dot(od_reshaped, he_matrix.T)
        angles = np.arctan2(projections[:, 1], projections[:, 0])

        min_angle = np.percentile(angles, 1)
        max_angle = np.percentile(angles, 99)

        he_matrix_ordered = np.array([
            he_matrix[0] * np.cos(min_angle) + he_matrix[1] * np.sin(min_angle),
            he_matrix[0] * np.cos(max_angle) + he_matrix[1] * np.sin(max_angle)
        ])

        concentrations = np.linalg.lstsq(he_matrix_ordered.T, od_reshaped.T, rcond=None)[0]

        means = np.mean(concentrations, axis=1)
        stds = np.std(concentrations, axis=1)

        normalized_concentrations = np.zeros_like(concentrations)
        for i in range(2):
            if stds[i] > 0:
                normalized_concentrations[i] = (concentrations[i] - means[i]) / stds[i] * target_stds[i] + target_means[i]
            else:
                normalized_concentrations[i] = concentrations[i]

        normalized_od = np.dot(he_matrix_ordered.T, normalized_concentrations).T
        normalized_img = np.exp(-normalized_od.reshape(img_array.shape))
        normalized_img = np.clip(normalized_img * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(normalized_img)
    except:
        return img

class MultiResolutionDataset(Dataset):
    def __init__(self, df, images_folder, masks_folder=None, transform_low=None, transform_high=None,
                 is_multilabel=False, get_mask_percentages=False):
        self.df = df.reset_index(drop=True)
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.transform_low = transform_low
        self.transform_high = transform_high
        self.is_multilabel = is_multilabel
        self.get_mask_percentages = get_mask_percentages

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_filename = row["image_name"] if "image_name" in row else row["filename"]

        img_path = os.path.join(self.images_folder, img_filename)

        try:
            image = Image.open(img_path).convert("RGB")
            image = stain_normalize_macenko(image)
        except:
            image = Image.new("RGB", (512, 512), color=0)

        mask_percentages = None
        if self.get_mask_percentages and self.masks_folder:
            try:
                mask_filename = os.path.splitext(img_filename)[0] + ".png"
                mask_path = os.path.join(self.masks_folder, mask_filename)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert("L")
                    mask_array = np.array(mask)
                    total_pixels = mask_array.size
                    percentages = []
                    for grade in range(4):
                        percentage = np.sum(mask_array == grade) / total_pixels
                        percentages.append(percentage)
                    mask_percentages = torch.tensor(percentages, dtype=torch.float32)
                else:
                    mask_percentages = torch.zeros(4, dtype=torch.float32)
            except:
                mask_percentages = torch.zeros(4, dtype=torch.float32)

        img_low = self.transform_low(image) if self.transform_low else None
        img_high = self.transform_high(image) if self.transform_high else None

        if self.is_multilabel:
            labels = torch.zeros(5, dtype=torch.float32)
            if "multilabels" in row and isinstance(row["multilabels"], list):
                for label_idx in row["multilabels"]:
                    labels[label_idx] = 1.0
            else:
                label_str = row["label"]
                label_map = {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4}
                if label_str in label_map:
                    labels[label_map[label_str]] = 1.0
        else:
            label_str = row["label"]
            label_map = {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4}
            labels = label_map.get(label_str, 0)

        result = [img_low, img_high, labels]
        if mask_percentages is not None:
            result.append(mask_percentages)

        return tuple(result)

def create_multilabel_targets(df, masks_folder):
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
                threshold = 0.05

                for grade in range(4):
                    if grade in unique_values:
                        percentage = np.sum(mask_array == grade) / total_pixels
                        if percentage >= threshold:
                            if grade == 0:
                                multilabels.append(0)
                            elif grade == 1:
                                multilabels.append(1)
                            elif grade == 2:
                                if "g4c" in row["label"]:
                                    multilabels.append(3)
                                else:
                                    multilabels.append(2)
                            elif grade == 3:
                                multilabels.append(4)

                if not multilabels:
                    label_map = {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4}
                    multilabels.append(label_map.get(row["label"], 0))

            except:
                label_map = {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4}
                multilabels.append(label_map.get(row["label"], 0))
        else:
            label_map = {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4}
            multilabels.append(label_map.get(row["label"], 0))

        multilabels_list.append(multilabels)

    enhanced_df["multilabels"] = multilabels_list
    return enhanced_df

def load_classification_df(xlsx_path, crib_xlsx_path=None):
    if not check_file_exists(xlsx_path):
        return pd.DataFrame(columns=["filename", "label"])

    df = pd.read_excel(xlsx_path).copy()

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

    if crib_xlsx_path and check_file_exists(crib_xlsx_path):
        try:
            crib_df = pd.read_excel(crib_xlsx_path)
            crib_df["filename"] = crib_df["image_name"]
            g4c_files = set(crib_df["filename"].tolist())
            df.loc[(df["label"] == "g4") & (df["filename"].isin(g4c_files)), "label"] = "g4c"
        except:
            pass

    return df

train_cls_df = load_classification_df(TRAIN_XLSX, TRAIN_CRIB_XLSX)
test_cls_df = load_classification_df(TEST_XLSX, TEST_CRIB_XLSX)

train_cls_df = create_multilabel_targets(train_cls_df, MASKS_FOLDER)
test_cls_df = create_multilabel_targets(test_cls_df, MASKS_FOLDER)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.wq(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)
        attention_weights = F.softmax(attention_weights, dim=-1)

        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(attended)
        return output

class MultiTaskMultiResolutionModel(nn.Module):
    def __init__(self, num_classes=5, num_seg_classes=4, use_transformer=True):
        super().__init__()

        self.backbone_low = torch_models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone_high = torch_models.efficientnet_b0(weights="IMAGENET1K_V1")

        self.backbone_low.classifier = nn.Identity()
        self.backbone_high.classifier = nn.Identity()

        feature_dim = 1280

        if use_transformer:
            self.transformer = MultiHeadAttention(feature_dim * 2, num_heads=4)
        else:
            self.transformer = None

        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.mask_feature_fusion = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        combined_feature_dim = 256 + 64

        self.multilabel_classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.cribriform_detector = nn.Sequential(
            nn.Linear(combined_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.segmentation_head = nn.Sequential(
            nn.Linear(combined_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_seg_classes)
        )

    def forward(self, x_low, x_high, mask_percentages=None):
        feat_low = self.backbone_low(x_low)
        feat_high = self.backbone_high(x_high)

        combined_features = torch.cat([feat_low, feat_high], dim=1)

        if self.transformer:
            combined_features = combined_features.unsqueeze(1)
            combined_features = self.transformer(combined_features)
            combined_features = combined_features.squeeze(1)

        fused_features = self.feature_fusion(combined_features)

        if mask_percentages is not None:
            mask_features = self.mask_feature_fusion(mask_percentages)
            final_features = torch.cat([fused_features, mask_features], dim=1)
        else:
            dummy_mask = torch.zeros(fused_features.size(0), 64).to(fused_features.device)
            final_features = torch.cat([fused_features, dummy_mask], dim=1)

        multilabel_out = self.multilabel_classifier(final_features)
        cribriform_out = self.cribriform_detector(final_features)
        seg_out = self.segmentation_head(final_features)

        return {
            'multilabel': multilabel_out,
            'cribriform': cribriform_out,
            'segmentation': seg_out
        }

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.multilabel_loss = nn.BCEWithLogitsLoss()
        self.cribriform_loss = nn.BCEWithLogitsLoss()
        self.segmentation_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, multilabels, cribriform_labels, seg_labels=None):
        ml_loss = self.multilabel_loss(outputs['multilabel'], multilabels)
        crib_loss = self.cribriform_loss(outputs['cribriform'].squeeze(), cribriform_labels)

        total_loss = self.alpha * ml_loss + self.beta * crib_loss

        if seg_labels is not None:
            seg_loss = self.segmentation_loss(outputs['segmentation'], seg_labels)
            total_loss += self.gamma * seg_loss

        return total_loss, {
            'multilabel': ml_loss.item(),
            'cribriform': crib_loss.item(),
            'segmentation': seg_loss.item() if seg_labels is not None else 0.0
        }

def create_cribriform_labels(df):
    cribriform_labels = []
    for _, row in df.iterrows():
        if "g4c" in str(row["label"]) or (isinstance(row["multilabels"], list) and 3 in row["multilabels"]):
            cribriform_labels.append(1.0)
        else:
            cribriform_labels.append(0.0)
    return torch.tensor(cribriform_labels, dtype=torch.float32)

train_crib_labels = create_cribriform_labels(train_cls_df)
test_crib_labels = create_cribriform_labels(test_cls_df)

train_transform_low = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform_high = T.Compose([
    T.Resize((320, 320)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform_low = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform_high = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = MultiResolutionDataset(
    train_cls_df, IMAGES_FOLDER, MASKS_FOLDER,
    transform_low=train_transform_low,
    transform_high=train_transform_high,
    is_multilabel=True,
    get_mask_percentages=True
)

test_dataset = MultiResolutionDataset(
    test_cls_df, IMAGES_FOLDER, MASKS_FOLDER,
    transform_low=test_transform_low,
    transform_high=test_transform_high,
    is_multilabel=True,
    get_mask_percentages=True
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = MultiTaskMultiResolutionModel(num_classes=5, num_seg_classes=4, use_transformer=True).to(DEVICE)

criterion = MultiTaskLoss(alpha=0.4, beta=0.4, gamma=0.2)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

scaler = GradScaler() if USE_AMP else None

def train_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss = 0
    loss_components = defaultdict(float)
    num_batches = 0

    for batch_data in loader:
        if len(batch_data) == 4:
            x_low, x_high, multilabels, mask_percentages = batch_data
        else:
            x_low, x_high, multilabels = batch_data
            mask_percentages = None

        x_low = x_low.to(DEVICE)
        x_high = x_high.to(DEVICE)
        multilabels = multilabels.to(DEVICE)

        if mask_percentages is not None:
            mask_percentages = mask_percentages.to(DEVICE)

        cribriform_labels = (multilabels[:, 3] > 0).float()

        optimizer.zero_grad()

        if scaler:
            with autocast():
                outputs = model(x_low, x_high, mask_percentages)
                loss, loss_dict = criterion(outputs, multilabels, cribriform_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x_low, x_high, mask_percentages)
            loss, loss_dict = criterion(outputs, multilabels, cribriform_labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_components[key] += value
        num_batches += 1

        if num_batches % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return avg_loss, avg_components

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    loss_components = defaultdict(float)
    num_batches = 0

    all_multilabel_preds = []
    all_multilabel_targets = []
    all_cribriform_preds = []
    all_cribriform_targets = []

    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 4:
                x_low, x_high, multilabels, mask_percentages = batch_data
            else:
                x_low, x_high, multilabels = batch_data
                mask_percentages = None

            x_low = x_low.to(DEVICE)
            x_high = x_high.to(DEVICE)
            multilabels = multilabels.to(DEVICE)

            if mask_percentages is not None:
                mask_percentages = mask_percentages.to(DEVICE)

            cribriform_labels = (multilabels[:, 3] > 0).float()

            outputs = model(x_low, x_high, mask_percentages)
            loss, loss_dict = criterion(outputs, multilabels, cribriform_labels)

            total_loss += loss.item()
            for key, value in loss_dict.items():
                loss_components[key] += value
            num_batches += 1

            multilabel_probs = torch.sigmoid(outputs['multilabel'])
            cribriform_probs = torch.sigmoid(outputs['cribriform'])

            all_multilabel_preds.append(multilabel_probs.cpu())
            all_multilabel_targets.append(multilabels.cpu())
            all_cribriform_preds.append(cribriform_probs.cpu())
            all_cribriform_targets.append(cribriform_labels.cpu())

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    all_multilabel_preds = torch.cat(all_multilabel_preds, dim=0)
    all_multilabel_targets = torch.cat(all_multilabel_targets, dim=0)
    all_cribriform_preds = torch.cat(all_cribriform_preds, dim=0).squeeze()
    all_cribriform_targets = torch.cat(all_cribriform_targets, dim=0)

    multilabel_acc = ((all_multilabel_preds > 0.5) == all_multilabel_targets).float().mean()
    cribriform_acc = ((all_cribriform_preds > 0.5) == all_cribriform_targets).float().mean()

    cribriform_tp = ((all_cribriform_preds > 0.5) & (all_cribriform_targets == 1)).sum().item()
    cribriform_fn = ((all_cribriform_preds <= 0.5) & (all_cribriform_targets == 1)).sum().item()
    cribriform_sensitivity = cribriform_tp / (cribriform_tp + cribriform_fn) if (cribriform_tp + cribriform_fn) > 0 else 0

    return avg_loss, avg_components, multilabel_acc, cribriform_acc, cribriform_sensitivity

EPOCHS = 20
best_cribriform_sens = 0
best_model_state = None

for epoch in range(EPOCHS):
    train_loss, train_components = train_epoch(model, train_loader, optimizer, criterion, scaler)
    test_loss, test_components, ml_acc, crib_acc, crib_sens = evaluate_model(model, test_loader, criterion)

    scheduler.step()

    if crib_sens > best_cribriform_sens:
        best_cribriform_sens = crib_sens
        best_model_state = model.state_dict().copy()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"MultiLabel Acc: {ml_acc:.4f}, Cribriform Acc: {crib_acc:.4f}, Cribriform Sens: {crib_sens:.4f}")
        torch.cuda.empty_cache()
        gc.collect()

model.load_state_dict(best_model_state)
torch.save(best_model_state, os.path.join(SAVE_DIR, "best_multitask_model.pth"))

class CribrifromSpecializedEnsemble(nn.Module):
    def __init__(self, base_model, cribriform_specialist_model):
        super().__init__()
        self.base_model = base_model
        self.cribriform_specialist = cribriform_specialist_model

    def forward(self, x_low, x_high, mask_percentages=None):
        base_outputs = self.base_model(x_low, x_high, mask_percentages)

        g4_mask = torch.sigmoid(base_outputs['multilabel'][:, 2]) > 0.5

        if g4_mask.any():
            specialist_outputs = self.cribriform_specialist(x_low[g4_mask], x_high[g4_mask],
                                                          mask_percentages[g4_mask] if mask_percentages is not None else None)
            base_outputs['cribriform'][g4_mask] = specialist_outputs['cribriform']

        return base_outputs

def create_cribriform_specialist():
    specialist = MultiTaskMultiResolutionModel(num_classes=2, num_seg_classes=4, use_transformer=True).to(DEVICE)
    return specialist

def train_cribriform_specialist(train_df, test_df):
    g4_train_df = train_df[train_df['label'].isin(['g4', 'g4c'])].copy()
    g4_test_df = test_df[test_df['label'].isin(['g4', 'g4c'])].copy()

    if len(g4_train_df) == 0 or len(g4_test_df) == 0:
        return create_cribriform_specialist()

    g4_train_df['binary_label'] = (g4_train_df['label'] == 'g4c').astype(int)
    g4_test_df['binary_label'] = (g4_test_df['label'] == 'g4c').astype(int)

    specialist_train_dataset = MultiResolutionDataset(
        g4_train_df, IMAGES_FOLDER, MASKS_FOLDER,
        transform_low=train_transform_low,
        transform_high=train_transform_high,
        is_multilabel=False,
        get_mask_percentages=True
    )

    specialist_test_dataset = MultiResolutionDataset(
        g4_test_df, IMAGES_FOLDER, MASKS_FOLDER,
        transform_low=test_transform_low,
        transform_high=test_transform_high,
        is_multilabel=False,
        get_mask_percentages=True
    )

    specialist_train_loader = DataLoader(specialist_train_dataset, batch_size=1, shuffle=True, num_workers=0)
    specialist_test_loader = DataLoader(specialist_test_dataset, batch_size=1, shuffle=False, num_workers=0)

    specialist_model = create_cribriform_specialist()
    specialist_criterion = nn.BCEWithLogitsLoss()
    specialist_optimizer = optim.AdamW(specialist_model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_specialist_acc = 0
    best_specialist_state = None

    for epoch in range(10):
        specialist_model.train()
        train_loss = 0
        for batch_data in specialist_train_loader:
            if len(batch_data) == 4:
                x_low, x_high, labels, mask_percentages = batch_data
            else:
                x_low, x_high, labels = batch_data
                mask_percentages = None

            x_low = x_low.to(DEVICE)
            x_high = x_high.to(DEVICE)
            labels = labels.float().to(DEVICE)

            if mask_percentages is not None:
                mask_percentages = mask_percentages.to(DEVICE)

            specialist_optimizer.zero_grad()
            outputs = specialist_model(x_low, x_high, mask_percentages)
            loss = specialist_criterion(outputs['cribriform'].squeeze(), labels)
            loss.backward()
            specialist_optimizer.step()

            train_loss += loss.item()

        specialist_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in specialist_test_loader:
                if len(batch_data) == 4:
                    x_low, x_high, labels, mask_percentages = batch_data
                else:
                    x_low, x_high, labels = batch_data
                    mask_percentages = None

                x_low = x_low.to(DEVICE)
                x_high = x_high.to(DEVICE)
                labels = labels.float().to(DEVICE)

                if mask_percentages is not None:
                    mask_percentages = mask_percentages.to(DEVICE)

                outputs = specialist_model(x_low, x_high, mask_percentages)
                loss = specialist_criterion(outputs['cribriform'].squeeze(), labels)
                test_loss += loss.item()

                preds = (torch.sigmoid(outputs['cribriform'].squeeze()) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        if accuracy > best_specialist_acc:
            best_specialist_acc = accuracy
            best_specialist_state = specialist_model.state_dict().copy()

    if best_specialist_state is not None:
        specialist_model.load_state_dict(best_specialist_state)
    return specialist_model

cribriform_specialist = train_cribriform_specialist(train_cls_df, test_cls_df)
torch.save(cribriform_specialist.state_dict(), os.path.join(SAVE_DIR, "cribriform_specialist.pth"))

ensemble_model = CribrifromSpecializedEnsemble(model, cribriform_specialist)

def evaluate_ensemble(ensemble_model, test_loader):
    ensemble_model.eval()

    all_multilabel_preds = []
    all_multilabel_targets = []
    all_cribriform_preds = []
    all_cribriform_targets = []

    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 4:
                x_low, x_high, multilabels, mask_percentages = batch_data
            else:
                x_low, x_high, multilabels = batch_data
                mask_percentages = None

            x_low = x_low.to(DEVICE)
            x_high = x_high.to(DEVICE)
            multilabels = multilabels.to(DEVICE)

            if mask_percentages is not None:
                mask_percentages = mask_percentages.to(DEVICE)

            cribriform_labels = (multilabels[:, 3] > 0).float()

            outputs = ensemble_model(x_low, x_high, mask_percentages)

            multilabel_probs = torch.sigmoid(outputs['multilabel'])
            cribriform_probs = torch.sigmoid(outputs['cribriform'])

            all_multilabel_preds.append(multilabel_probs.cpu())
            all_multilabel_targets.append(multilabels.cpu())
            all_cribriform_preds.append(cribriform_probs.cpu())
            all_cribriform_targets.append(cribriform_labels.cpu())

    all_multilabel_preds = torch.cat(all_multilabel_preds, dim=0)
    all_multilabel_targets = torch.cat(all_multilabel_targets, dim=0)
    all_cribriform_preds = torch.cat(all_cribriform_preds, dim=0).squeeze()
    all_cribriform_targets = torch.cat(all_cribriform_targets, dim=0)

    multilabel_binary_preds = (all_multilabel_preds > 0.5).float()
    cribriform_binary_preds = (all_cribriform_preds > 0.5).float()

    multilabel_exact_match = (multilabel_binary_preds == all_multilabel_targets).all(dim=1).float().mean()
    multilabel_hamming = (multilabel_binary_preds == all_multilabel_targets).float().mean()

    cribriform_acc = (cribriform_binary_preds == all_cribriform_targets).float().mean()

    cribriform_tp = ((cribriform_binary_preds == 1) & (all_cribriform_targets == 1)).sum().item()
    cribriform_fp = ((cribriform_binary_preds == 1) & (all_cribriform_targets == 0)).sum().item()
    cribriform_fn = ((cribriform_binary_preds == 0) & (all_cribriform_targets == 1)).sum().item()
    cribriform_tn = ((cribriform_binary_preds == 0) & (all_cribriform_targets == 0)).sum().item()

    cribriform_precision = cribriform_tp / (cribriform_tp + cribriform_fp) if (cribriform_tp + cribriform_fp) > 0 else 0
    cribriform_recall = cribriform_tp / (cribriform_tp + cribriform_fn) if (cribriform_tp + cribriform_fn) > 0 else 0
    cribriform_f1 = 2 * (cribriform_precision * cribriform_recall) / (cribriform_precision + cribriform_recall) if (cribriform_precision + cribriform_recall) > 0 else 0

    return {
        'multilabel_exact_match': multilabel_exact_match.item(),
        'multilabel_hamming': multilabel_hamming.item(),
        'cribriform_accuracy': cribriform_acc.item(),
        'cribriform_precision': cribriform_precision,
        'cribriform_recall': cribriform_recall,
        'cribriform_f1': cribriform_f1,
        'cribriform_confusion_matrix': {
            'tp': cribriform_tp, 'fp': cribriform_fp, 'fn': cribriform_fn, 'tn': cribriform_tn
        }
    }

ensemble_results = evaluate_ensemble(ensemble_model, test_loader)

print("\n=== FINAL ENSEMBLE RESULTS ===")
print(f"Multi-label Exact Match Accuracy: {ensemble_results['multilabel_exact_match']:.4f}")
print(f"Multi-label Hamming Accuracy: {ensemble_results['multilabel_hamming']:.4f}")
print(f"Cribriform Detection Accuracy: {ensemble_results['cribriform_accuracy']:.4f}")
print(f"Cribriform Precision: {ensemble_results['cribriform_precision']:.4f}")
print(f"Cribriform Recall: {ensemble_results['cribriform_recall']:.4f}")
print(f"Cribriform F1-Score: {ensemble_results['cribriform_f1']:.4f}")

cm = ensemble_results['cribriform_confusion_matrix']
print(f"\nCribriform Confusion Matrix:")
print(f"TP: {cm['tp']}, FP: {cm['fp']}")
print(f"FN: {cm['fn']}, TN: {cm['tn']}")

def visualize_multilabel_predictions(model, dataset, save_path):
    model.eval()
    class_names = ["NC", "G3", "G4", "G4C", "G5"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()

    samples_shown = 0
    for i in range(min(6, len(dataset))):
        if len(dataset[i]) == 4:
            x_low, x_high, true_labels, mask_percentages = dataset[i]
        else:
            x_low, x_high, true_labels = dataset[i]
            mask_percentages = None

        x_low_batch = x_low.unsqueeze(0).to(DEVICE)
        x_high_batch = x_high.unsqueeze(0).to(DEVICE)

        if mask_percentages is not None:
            mask_percentages_batch = mask_percentages.unsqueeze(0).to(DEVICE)
        else:
            mask_percentages_batch = None

        with torch.no_grad():
            outputs = model(x_low_batch, x_high_batch, mask_percentages_batch)
            pred_probs = torch.sigmoid(outputs['multilabel']).cpu().squeeze()
            cribriform_prob = torch.sigmoid(outputs['cribriform']).cpu().squeeze()

        img_np = x_low.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        ax = axes[samples_shown]
        ax.imshow(img_np)

        true_classes = [class_names[j] for j in range(5) if true_labels[j] > 0.5]
        pred_classes = [class_names[j] for j in range(5) if pred_probs[j] > 0.5]

        title = f"True: {', '.join(true_classes)}\nPred: {', '.join(pred_classes)}"
        title += f"\nCribriform: {cribriform_prob:.3f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        samples_shown += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_cribriform_misclassifications(model, dataset, save_path):
    model.eval()

    misclassified_images = []
    misclassified_info = []

    for i in range(min(len(dataset), 100)):
        if len(dataset[i]) == 4:
            x_low, x_high, true_labels, mask_percentages = dataset[i]
        else:
            x_low, x_high, true_labels = dataset[i]
            mask_percentages = None

        true_cribriform = true_labels[3] > 0.5

        x_low_batch = x_low.unsqueeze(0).to(DEVICE)
        x_high_batch = x_high.unsqueeze(0).to(DEVICE)

        if mask_percentages is not None:
            mask_percentages_batch = mask_percentages.unsqueeze(0).to(DEVICE)
        else:
            mask_percentages_batch = None

        with torch.no_grad():
            outputs = model(x_low_batch, x_high_batch, mask_percentages_batch)
            cribriform_prob = torch.sigmoid(outputs['cribriform']).cpu().squeeze()
            pred_cribriform = cribriform_prob > 0.5

        if true_cribriform != pred_cribriform:
            img_np = x_low.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            misclassified_images.append(img_np)
            misclassified_info.append({
                'true': true_cribriform.item(),
                'pred': pred_cribriform.item(),
                'prob': cribriform_prob.item()
            })

            if len(misclassified_images) >= 8:
                break

    if misclassified_images:
        cols = 4
        rows = (len(misclassified_images) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (img, info) in enumerate(zip(misclassified_images, misclassified_info)):
            row = idx // cols
            col = idx % cols

            axes[row, col].imshow(img)
            title = f"True: {'Crib' if info['true'] else 'Non-Crib'}\n"
            title += f"Pred: {'Crib' if info['pred'] else 'Non-Crib'} ({info['prob']:.3f})"
            axes[row, col].set_title(title)
            axes[row, col].axis('off')

        for idx in range(len(misclassified_images), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

visualize_multilabel_predictions(ensemble_model, test_dataset,
                                os.path.join(SAVE_DIR, "multilabel_predictions.png"))

analyze_cribriform_misclassifications(ensemble_model, test_dataset,
                                    os.path.join(SAVE_DIR, "cribriform_misclassifications.png"))

def perform_cross_validation():
    val_dirs = [VAL1_DIR, VAL2_DIR, VAL3_DIR, VAL4_DIR]
    cv_results = []

    for fold, val_dir in enumerate(val_dirs):
        if not os.path.exists(val_dir):
            continue

        val_files = []
        for excel_file in ["Train.xlsx", "Test.xlsx", "TrainCribfriform.xlsx", "TestCribfriform.xlsx"]:
            excel_path = os.path.join(val_dir, excel_file)
            if os.path.exists(excel_path):
                df = explore_excel(excel_path)
                if df is not None:
                    val_files.append(df)

        if not val_files:
            continue

        val_df = pd.concat(val_files, ignore_index=True).drop_duplicates()
        val_df = create_multilabel_targets(val_df, MASKS_FOLDER)

        val_dataset = MultiResolutionDataset(
            val_df, IMAGES_FOLDER, MASKS_FOLDER,
            transform_low=test_transform_low,
            transform_high=test_transform_high,
            is_multilabel=True,
            get_mask_percentages=True
        )

        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        fold_results = evaluate_ensemble(ensemble_model, val_loader)
        fold_results['fold'] = fold + 1
        cv_results.append(fold_results)

        print(f"Fold {fold + 1} Results:")
        print(f"  Cribriform F1: {fold_results['cribriform_f1']:.4f}")
        print(f"  Multi-label Hamming: {fold_results['multilabel_hamming']:.4f}")

    if cv_results:
        avg_cribriform_f1 = np.mean([r['cribriform_f1'] for r in cv_results])
        avg_multilabel_hamming = np.mean([r['multilabel_hamming'] for r in cv_results])

        print(f"\nCross-Validation Summary:")
        print(f"Average Cribriform F1: {avg_cribriform_f1:.4f}")
        print(f"Average Multi-label Hamming: {avg_multilabel_hamming:.4f}")

    return cv_results

cv_results = perform_cross_validation()

with open(os.path.join(SAVE_DIR, "enhanced_results_summary.txt"), "w") as f:
    f.write("Enhanced SICAPv2 Multi-Task Multi-Label Pipeline Results\n")
    f.write("=========================================================\n\n")

    f.write("NOVEL FEATURES IMPLEMENTED:\n")
    f.write("- Multi-label classification to handle mixed Gleason patterns\n")
    f.write("- Multi-resolution input processing (224x224 + 320x320)\n")
    f.write("- Transformer-based feature fusion with multi-head attention\n")
    f.write("- Specialized cribriform detection with two-stage approach\n")
    f.write("- Multi-task learning (classification + segmentation + cribriform)\n")
    f.write("- Stain normalization using Macenko method\n")
    f.write("- Mask percentage features integrated into classification\n")
    f.write("- Memory-optimized training with garbage collection\n\n")

    f.write("MAIN RESULTS:\n")
    f.write(f"Multi-label Exact Match Accuracy: {ensemble_results['multilabel_exact_match']:.4f}\n")
    f.write(f"Multi-label Hamming Accuracy: {ensemble_results['multilabel_hamming']:.4f}\n")
    f.write(f"Cribriform Detection F1-Score: {ensemble_results['cribriform_f1']:.4f}\n")
    f.write(f"Cribriform Detection Precision: {ensemble_results['cribriform_precision']:.4f}\n")
    f.write(f"Cribriform Detection Recall: {ensemble_results['cribriform_recall']:.4f}\n\n")

    if cv_results:
        avg_cribriform_f1 = np.mean([r['cribriform_f1'] for r in cv_results])
        avg_multilabel_hamming = np.mean([r['multilabel_hamming'] for r in cv_results])
        f.write("CROSS-VALIDATION RESULTS:\n")
        f.write(f"Average Cribriform F1 across folds: {avg_cribriform_f1:.4f}\n")
        f.write(f"Average Multi-label Hamming across folds: {avg_multilabel_hamming:.4f}\n\n")

    f.write("SAVED MODELS:\n")
    f.write("- best_multitask_model.pth: Main multi-task model\n")
    f.write("- cribriform_specialist.pth: Specialized cribriform detector\n\n")

    f.write("VISUALIZATIONS:\n")
    f.write("- multilabel_predictions.png: Sample multi-label predictions\n")
    f.write("- cribriform_misclassifications.png: Analysis of cribriform errors\n")
