"""
Stage 1: Patch-Level Weak Supervision Fine-Tuning
=================================================
Fine-tunes a convnext_tiny backbone on tissue patches.
Creates 5 separate backbones to prevent data leakage for the 5 folds.
"""

import os
import re
import gc
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.metrics import balanced_accuracy_score

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
IS_LIGHTNING = os.path.exists('/teamspace')
try:
    workspace = Path(__file__).resolve().parent
except NameError:
    workspace = Path.cwd().resolve()

def find_first_dir(search_roots, dirname):
    for base in search_roots:
        if not base or not os.path.exists(base): continue
        for root, dirs, _ in os.walk(base):
            if dirname in dirs: return os.path.join(root, dirname)
    return None

def find_first_file(search_roots, filename):
    for base in search_roots:
        if not base or not os.path.exists(base): continue
        for root, _, files in os.walk(base):
            if filename in files: return os.path.join(root, filename)
    return None

if IS_LIGHTNING:
    LIGHTNING_WORK_DIR = Path('/teamspace/studios/this_studio')
    DATA_SEARCH_ROOTS = [str(workspace), '/teamspace/datasets', '/teamspace/studios/this_studio/ebt-22']
    ORIG_DATA_DIR = find_first_dir(DATA_SEARCH_ROOTS, 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations_fixed.csv')
    if ANNOTATIONS_CSV is None: ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations.csv')
    AUG_TRAIN_DIR = str(LIGHTNING_WORK_DIR / 'v3_augmented_all')
    AUG_TRAIN_MANIFEST = str(LIGHTNING_WORK_DIR / 'v3_augmented_all_manifest.csv')
    OUTPUT_DIR = str(LIGHTNING_WORK_DIR / 'output_stage1')
else:
    DATA_SEARCH_ROOTS = [str(workspace), str(workspace.parent), str(workspace / 'Data'), str(workspace.parent / 'Data')]
    ORIG_DATA_DIR = find_first_dir(DATA_SEARCH_ROOTS, 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations_fixed.csv')
    if ANNOTATIONS_CSV is None: ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations.csv')
    _aug_manifest = workspace / 'augmented_data_22' / 'augmented_data_22' / 'combined_manifest.csv'
    if _aug_manifest.exists():
        AUG_TRAIN_MANIFEST = str(_aug_manifest)
        AUG_TRAIN_DIR = str(workspace / 'augmented_data_22' / 'augmented_data_22')
    else:
        AUG_TRAIN_MANIFEST = str(workspace / 'v3_augmented_all_manifest.csv')
        AUG_TRAIN_DIR = str(workspace / 'v3_augmented_all')
    OUTPUT_DIR = str(workspace / 'output_stage1')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Patching config
IMAGE_RESIZE = 512
PATCH_SCALE_CHOICES = [96, 128, 192]
PATCH_OUTPUT_SIZE = 224
MIN_TISSUE = 0.40

# Training config
BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

LABEL_MAP = {'HGC': 0, 'LGC': 1, 'NST': 2, 'NTL': 2, 'Normal': 2}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
IMAGE_PATH_INDEX = {}

def resolve_manifest_path(row, orig_dir, aug_dir):
    import os
    full_path = str(row.get('full_path', ''))
    is_aug = row.get('is_augmented', False)
    
    if is_aug:
        candidate = os.path.join(aug_dir, full_path)
        if os.path.exists(candidate): return candidate
    else:
        candidate = os.path.join(orig_dir, full_path)
        if os.path.exists(candidate): return candidate
    
    filename = os.path.basename(full_path).lower()
    if filename in IMAGE_PATH_INDEX: return IMAGE_PATH_INDEX[filename]
    
    return None

def scan_for_images():
    global IMAGE_PATH_INDEX
    print("SCANNING FILESYSTEM")
    search_dirs = [str(Path(ORIG_DATA_DIR).parent), AUG_TRAIN_DIR]
    count = 0
    for d in search_dirs:
        if not os.path.exists(d): continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    IMAGE_PATH_INDEX[f.lower()] = os.path.join(root, f)
                    count += 1
    print(f"  Indexed {count} images")

def extract_patient_id(filename):
    s = str(filename)
    s = re.sub(r'_aug\d+\.png$', '', s)
    s = re.sub(r'_orig\.png$', '', s)
    s = re.sub(r'_(WLI2NBI|NBI2WLI)\.png$', '', s)
    s = re.sub(r'\.png$', '', s)
    for pat in [r'case_(\d+)', r'cys_case_(\d+)']:
        m = re.search(pat, s)
        if m: return int(m.group(1))
    return -1

# ---------------------------------------------------------
# ON-THE-FLY PATCH DATASET
# ---------------------------------------------------------
def compute_quality(patch_bgr):
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    mask = (v < 245) & (v > 15) & (s > 10)
    tissue_frac = mask.sum() / mask.size
    return tissue_frac >= MIN_TISSUE

class WeakPatchDataset(Dataset):
    def __init__(self, df, patches_per_epoch_per_image=1):
        self.df = df.reset_index(drop=True)
        self.indices = np.repeat(np.arange(len(self.df)), patches_per_epoch_per_image)
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.indices[idx]]
        path = row['path']
        label = row['label']

        img = cv2.imread(str(path))
        if img is None:
            return torch.zeros((3, PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE)), label
        
        h, w = img.shape[:2]
        s = IMAGE_RESIZE / max(h, w)
        if s != 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        
        H, W = img.shape[:2]
        
        patch = None
        for _ in range(10): # Max 10 attempts
            scale = random.choice(PATCH_SCALE_CHOICES)
            if scale > min(H, W): scale = min(H, W)
            y = random.randint(0, H - scale)
            x = random.randint(0, W - scale)
            crop = img[y:y+scale, x:x+scale]
            if compute_quality(crop):
                patch = cv2.resize(crop, (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE))
                break
        
        if patch is None:
            min_dim = min(H, W)
            y0, x0 = (H - min_dim) // 2, (W - min_dim) // 2
            crop = img[y0:y0 + min_dim, x0:x0 + min_dim]
            patch = cv2.resize(crop, (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE))

        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        return tensor, label

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
def train_fold(fold, train_df, val_df):
    print(f"\n{'='*50}\nSTARTING STAGE 1: FOLD {fold}\n{'='*50}")
    
    counts = train_df['label'].value_counts().to_dict()
    total = sum(counts.values())
    weights = [total / counts.get(i, 1) for i in range(NUM_CLASSES)]
    weights[0] *= 2.0
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Class Weights: {weights}")

    train_ds = WeakPatchDataset(train_df, patches_per_epoch_per_image=1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    val_ds = WeakPatchDataset(val_df, patches_per_epoch_per_image=5)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val_acc = 0.0
    model_save_path = os.path.join(OUTPUT_DIR, f'convnext_finetuned_fold{fold}.pt')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs = (inputs - IMNET_MEAN) / IMNET_STD
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += inputs.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
            
        scheduler.step()
        train_acc = train_correct / train_total
        
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0
        
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                inputs = (inputs - IMNET_MEAN) / IMNET_STD
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        val_acc = balanced_accuracy_score(val_targets, val_preds)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.3f} | Val Bal-Acc: {val_acc:.3f} | Val Loss: {val_loss:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model to {model_save_path}")

    print(f"Fold {fold} finished. Best Val Acc: {best_val_acc:.3f}")
    
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    set_seed(42)
    scan_for_images()
    
    df_orig = pd.read_csv(ANNOTATIONS_CSV)
    df_orig.columns = df_orig.columns.str.strip()
    df_orig['patient_id'] = df_orig['HLY'].apply(extract_patient_id)
    df_orig['label']  = df_orig['tissue type'].map(LABEL_MAP).fillna(2).astype(int)
    df_orig['path']   = df_orig.apply(lambda r: IMAGE_PATH_INDEX.get(str(r['HLY']).strip().lower()), axis=1)
    base_df = df_orig[df_orig['path'].notna()].copy()
    
    full_manifest = pd.read_csv(AUG_TRAIN_MANIFEST)
    full_manifest['label'] = full_manifest['tissue type'].map(LABEL_MAP).fillna(2).astype(int)
    full_manifest['path'] = full_manifest.apply(lambda r: resolve_manifest_path(r, ORIG_DATA_DIR, AUG_TRAIN_DIR), axis=1)
    aug_full_df = full_manifest[full_manifest['path'].notna()].copy()

    all_patients = sorted(list(base_df['patient_id'].unique()))
    print(f"Total valid patients: {len(all_patients)}")

    folds_config = [
        (
            [0, 1, 2, 4, 5, 7, 10, 11, 12, 16, 17, 18, 22, 23, 24, 25],
            [6, 9, 14],
            [8, 13, 21]
        ),
        (
            [0, 2, 4, 6, 8, 9, 10, 11, 13, 14, 16, 17, 21, 23, 24, 25],
            [12, 18, 22],
            [1, 5, 7]
        ),
        (
            [1, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 21, 22, 24, 25],
            [4, 11, 23],
            [0, 2, 10]
        ),
        (
            [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 18, 21, 22, 23, 24],
            [16, 17, 25],
            [6, 9, 14]
        ),
        (
            [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 23, 25],
            [8, 13, 21],
            [18, 22, 24]
        )
    ]

    for fold_idx, (train_pids, val_pids, test_pids) in enumerate(folds_config, 1):
        print(f"\nChecking Fold {fold_idx}...")
        save_path = os.path.join(OUTPUT_DIR, f'convnext_finetuned_fold{fold_idx}.pt')
        if os.path.exists(save_path):
            print(f"Fold {fold_idx} already finished. Skipping.")
            continue
            
        train_df = aug_full_df[aug_full_df['patient_id'].isin(train_pids)]
        val_df = base_df[base_df['patient_id'].isin(val_pids)]
        
        train_fold(fold_idx, train_df, val_df)

if __name__ == '__main__':
    main()
