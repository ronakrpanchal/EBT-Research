"""
v3-Fixed: Bladder Classification — 5-Fold CV (Fixed & Maximized)
================================================================
- FROZEN DINOv2 + DenseNet121 (no fine-tuning)
- Tiny CLAM (teacher hidden=384, students hidden=256)
- Teacher trained on train; Students on train+val with KL-distillation
- 12-view image-space TTA at inference
- Constrained threshold tuning (HGC recall >= 92%)
- Feature-space mixup during training
- Balanced fold splits ensuring all 3 classes in every split

Bug fixes vs original:
  1. Fixed augmented manifest path resolution
  2. Fixed full_path -> actual file path mapping
  3. Fixed platform detection for local Windows
  4. Fixed patient ID extraction for augmented filenames
  5. Removed fold-skipping guard (warn instead of skip)
  6. Completely rebalanced fold splits
"""

import os
import re
import sys
import copy
import math
import json
import time
import random
import hashlib
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, recall_score, precision_score, f1_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# 1: CONFIGURATION
# ════════════════════════════════════════════════════════════
IS_KAGGLE = os.path.exists('/kaggle')
IS_LIGHTNING = os.path.exists('/teamspace')
IS_LOCAL = not IS_KAGGLE and not IS_LIGHTNING
try:
    workspace = Path(__file__).resolve().parent
except NameError:
    workspace = Path.cwd().resolve()


def find_first_dir(search_roots, dirname):
    for base in search_roots:
        if not base or not os.path.exists(base):
            continue
        for root, dirs, _ in os.walk(base):
            if dirname in dirs:
                return os.path.join(root, dirname)
    return None


def find_first_file(search_roots, filename):
    for base in search_roots:
        if not base or not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    return None


# ── Path resolution ──
if IS_KAGGLE:
    AUG_TRAIN_DIR = '/kaggle/working/v3_augmented'
    AUG_TRAIN_MANIFEST = '/kaggle/working/v3_augmented_manifest.csv'
    DATA_SEARCH_ROOTS = ['/kaggle/input', '/kaggle/working']

    DATASET_ROOT = '/kaggle/input/ebt-22'
    for root, dirs, _ in os.walk('/kaggle/input'):
        if 'EndoscopicBladderTissue' in dirs:
            DATASET_ROOT = root
            break
    ORIG_DATA_DIR = os.path.join(DATASET_ROOT, 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = None
    for root, _, files in os.walk('/kaggle/input'):
        if 'annotations_fixed.csv' in files:
            ANNOTATIONS_CSV = os.path.join(root, 'annotations_fixed.csv')
            break

    OUTPUT_DIR = '/kaggle/working/output_5foldcv_v4_convnext'
    CACHE_DIR = '/kaggle/working/feat_cache_v4'

elif IS_LIGHTNING:
    LIGHTNING_WORK_DIR = Path('/teamspace/studios/this_studio')
    if not LIGHTNING_WORK_DIR.exists():
        LIGHTNING_WORK_DIR = workspace
    DATA_SEARCH_ROOTS = [
        str(workspace),
        '/teamspace/datasets',
        '/teamspace/studios/this_studio/ebt-22',
    ]
    ORIG_DATA_DIR = find_first_dir(DATA_SEARCH_ROOTS, 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations_fixed.csv')
    if ANNOTATIONS_CSV is None:
        ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations.csv')

    AUG_TRAIN_DIR = str(LIGHTNING_WORK_DIR / 'v3_augmented_all')
    AUG_TRAIN_MANIFEST = str(LIGHTNING_WORK_DIR / 'v3_augmented_all_manifest.csv')
    OUTPUT_DIR = str(LIGHTNING_WORK_DIR / 'output_5foldcv_v4_convnext')
    CACHE_DIR = str(LIGHTNING_WORK_DIR / 'feat_cache_v4')

else:
    # ── LOCAL (Windows / Linux) ──
    DATA_SEARCH_ROOTS = [
        str(workspace),
        str(workspace.parent),
        str(workspace / 'Data'),
        str(workspace.parent / 'Data'),
    ]
    ORIG_DATA_DIR = find_first_dir(DATA_SEARCH_ROOTS, 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations_fixed.csv')
    if ANNOTATIONS_CSV is None:
        ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations.csv')

    # FIX #1: Point to actual augmented manifest
    _aug_manifest_candidate = workspace / 'augmented_data_22' / 'augmented_data_22' / 'combined_manifest.csv'
    if _aug_manifest_candidate.exists():
        AUG_TRAIN_MANIFEST = str(_aug_manifest_candidate)
        AUG_TRAIN_DIR = str(workspace / 'augmented_data_22' / 'augmented_data_22')
    else:
        AUG_TRAIN_MANIFEST = str(workspace / 'v3_augmented_all_manifest.csv')
        AUG_TRAIN_DIR = str(workspace / 'v3_augmented_all')

    OUTPUT_DIR = str(workspace / 'output_5foldcv_v4_convnext')
    CACHE_DIR = str(workspace / 'feat_cache_v4')

if ORIG_DATA_DIR is None:
    raise FileNotFoundError(f"Could not find EndoscopicBladderTissue under: {DATA_SEARCH_ROOTS}")
if ANNOTATIONS_CSV is None:
    raise FileNotFoundError(f"Could not find annotations CSV under: {DATA_SEARCH_ROOTS}")

print(f"[CONFIG] ORIG_DATA_DIR:      {ORIG_DATA_DIR}")
print(f"[CONFIG] ANNOTATIONS_CSV:    {ANNOTATIONS_CSV}")
print(f"[CONFIG] AUG_TRAIN_MANIFEST: {AUG_TRAIN_MANIFEST}")
print(f"[CONFIG] AUG_TRAIN_DIR:      {AUG_TRAIN_DIR}")
print(f"[CONFIG] OUTPUT_DIR:         {OUTPUT_DIR}")
print(f"[CONFIG] CACHE_DIR:          {CACHE_DIR}")

# Class config
NUM_CLASSES = 3
CLASS_NAMES = ['HGC', 'LGC', 'Normal']
LABEL_MAP = {'HGC': 'HGC', 'LGC': 'LGC', 'NST': 'Normal', 'NTL': 'Normal'}
CLASS_TO_IDX = {'HGC': 0, 'LGC': 1, 'Normal': 2}
IDX_TO_CLASS = {0: 'HGC', 1: 'LGC', 2: 'Normal'}
CANCER_CLASSES = {0, 1}

# Image preprocessing
IMAGE_RESIZE = 512
PATCH_SCALES = [96, 128, 192]
PATCH_OUTPUT_SIZE = 224
PATCH_STRIDE_FRAC = 0.5
MIN_TISSUE = 0.40
MAX_BRIGHT = 245
MIN_BRIGHT = 15
MIN_SAT = 10
MIN_FOCUS = 8.0
TOP_QUALITY_FRAC = 0.85
MAX_PATCHES_PER_IMAGE = 60
CLAHE_CLIP = 1.5
CLAHE_GRID = (16, 16)

# Feature extraction
FEAT_BATCH = 128
PATCH_BATCH_TARGET = 512
CACHE_VERSION = 'v4_convnext'

# ════════════════════════════════════════════════════════════
# BACKBONE FREEZE CONFIG
# ════════════════════════════════════════════════════════════
FREEZE_BACKBONE = True

# CLAM Configuration
MIL_HIDDEN = 128             # Bottleneck capacity
MIL_DROPOUT = 0.50           # Stronger dropout
N_ATT_HEADS = 1
CLAM_K_SAMPLE = 8
FEAT_NOISE_STD = 0.025       # More jitter to features
FEAT_DROP_P = 0.25           # Drop 25% of patch features

# Training
LR = 5e-5                    # Slightly lower learning rate
WD = 5e-3                    # Heavy weight decay (L2 penalty)
EPOCHS = 40
WARMUP_EPOCHS = 4
PATIENCE = 10
GRAD_CLIP = 1.0

# Loss
FOCAL_GAMMA = 2.0
LABEL_SMOOTH = 0.05
BAG_LOSS_W = 1.0
INST_LOSS_W = 0.05
HIER_LOSS_W = 0.10
ORDINAL_LOSS_W = 0.10

# Class weighting
HGC_WEIGHT_BOOST = 2.5

# Patch limits
MAX_PATCHES_TRAIN = 100      # Force model to use fewer, random patches per epoch
MAX_PATCHES_TEST = 400

# Image-space TTA — expanded to 12 views
N_TTA_VIEWS = 12

# Per-bag standardization
USE_PER_BAG_STD = True

# Feature-space mixup
USE_MIXUP = True
MIXUP_ALPHA = 0.3

# Device / GPU compatibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
CUDA_CAPABILITY = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
IS_T4 = torch.cuda.is_available() and ('T4' in CUDA_NAME.upper() or CUDA_CAPABILITY == (7, 5))
IS_P100 = torch.cuda.is_available() and ('P100' in CUDA_NAME.upper() or CUDA_CAPABILITY[0] == 6)
AMP_DTYPE = torch.float16
USE_AMP = torch.cuda.is_available()

if IS_T4:
    FEAT_BATCH = min(FEAT_BATCH, 64)
    PATCH_BATCH_TARGET = min(PATCH_BATCH_TARGET, 384)
    CACHE_VERSION = f'{CACHE_VERSION}_t4'

if IS_P100:
    FEAT_BATCH = min(FEAT_BATCH, 32)
    PATCH_BATCH_TARGET = min(PATCH_BATCH_TARGET, 256)
    MAX_PATCHES_TEST = min(MAX_PATCHES_TEST, 300)

# CPU fallback: skip DINOv2, use only DenseNet (much faster on CPU)
SKIP_DINO_ON_CPU = not torch.cuda.is_available()

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ════════════════════════════════════════════════════════════
# 2: DATA LOADING
# ════════════════════════════════════════════════════════════

def safe_torch_save(obj, path):
    """Write PyTorch objects atomically and avoid the zip writer used by default."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, 'wb') as f:
            torch.save(obj, f, _use_new_zipfile_serialization=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def extract_patient_id(filename):
    """Extract patient ID from filename, handling all naming conventions.
    
    FIX #4: Also strips augmentation suffixes like _WLI2NBI, _NBI2WLI
    """
    s = str(filename)
    # Strip augmentation suffixes
    s = re.sub(r'_aug\d+\.png$', '', s)
    s = re.sub(r'_orig\.png$', '', s)
    s = re.sub(r'_(WLI2NBI|NBI2WLI)\.png$', '', s)
    s = re.sub(r'\.png$', '', s)
    
    for pat in [r'case_(\d+)', r'cys_case_(\d+)']:
        m = re.search(pat, s)
        if m:
            return int(m.group(1))
    return -1


IMAGE_PATH_INDEX = {}


def scan_for_images():
    """Build a global index of all image files for fast lookup."""
    global IMAGE_PATH_INDEX
    print("" + "=" * 60)
    print("SCANNING FILESYSTEM")
    print("=" * 60)
    
    search_dirs = []
    if IS_KAGGLE:
        search_dirs = ['/kaggle/input', '/kaggle/working']
    else:
        # FIX #3: Include both original and augmented data directories
        search_dirs = [
            str(Path(ORIG_DATA_DIR).parent),
            AUG_TRAIN_DIR,
        ]
    
    count = 0
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    IMAGE_PATH_INDEX[f.lower()] = os.path.join(root, f)
                    count += 1
    print(f"  Indexed {count} images")


# ════════════════════════════════════════════════════════════
class LabNormalizer:
    def __init__(self):
        self.ref = None

    def fit(self, images_bgr):
        stats = {'L': [], 'a': [], 'b': []}
        for img in images_bgr:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            for i, ch in enumerate(['L', 'a', 'b']):
                stats[ch].append({'m': lab[:, :, i].mean(), 's': lab[:, :, i].std() + 1e-6})
        self.ref = {ch: {'m': np.median([s['m'] for s in stats[ch]]),
                         's': np.median([s['s'] for s in stats[ch]])} for ch in ['L', 'a', 'b']}
        return self

    def transform(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        for i, ch in enumerate(['L', 'a', 'b']):
            c = lab[:, :, i]
            sm, ss = c.mean(), c.std() + 1e-6
            lab[:, :, i] = np.clip((c - sm) * (self.ref[ch]['s'] / ss) + self.ref[ch]['m'], 0, 255)
        lab = lab.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_image(path, norm=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = IMAGE_RESIZE / max(h, w)
    if s != 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if norm:
        img = norm.transform(img)
    return img


def compute_quality(patch_bgr):
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    mask = (v < MAX_BRIGHT) & (v > MIN_BRIGHT) & (s > MIN_SAT)
    tissue_frac = mask.sum() / mask.size
    if tissue_frac < MIN_TISSUE:
        return -1.0
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    focus = cv2.Laplacian(gray, cv2.CV_64F).var()
    if focus < MIN_FOCUS:
        return -1.0
    focus_norm = min(focus / 100.0, 1.0)
    sat_std = s[mask].std() / 50.0 if mask.sum() > 10 else 0
    sat_norm = min(sat_std, 1.0)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = min(edges.sum() / (255.0 * edges.size) * 10, 1.0)
    return 0.3 * tissue_frac + 0.3 * focus_norm + 0.2 * sat_norm + 0.2 * edge_density


def extract_multiscale_patches(image_bgr, max_patches=None):
    if max_patches is None:
        max_patches = MAX_PATCHES_PER_IMAGE
    H, W = image_bgr.shape[:2]
    candidates = []
    cap = max_patches * 3
    for scale in PATCH_SCALES:
        if scale > min(H, W):
            continue
        stride = max(1, int(scale * PATCH_STRIDE_FRAC))
        for y in range(0, H - scale + 1, stride):
            for x in range(0, W - scale + 1, stride):
                if len(candidates) >= cap:
                    break
                crop = image_bgr[y:y + scale, x:x + scale]
                q = compute_quality(crop)
                if q > 0:
                    resized = cv2.resize(crop, (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
                    candidates.append((resized, q, scale))
            if len(candidates) >= cap:
                break
    if not candidates:
        min_dim = min(H, W)
        y0, x0 = (H - min_dim) // 2, (W - min_dim) // 2
        crop = image_bgr[y0:y0 + min_dim, x0:x0 + min_dim]
        return [cv2.resize(crop, (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE))]
    candidates.sort(key=lambda x: x[1], reverse=True)
    n_keep = max(1, int(len(candidates) * TOP_QUALITY_FRAC))
    candidates = candidates[:n_keep][:max_patches]
    return [c[0] for c in candidates]


def fit_normalizer(df):
    print("  Fitting LAB normalizer...")
    samples = []
    sample_paths = df.sample(min(50, len(df)), random_state=42).path.values
    for fp in sample_paths:
        try:
            img = cv2.imread(str(fp))
            if img is not None:
                h, w = img.shape[:2]
                s = IMAGE_RESIZE / max(h, w)
                if s != 1:
                    img = cv2.resize(img, (int(w * s), int(h * s)))
                samples.append(img)
        except Exception:
            pass
    if not samples:
        return None
    norm = LabNormalizer().fit(samples)
    print(f"  ✓ Normalizer fitted on {len(samples)} images")
    return norm


# ════════════════════════════════════════════════════════════
# 4: FEATURE EXTRACTION (FROZEN BACKBONES)
# ════════════════════════════════════════════════════════════
convnext_model = None
feat_dim = 0

def load_backbones():
    global convnext_model, feat_dim, IMNET_MEAN, IMNET_STD
    print("" + "=" * 60)
    print("LOADING FROZEN CONVNEXT BACKBONE")
    print("=" * 60)
    
    IMNET_MEAN = IMNET_MEAN.to(DEVICE)
    IMNET_STD = IMNET_STD.to(DEVICE)
    
    print("  Loading convnext_tiny...")
    try:
        convnext_model = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        convnext_model.eval().to(DEVICE)
        for p in convnext_model.parameters():
            p.requires_grad = False
        feat_dim = 768
        print(f"  ✓ convnext_tiny — FROZEN, dim={feat_dim}")
    except Exception as e:
        print(f"  ⚠ ConvNeXt failed: {e}")
        raise e
    
    return feat_dim


def bgr_to_tensor(patch_bgr):
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def _get_cache_key(path, tta_idx=None):
    parts = f"{path}|{IMAGE_RESIZE}|{PATCH_SCALES}|{MAX_PATCHES_PER_IMAGE}|{CACHE_VERSION}"
    if tta_idx is not None:
        parts += f"|tta{tta_idx}"
    return hashlib.md5(parts.encode()).hexdigest()


@torch.inference_mode()
def extract_dual_features(tensor_list):
    if not tensor_list:
        return torch.empty((0, feat_dim), dtype=torch.float16)
    all_feats = []
    for i in range(0, len(tensor_list), FEAT_BATCH):
        batch = torch.stack(tensor_list[i:i + FEAT_BATCH]).to(DEVICE, non_blocking=True)
        batch_norm = (batch - IMNET_MEAN) / IMNET_STD
        with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
            feats = convnext_model(batch_norm).float().cpu()
        all_feats.append(feats)
    return torch.cat(all_feats, 0).half()


def extract_features_for_split(df, desc="Extracting", norm=None, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    results, skipped, cache_hits = [], 0, 0
    pending_tensors, pending_meta = [], []
    
    def flush():
        nonlocal pending_tensors, pending_meta
        if not pending_tensors:
            return
        feats = extract_dual_features(pending_tensors)
        start = 0
        for meta in pending_meta:
            n = meta['n_patches']
            block = feats[start:start + n]
            start += n
            if meta['cache_path']:
                try:
                    torch.save({'features': block}, meta['cache_path'])
                except Exception:
                    pass
            results.append({
                'features': meta['cache_path'] if meta['cache_path'] else block,
                'label': meta['label'],
                'label_name': meta['label_name'],
                'patient': meta['patient'],
                'path': meta['path'],
                'n_patches': block.shape[0],
            })
        pending_tensors.clear()
        pending_meta.clear()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        cache_path = None
        if use_cache:
            cache_key = _get_cache_key(str(row.path))
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pt")
            if os.path.exists(cache_path):
                try:
                    results.append({
                        'features': cache_path,
                        'label': int(row.target),
                        'label_name': row.label,
                        'patient': int(row.patient),
                        'path': row.path,
                        'n_patches': -1,
                    })
                    cache_hits += 1
                    continue
                except Exception:
                    pass
        
        try:
            img = load_image(str(row.path), norm)
        except Exception:
            skipped += 1
            continue
        
        patches = extract_multiscale_patches(img)
        if not patches:
            skipped += 1
            continue
        
        tensors = [bgr_to_tensor(p) for p in patches]
        if len(tensors) > MAX_PATCHES_PER_IMAGE:
            idx = random.sample(range(len(tensors)), MAX_PATCHES_PER_IMAGE)
            tensors = [tensors[i] for i in sorted(idx)]
        
        pending_tensors.extend(tensors)
        pending_meta.append({
            'label': int(row.target),
            'label_name': row.label,
            'patient': int(row.patient),
            'path': str(row.path),
            'n_patches': len(tensors),
            'cache_path': cache_path,
        })
        
        if len(pending_tensors) >= PATCH_BATCH_TARGET:
            flush()
    
    flush()
    
    if skipped:
        print(f"  ⚠ Skipped {skipped}")
    if cache_hits:
        print(f"  ⚡ Cache hits: {cache_hits}/{len(df)}")
    print(f"  ✓ Extracted {len(results)} bags")
    return results


# ════════════════════════════════════════════════════════════
# 5: TINY CLAM MODEL
# ════════════════════════════════════════════════════════════
class TinyCLAM(nn.Module):
    """
    Minimal CLAM: single attention head, no adapter.
    Hidden dim configurable (384 for teacher, 256 for students).
    """
    def __init__(self, feat_dim_in, hidden=256, n_classes=NUM_CLASSES,
                 dropout=0.25, k_sample=CLAM_K_SAMPLE):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.feat_noise = FEAT_NOISE_STD
        self.feat_drop = nn.Dropout(FEAT_DROP_P)
        
        # Single linear encoder (no adapter)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Single gated attention head per class
        self.att_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 4),
                nn.Tanh(),
            ) for _ in range(n_classes)
        ])
        self.gate_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 4),
                nn.Sigmoid(),
            ) for _ in range(n_classes)
        ])
        self.att_combine = nn.ModuleList([
            nn.Linear(hidden // 4, 1) for _ in range(n_classes)
        ])
        
        # Per-class instance classifiers
        self.inst_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 64), nn.GELU(),
                nn.Linear(64, 2)
            ) for _ in range(n_classes)
        ])
        
        # Per-class bag classifiers
        self.bag_classifiers = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(n_classes)
        ])

    def _inst_loss(self, scores, h, classifier, k):
        N = scores.shape[0]
        k = min(k, N // 2, 8)
        if k < 1:
            return torch.tensor(0.0, device=h.device)
        top_idx = torch.topk(scores, k).indices
        bot_idx = torch.topk(scores, k, largest=False).indices
        feats = torch.cat([h[top_idx], h[bot_idx]], dim=0)
        labels = torch.cat([torch.ones(k, dtype=torch.long), torch.zeros(k, dtype=torch.long)]).to(h.device)
        return F.cross_entropy(classifier(feats), labels)

    def forward(self, x, label=None):
        input_dtype = x.dtype
        
        # Per-bag standardization
        if USE_PER_BAG_STD:
            x_f = x.float()
            x = ((x_f - x_f.mean(0, keepdim=True)) / (x_f.std(0, keepdim=True) + 1e-6)).to(input_dtype)
        
        if self.training:
            x = x.float()
            x = x + torch.randn_like(x) * self.feat_noise
            x = x.to(input_dtype)
            x = self.feat_drop(x)
        
        h = self.fc(x)  # (N, hidden)
        
        logits = []
        total_inst = torch.tensor(0.0, device=x.device)
        
        for c in range(self.n_classes):
            a = self.att_branches[c](h)
            g = self.gate_branches[c](h)
            scores = self.att_combine[c](a * g).squeeze(-1)
            weights = F.softmax(scores, dim=0)
            bag = torch.sum(weights.unsqueeze(-1) * h, dim=0)
            logits.append(self.bag_classifiers[c](bag))
            
            if self.training and label is not None and label.item() == c:
                total_inst += self._inst_loss(scores.detach(), h, self.inst_classifiers[c], self.k_sample)
        
        return {'logits': torch.cat(logits), 'inst_loss': total_inst}


# ════════════════════════════════════════════════════════════
# 6: LOSSES
# ════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def hierarchical_loss(logits, label):
    cancer_score = logits[0] + logits[1]
    normal_score = logits[2]
    binary_logit = torch.stack([cancer_score, normal_score]).unsqueeze(0)
    binary_target = torch.tensor([0] if label.item() in CANCER_CLASSES else [1],
                                 dtype=torch.long, device=label.device)
    return F.cross_entropy(binary_logit, binary_target)


def ordinal_loss(logits, label):
    probs = F.softmax(logits, dim=0)
    severity = torch.arange(NUM_CLASSES, dtype=torch.float32, device=logits.device)
    pred_sev = (probs * severity).sum()
    true_sev = severity[label]
    return (pred_sev - true_sev) ** 2


def supervised_loss(output, label, class_weights=None, focal_criterion=None):
    logits = output['logits'].unsqueeze(0)
    target = label.unsqueeze(0)
    bag_loss = focal_criterion(logits, target) if focal_criterion else \
        F.cross_entropy(logits, target, weight=class_weights, label_smoothing=LABEL_SMOOTH)
    hier = hierarchical_loss(output['logits'], label)
    ordi = ordinal_loss(output['logits'], label)
    return BAG_LOSS_W * bag_loss + HIER_LOSS_W * hier + ORDINAL_LOSS_W * ordi + INST_LOSS_W * output['inst_loss']





# ════════════════════════════════════════════════════════════
# 7: TRAINING UTILITIES
# ════════════════════════════════════════════════════════════
def compute_class_weights(data_list):
    counts = Counter(d['label'] for d in data_list)
    total = sum(counts.values())
    weights = []
    for c in range(NUM_CLASSES):
        w = total / (NUM_CLASSES * max(counts.get(c, 0), 1))
        if c == CLASS_TO_IDX['HGC']:
            w *= HGC_WEIGHT_BOOST
        weights.append(w)
    wt = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights (HGC ×{HGC_WEIGHT_BOOST}):")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls}: count={counts.get(i,0)} → w={weights[i]:.3f}")
    return wt


def class_balanced_sample(data_list):
    by_class = defaultdict(list)
    for d in data_list:
        by_class[d['label']].append(d)
    max_count = max(len(v) for v in by_class.values())
    balanced = []
    for cls, items in by_class.items():
        balanced.extend(items)
        if len(items) < max_count:
            balanced.extend(random.choices(items, k=max_count - len(items)))
    random.shuffle(balanced)
    return balanced


def mixup_features(data_list, alpha=MIXUP_ALPHA):
    """Feature-space mixup: create synthetic bags by mixing two bags of the same class."""
    if not USE_MIXUP or alpha <= 0:
        return data_list
    
    by_class = defaultdict(list)
    for d in data_list:
        by_class[d['label']].append(d)
    
    synthetic = []
    for cls, items in by_class.items():
        if len(items) < 2:
            continue
        # Create a few synthetic samples per class
        n_synth = max(1, len(items) // 5)
        for _ in range(n_synth):
            i, j = random.sample(range(len(items)), 2)
            lam = np.random.beta(alpha, alpha)
            bag_a = items[i]['features']
            if isinstance(bag_a, str):
                bag_a = torch.load(bag_a, map_location='cpu', weights_only=False)['features']
            bag_a = bag_a.float()
            
            bag_b = items[j]['features']
            if isinstance(bag_b, str):
                bag_b = torch.load(bag_b, map_location='cpu', weights_only=False)['features']
            bag_b = bag_b.float()
            # Truncate to shorter bag
            min_len = min(bag_a.shape[0], bag_b.shape[0])
            mixed = (lam * bag_a[:min_len] + (1 - lam) * bag_b[:min_len]).half()
            synthetic.append({
                'features': mixed,
                'label': cls,
                'label_name': items[i]['label_name'],
                'patient': -1,  # synthetic
                'path': 'synthetic_mixup',
                'n_patches': mixed.shape[0],
            })
    
    return data_list + synthetic


def warmup_cosine(optimizer, warmup, total):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 8: MODEL TRAINING
# ════════════════════════════════════════════════════════════
def train_model(train_data, val_data, class_weights, fold_idx=None):
    print("" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    set_seed(42)
    model = TinyCLAM(
        feat_dim_in=feat_dim,
        hidden=MIL_HIDDEN,
        dropout=MIL_DROPOUT,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")
    
    focal = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTH).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = warmup_cosine(optim, WARMUP_EPOCHS, EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_data = class_balanced_sample(mixup_features(train_data))
        
        running = 0.0
        correct = 0
        total = 0
        
        for sample in epoch_data:
            feats = sample['features']
            if isinstance(feats, str):
                feats = torch.load(feats, map_location='cpu', weights_only=False)['features']
            feats = feats.to(DEVICE)
            if feats.shape[0] > MAX_PATCHES_TRAIN:
                idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TRAIN]
                feats = feats[idx]
            label = torch.tensor(sample['label'], dtype=torch.long, device=DEVICE)
            
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                out = model(feats, label=label)
                loss = supervised_loss(out, label, class_weights, focal)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
            
            running += loss.item()
            correct += (out['logits'].argmax().item() == sample['label'])
            total += 1
        
        sched.step()
        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)
        
        # Val
        model.eval()
        v_loss = 0.0
        v_correct = 0
        with torch.no_grad():
            for sample in val_data:
                feats = sample['features']
                if isinstance(feats, str):
                    feats = torch.load(feats, map_location='cpu', weights_only=False)['features']
                feats = feats.to(DEVICE)
                if feats.shape[0] > MAX_PATCHES_TEST:
                    idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TEST]
                    feats = feats[idx]
                label = torch.tensor(sample['label'], dtype=torch.long, device=DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    out = model(feats)
                loss = F.cross_entropy(out['logits'].float().unsqueeze(0),
                                       label.unsqueeze(0), weight=class_weights)
                v_loss += loss.item()
                if out['logits'].argmax().item() == sample['label']:
                    v_correct += 1
        v_loss /= max(len(val_data), 1)
        v_acc = v_correct / max(len(val_data), 1)
        
        improved = ""
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            improved = " *"
        else:
            patience += 1
        
        if epoch < 3 or (epoch + 1) % 3 == 0 or improved or epoch == EPOCHS - 1:
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                  f"val_loss={v_loss:.4f} val_acc={v_acc:.3f}{improved}")
        
        if patience >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
        print(f"    ✓ Restored model (val_loss={best_val_loss:.4f})")
    
    model_name = f'model_fold{fold_idx}.pt' if fold_idx else 'model.pt'
    model_path = os.path.join(OUTPUT_DIR, model_name)
    safe_torch_save(model.state_dict(), model_path)
    print(f"    [OK] Model checkpoint saved: {model_path}")
    return model


# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# 10: IMAGE-SPACE TTA + INFERENCE (12 views)
# ════════════════════════════════════════════════════════════
def tta_transform(img_bgr, tta_idx):
    """Apply one of 12 TTA transformations to image."""
    if tta_idx == 0:
        return img_bgr  # identity
    elif tta_idx == 1:
        return cv2.flip(img_bgr, 1)  # h-flip
    elif tta_idx == 2:
        return cv2.flip(img_bgr, 0)  # v-flip
    elif tta_idx == 3:
        return cv2.flip(cv2.flip(img_bgr, 1), 0)  # h+v
    elif tta_idx == 4:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif tta_idx == 5:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif tta_idx == 6:
        # Slight brightness up
        return cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=10)
    elif tta_idx == 7:
        # Slight brightness down
        return cv2.convertScaleAbs(img_bgr, alpha=0.9, beta=-10)
    elif tta_idx == 8:
        # Gaussian blur (slight)
        return cv2.GaussianBlur(img_bgr, (3, 3), 0.5)
    elif tta_idx == 9:
        # Contrast boost
        return cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=0)
    elif tta_idx == 10:
        # Saturation boost
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif tta_idx == 11:
        # Rotate 180
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    return img_bgr


def extract_features_with_tta(df, norm, desc="TTA features"):
    """Extract features for all images × N_TTA_VIEWS, with caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        per_image_features = []
        
        try:
            img_orig = load_image(str(row.path), norm)
        except Exception:
            results.append(None)
            continue
        
        for tta_idx in range(N_TTA_VIEWS):
            cache_key = _get_cache_key(str(row.path), tta_idx=tta_idx)
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pt")
            
            if os.path.exists(cache_path):
                if os.path.exists(cache_path):
                    per_image_features.append(cache_path)
                    continue
            
            img_tta = tta_transform(img_orig, tta_idx)
            patches = extract_multiscale_patches(img_tta)
            if not patches:
                per_image_features.append(None)
                continue
            tensors = [bgr_to_tensor(p) for p in patches]
            if len(tensors) > MAX_PATCHES_PER_IMAGE:
                idx = random.sample(range(len(tensors)), MAX_PATCHES_PER_IMAGE)
                tensors = [tensors[i] for i in sorted(idx)]
            
            feats = extract_dual_features(tensors)
            try:
                torch.save({'features': feats}, cache_path)
            except Exception:
                pass
            per_image_features.append(cache_path if cache_path else feats)
        
        results.append({
            'tta_features': per_image_features,
            'label': int(row.target),
            'patient': int(row.patient),
            'path': str(row.path),
        })
    
    valid = sum(1 for r in results if r is not None)
    print(f"  ✓ Extracted TTA features for {valid} images × {N_TTA_VIEWS} views")
    return results


@torch.no_grad()
def predict_with_tta(model, tta_data_item):
    """For one image: run model × all TTA views, average softmax."""
    if tta_data_item is None:
        return np.array([1/3, 1/3, 1/3])
    
    all_probs = []
    for tta_feats in tta_data_item['tta_features']:
        if tta_feats is None:
            continue
        if isinstance(tta_feats, str):
            try:
                feats = torch.load(tta_feats, map_location='cpu', weights_only=False)['features']
            except Exception:
                continue
        else:
            if tta_feats.shape[0] == 0:
                continue
            feats = tta_feats
        feats = feats.to(DEVICE)
        if feats.shape[0] > MAX_PATCHES_TEST:
            idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TEST]
            feats = feats[idx]
        
        model.eval()
        with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
            out = model(feats)
        probs = F.softmax(out['logits'].float(), dim=0).cpu().numpy()
        all_probs.append(probs)
    
    if not all_probs:
        return np.array([1/3, 1/3, 1/3])
    avg_probs = np.mean(all_probs, axis=0)
    avg_probs = avg_probs / (avg_probs.sum() + 1e-8)
    return avg_probs


def evaluate_full(model, tta_data, desc="Evaluating"):
    all_preds = []
    all_labels = []
    all_probs = []
    for item in tqdm(tta_data, desc=desc):
        if item is None:
            continue
        probs = predict_with_tta(model, item)
        all_preds.append(int(np.argmax(probs)))
        all_labels.append(item['label'])
        all_probs.append(probs)
    return all_preds, all_labels, all_probs


# ════════════════════════════════════════════════════════════
# 11: CONSTRAINED THRESHOLD TUNING
# ════════════════════════════════════════════════════════════
def tune_constrained_thresholds(val_labels, val_probs, hgc_recall_target=0.92):
    """Grid search for thresholds that maximize balanced accuracy
    while keeping HGC recall >= target."""
    print("" + "=" * 60)
    print(f"  CONSTRAINED THRESHOLD TUNING (HGC recall ≥ {hgc_recall_target*100:.0f}%)")
    print("=" * 60)
    
    val_labels = np.array(val_labels)
    val_probs = np.array(val_probs)
    hgc_idx = CLASS_TO_IDX['HGC']
    lgc_idx = CLASS_TO_IDX['LGC']
    hgc_true = (val_labels == hgc_idx)
    n_hgc = hgc_true.sum()
    
    base_preds = np.argmax(val_probs, axis=1)
    base_bal = balanced_accuracy_score(val_labels, base_preds)
    base_hgc_r = ((base_preds == hgc_idx) & hgc_true).sum() / max(n_hgc, 1)
    print(f"  Argmax baseline: bal_acc={base_bal*100:.1f}%, HGC_recall={base_hgc_r*100:.1f}%")
    
    if n_hgc == 0:
        print("  ⚠ No HGC in val — skipping threshold tuning, using argmax")
        return None, None
    
    best = None
    best_bal = -1
    
    for hgc_t in np.arange(0.10, 0.55, 0.02):
        for lgc_t in np.arange(0.20, 0.55, 0.02):
            preds = []
            for p in val_probs:
                if p[hgc_idx] > hgc_t:
                    preds.append(hgc_idx)
                elif p[lgc_idx] > lgc_t:
                    preds.append(lgc_idx)
                else:
                    preds.append(int(np.argmax(p)))
            preds = np.array(preds)
            hgc_r = ((preds == hgc_idx) & hgc_true).sum() / max(n_hgc, 1)
            if hgc_r < hgc_recall_target:
                continue
            bal = balanced_accuracy_score(val_labels, preds)
            if bal > best_bal:
                best_bal = bal
                hgc_p = ((preds == hgc_idx) & hgc_true).sum() / max((preds == hgc_idx).sum(), 1)
                best = {
                    'hgc_threshold': float(hgc_t),
                    'lgc_threshold': float(lgc_t),
                    'bal_acc': bal,
                    'hgc_recall': hgc_r,
                    'hgc_precision': hgc_p,
                    'accuracy': accuracy_score(val_labels, preds),
                }
    
    if best:
        print(f"  ✓ Best: HGC_t={best['hgc_threshold']:.3f}, LGC_t={best['lgc_threshold']:.3f}")
        print(f"    val: bal_acc={best['bal_acc']*100:.1f}%, "
              f"HGC_R={best['hgc_recall']*100:.1f}%, HGC_P={best['hgc_precision']*100:.1f}%")
        return {hgc_idx: best['hgc_threshold'], lgc_idx: best['lgc_threshold']}, best
    
    print(f"  ⚠ Constraint not satisfiable at {hgc_recall_target*100:.0f}%.")
    if hgc_recall_target > 0.80:
        return tune_constrained_thresholds(val_labels, val_probs, hgc_recall_target=hgc_recall_target - 0.05)
    print("  → Falling back to argmax")
    return None, None


def apply_thresholds(probs_list, thresholds):
    """Apply tuned thresholds to probability list."""
    if thresholds is None:
        return [int(np.argmax(p)) for p in probs_list]
    
    preds = []
    hgc_idx = CLASS_TO_IDX['HGC']
    lgc_idx = CLASS_TO_IDX['LGC']
    hgc_t = thresholds.get(hgc_idx, 0.5)
    lgc_t = thresholds.get(lgc_idx, 0.5)
    for p in probs_list:
        if p[hgc_idx] > hgc_t:
            preds.append(hgc_idx)
        elif p[lgc_idx] > lgc_t:
            preds.append(lgc_idx)
        else:
            preds.append(int(np.argmax(p)))
    return preds


# ════════════════════════════════════════════════════════════
# 12: REPORTING
# ════════════════════════════════════════════════════════════
def print_results(preds, labels, name):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    bal = balanced_accuracy_score(labels, preds)
    print(f"{'=' * 60}{name}{'=' * 60}")
    print(f"  Accuracy:          {acc*100:.2f}%")
    print(f"  Balanced Accuracy: {bal*100:.2f}%")
    print(f"{classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0)}")
    hgc_idx = CLASS_TO_IDX['HGC']
    hgc_true = (labels == hgc_idx)
    hgc_pred = (preds == hgc_idx)
    hgc_tp = (hgc_pred & hgc_true).sum()
    hgc_r = hgc_tp / max(hgc_true.sum(), 1)
    hgc_p = hgc_tp / max(hgc_pred.sum(), 1)
    print(f"  HGC: Recall={hgc_r*100:.2f}%, Precision={hgc_p*100:.2f}%")
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print(f"Confusion Matrix:{'':<10} {'  '.join(f'{c:>8s}' for c in CLASS_NAMES)}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:>10} {'  '.join(f'{cm[i,j]:8d}' for j in range(NUM_CLASSES))}")
    target_acc = "✅" if acc >= 0.90 else "❌"
    target_hgc = "✅" if hgc_r >= 0.95 else "❌"
    print(f"Target acc≥90%:    {target_acc} ({acc*100:.1f}%)")
    print(f"  Target HGC R≥95%:  {target_hgc} ({hgc_r*100:.1f}%)")
    return {'accuracy': acc, 'balanced_accuracy': bal,
            'hgc_recall': hgc_r, 'hgc_precision': hgc_p}


def save_cm(preds, labels, fname, title):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


# ════════════════════════════════════════════════════════════
# FIXED FOLD DEFINITIONS — ALL CLASSES IN EVERY SPLIT
# ════════════════════════════════════════════════════════════
# Patient class mapping:
#   HGC:    2, 4, 6, 8, 10, 16, 17, 24
#   LGC:    5, 9, 11, 12, 13, 18, 23, 25
#   Normal: 0, 1, 7, 14, 21, 22
#
# Design principle: each fold's test, val, and train contain at least one
# HGC patient, one LGC patient, and one Normal patient.

FIXED_FOLDS = [
    # Fold 1: Test has HGC(8), LGC(13), Normal(21)
    {
        'test':  [8, 13, 21],
        'val':   [6, 9, 14],
        'train': [2, 4, 5, 7, 10, 11, 12, 16, 17, 18, 22, 23, 24, 25, 0, 1],
    },
    # Fold 2: Test has HGC(17), LGC(11), Normal(22)
    {
        'test':  [17, 11, 22],
        'val':   [2, 18, 0],
        'train': [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 21, 23, 24, 25, 1],
    },
    # Fold 3: Test has HGC(10), LGC(12), Normal(7)
    {
        'test':  [10, 12, 7],
        'val':   [4, 25, 1],
        'train': [2, 5, 6, 8, 9, 11, 13, 14, 16, 17, 18, 21, 22, 23, 24, 0],
    },
    # Fold 4: Test has HGC(24), LGC(23), Normal(14)
    {
        'test':  [24, 23, 14],
        'val':   [16, 5, 22],
        'train': [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 21, 25, 0, 1],
    },
    # Fold 5: Test has HGC(6), LGC(25), Normal(1)
    {
        'test':  [6, 25, 1],
        'val':   [17, 23, 21],
        'train': [2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 22, 24, 0],
    },
]


def resolve_manifest_path(row, orig_data_dir, aug_data_dir):
    """FIX #2: Resolve manifest full_path to actual file path on disk.
    
    The manifest has relative paths like:
      - Originals: "HGC/case_002_pt_003_frame_0009.png"
      - Augmented: "WLI2NBI/HGC/case_002_pt_003_frame_0009_WLI2NBI.png"
    
    We need to prepend the correct root directory.
    """
    full_path = str(row.get('full_path', ''))
    is_aug = row.get('is_augmented', False)
    
    if is_aug:
        # Augmented images are under aug_data_dir
        candidate = os.path.join(aug_data_dir, full_path)
        if os.path.exists(candidate):
            return candidate
    else:
        # Original images are under orig_data_dir (EndoscopicBladderTissue/)
        candidate = os.path.join(orig_data_dir, full_path)
        if os.path.exists(candidate):
            return candidate
    
    # Fallback: look up by filename in global index
    filename = os.path.basename(full_path).lower()
    if filename in IMAGE_PATH_INDEX:
        return IMAGE_PATH_INDEX[filename]
    
    return None


def load_data_for_fold_v2(fold_idx, fold_spec):
    """Load data for a fixed fold spec. No WLI filter on val."""
    scan_for_images()

    train_pids = fold_spec['train']
    val_pids   = fold_spec['val']
    test_pids  = fold_spec['test']

    print("\n" + "═"*60)
    print(f"  FOLD {fold_idx}/5  (fixed patient splits)")
    print("═"*60)
    print(f"  Train patients: {sorted(train_pids)}")
    print(f"  Val   patients: {sorted(val_pids)}")
    print(f"  Test  patients: {sorted(test_pids)}")

    # ── Augmented train data ──
    if not os.path.exists(AUG_TRAIN_MANIFEST):
        raise FileNotFoundError(f"Manifest not found: {AUG_TRAIN_MANIFEST}. "
                                f"Check the path configuration.")
    
    full_manifest = pd.read_csv(AUG_TRAIN_MANIFEST)
    full_manifest['label']   = full_manifest['tissue type'].map(LABEL_MAP)
    full_manifest['target']  = full_manifest['label'].map(CLASS_TO_IDX)
    full_manifest['patient'] = full_manifest['patient_id']
    
    # FIX #2: Resolve paths properly
    full_manifest['path'] = full_manifest.apply(
        lambda r: resolve_manifest_path(r, ORIG_DATA_DIR, AUG_TRAIN_DIR), axis=1)
    
    # Drop rows with unresolvable paths
    n_before = len(full_manifest)
    full_manifest = full_manifest[full_manifest['path'].notna()].copy()
    n_after = len(full_manifest)
    if n_before != n_after:
        print(f"  ⚠ Dropped {n_before - n_after} images with unresolvable paths from manifest")
    
    # Drop NaN targets (unmappable tissue types)
    full_manifest = full_manifest[full_manifest['target'].notna()].copy()
    full_manifest['target'] = full_manifest['target'].astype(int)
    
    train_df = full_manifest[full_manifest['patient_id'].isin(train_pids)].copy()

    # ── Original (non-augmented) for val and test ──
    df_orig = pd.read_csv(ANNOTATIONS_CSV)
    df_orig.columns = df_orig.columns.str.strip()
    df_orig['patient_id'] = df_orig['HLY'].apply(extract_patient_id)
    df_orig['label']  = df_orig['tissue type'].map(LABEL_MAP)
    df_orig['target'] = df_orig['label'].map(CLASS_TO_IDX)
    df_orig['patient'] = df_orig['patient_id']
    df_orig['path']   = df_orig.apply(
        lambda r: IMAGE_PATH_INDEX.get(str(r['HLY']).strip().lower()), axis=1)
    df_orig = df_orig[df_orig['path'].notna()].copy()

    val_df  = df_orig[df_orig['patient_id'].isin(val_pids)].copy()
    test_df = df_orig[df_orig['patient_id'].isin(test_pids)].copy()

    # ── Disjointness check ──
    assert not set(train_pids) & set(val_pids),  "Train/Val overlap!"
    assert not set(train_pids) & set(test_pids), "Train/Test overlap!"
    assert not set(val_pids)   & set(test_pids), "Val/Test overlap!"
    print("  ✓ Patient disjointness verified")

    # ── Class distribution check ──
    for split_name, df_check in [('Val', val_df), ('Test', test_df)]:
        classes_present = set(df_check['label'].unique())
        missing = set(CLASS_NAMES) - classes_present
        if missing:
            # FIX #5: Warn instead of skipping
            print(f"  ⚠ WARNING: {split_name} missing classes: {missing}")

    for name, df in [('Train (aug)', train_df), ('Val', val_df), ('Test', test_df)]:
        dist = df['label'].value_counts().to_dict()
        print(f"  {name}: {len(df)} images")
        for cls in CLASS_NAMES:
            cnt = dist.get(cls, 0)
            pct = 100 * cnt / max(len(df), 1)
            print(f"    {cls}: {cnt} ({pct:.1f}%)")

    return train_df, val_df, test_df


def main():
    import datetime
    import gc

    start = time.time()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    exp_tag = 'frozen' if FREEZE_BACKBONE else 'partial_unfreeze'

    BASE_OUTPUT = OUTPUT_DIR
    os.makedirs(BASE_OUTPUT, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Bladder Classification — v4 5-Fold CV CONVNEXT               ║")
    print("║  Balanced patient splits (all classes in every split)      ║")
    print("║  Normal Model Training (No distillation)                   ║")
    print("║  12-view image-TTA + threshold tuning at inference         ║")
    print("║  Feature mixup                                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Device: {DEVICE} | AMP: {USE_AMP}")
    if SKIP_DINO_ON_CPU:
        print("⚠ CPU mode: DINOv2 disabled, using DenseNet121 only")

    fold_results = []
    all_preds, all_labels, all_pids = [], [], []

    for fold_idx, fold_spec in enumerate(FIXED_FOLDS, start=1):
        fold_dir = os.path.join(BASE_OUTPUT, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)

        global dino_model, dense_model, feat_dim
        feat_dim = load_backbones()

        train_df, val_df, test_df = load_data_for_fold_v2(fold_idx, fold_spec)

        norm = fit_normalizer(train_df.sample(min(len(train_df), 200), random_state=42))

        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx} — EXTRACTING FEATURES")
        print("=" * 60)

        train_data = extract_features_for_split(train_df, f"Fold{fold_idx} Train", norm, use_cache=True)
        val_data   = extract_features_for_split(val_df,   f"Fold{fold_idx} Val",   norm, use_cache=True)

        print(f"  Train bags: {len(train_data)} | Val bags: {len(val_data)}")

        # FIX #5: Don't skip folds with 0 HGC in val — just warn
        hgc_in_val = sum(1 for d in val_data if d['label'] == CLASS_TO_IDX['HGC'])
        if hgc_in_val == 0:
            print(f"  ⚠ WARNING: Val has 0 HGC bags (continuing anyway)")

        class_weights = compute_class_weights(train_data)

        # ── Train Model ──
        model = train_model(train_data, val_data, class_weights, fold_idx=fold_idx)
        model.eval()

        # ── TTA Evaluation ──
        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx} — TTA INFERENCE")
        print("=" * 60)
        val_tta  = extract_features_with_tta(val_df,  norm, f"Fold{fold_idx} Val TTA")
        test_tta = extract_features_with_tta(test_df, norm, f"Fold{fold_idx} Test TTA")

        # Get raw probs
        val_preds_raw, val_labels_f, val_probs_f = evaluate_full(
            model, val_tta, f"Fold{fold_idx} Val")
        test_preds_raw, test_labels_f, test_probs_f = evaluate_full(
            model, test_tta, f"Fold{fold_idx} Test")

        # ── Threshold tuning on val ──
        thresholds, tune_info = tune_constrained_thresholds(val_labels_f, val_probs_f)

        # Apply thresholds (or argmax if tuning failed)
        val_preds_f = apply_thresholds(val_probs_f, thresholds)
        test_preds_f = apply_thresholds(test_probs_f, thresholds)

        val_metrics  = print_results(val_preds_f,  val_labels_f,
                                     f"FOLD {fold_idx} VAL")
        test_metrics = print_results(test_preds_f, test_labels_f,
                                     f"FOLD {fold_idx} TEST")

        # ── Per-patient test breakdown ──
        test_patients_list = [item['patient'] for item in test_tta if item is not None]
        print(f"\n  Per-patient breakdown (Fold {fold_idx} Test):")
        for pid in sorted(set(test_patients_list)):
            idxs  = [i for i, p in enumerate(test_patients_list) if p == pid]
            p_lab = [test_labels_f[i] for i in idxs]
            p_prd = [test_preds_f[i]  for i in idxs]
            p_acc = accuracy_score(p_lab, p_prd)
            dist  = Counter(p_lab)
            dist_str = ', '.join(f"{IDX_TO_CLASS[k]}={v}"
                                 for k, v in sorted(dist.items()))
            print(f"    Patient {pid}: {len(idxs)} imgs ({dist_str}) → "
                  f"acc={p_acc*100:.1f}%")

        # ── Save artifacts ──
        save_cm(test_preds_f, test_labels_f,
                os.path.join(fold_dir, f'cm_test_fold{fold_idx}.png'),
                f"Fold {fold_idx} Test — {exp_tag}")
        save_cm(val_preds_f,  val_labels_f,
                os.path.join(fold_dir, f'cm_val_fold{fold_idx}.png'),
                f"Fold {fold_idx} Val — {exp_tag}")

        safe_torch_save(model.state_dict(),
                        os.path.join(fold_dir, f'model_fold{fold_idx}.pt'))

        # Save threshold info
        if tune_info:
            with open(os.path.join(fold_dir, f'thresholds_fold{fold_idx}.json'), 'w') as f:
                json.dump(tune_info, f, indent=2, default=str)

        fold_results.append({
            'fold': fold_idx,
            'train_pids': fold_spec['train'],
            'val_pids':   fold_spec['val'],
            'test_pids':  fold_spec['test'],
            'val_metrics':  val_metrics,
            'test_metrics': test_metrics,
            'thresholds': tune_info,
        })
        all_preds.extend(test_preds_f)
        all_labels.extend(test_labels_f)
        all_pids.extend(test_patients_list)

        # ── Free GPU memory ──
        del convnext_model, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════
    # AGGREGATE RESULTS
    # ════════════════════════════════════════════════════════
    if not fold_results:
        print("No folds completed. Check FIXED_FOLDS patient assignments.")
        return

    print("\n" + "═"*60)
    print("  5-FOLD CV — AGGREGATE RESULTS")
    print("═"*60)

    accs  = [r['test_metrics']['accuracy']          for r in fold_results]
    bals  = [r['test_metrics']['balanced_accuracy'] for r in fold_results]
    hgcrs = [r['test_metrics']['hgc_recall']        for r in fold_results]
    hgcps = [r['test_metrics']['hgc_precision']     for r in fold_results]

    header = f"\n  {'Fold':<8} {'Acc':>8} {'BalAcc':>8} {'HGC-R':>8} {'HGC-P':>8}  Test patients"
    print(header)
    print(f"  {'-'*70}")
    for r in fold_results:
        m = r['test_metrics']
        print(f"  Fold {r['fold']:<3}  {m['accuracy']*100:>7.1f}%  "
              f"{m['balanced_accuracy']*100:>7.1f}%  "
              f"{m['hgc_recall']*100:>7.1f}%  "
              f"{m['hgc_precision']*100:>7.1f}%  "
              f"{r['test_pids']}")
    print(f"  {'-'*70}")
    print(f"  Mean     {np.mean(accs)*100:>7.1f}%  "
          f"{np.mean(bals)*100:>7.1f}%  "
          f"{np.mean(hgcrs)*100:>7.1f}%  "
          f"{np.mean(hgcps)*100:>7.1f}%")
    print(f"  ± Std    {np.std(accs)*100:>7.1f}%  "
          f"{np.std(bals)*100:>7.1f}%  "
          f"{np.std(hgcrs)*100:>7.1f}%  "
          f"{np.std(hgcps)*100:>7.1f}%")

    # Grand confusion matrix
    save_cm(all_preds, all_labels,
            os.path.join(BASE_OUTPUT, f'cm_grand_{exp_tag}.png'),
            f"Grand CV — {exp_tag}")
    grand_metrics = print_results(all_preds, all_labels, "GRAND CV TEST")

    # Per-patient grand summary
    print("\n  Per-patient grand test accuracy:")
    for pid in sorted(set(all_pids)):
        idxs  = [i for i, p in enumerate(all_pids) if p == pid]
        p_lab = [all_labels[i] for i in idxs]
        p_prd = [all_preds[i]  for i in idxs]
        p_acc = accuracy_score(p_lab, p_prd)
        dist  = Counter(p_lab)
        dist_str = ', '.join(f"{IDX_TO_CLASS[k]}={v}"
                             for k, v in sorted(dist.items()))
        print(f"    Patient {pid}: ({dist_str}) → acc={p_acc*100:.1f}%")

    # Save JSON
    elapsed = time.time() - start
    results_out = {
        'experiment': f'v3_5foldcv_fixed_{exp_tag}',
        'timestamp':  timestamp,
        'runtime_minutes': elapsed / 60,
        'backbone': exp_tag,
        'fold_splitter': 'fixed_balanced_patient_splits',
        'n_folds': len(fold_results),
        'fold_results': fold_results,
        'grand_metrics': grand_metrics,
        'summary': {
            'acc_mean':   float(np.mean(accs)),
            'acc_std':    float(np.std(accs)),
            'bal_mean':   float(np.mean(bals)),
            'bal_std':    float(np.std(bals)),
            'hgcr_mean':  float(np.mean(hgcrs)),
            'hgcr_std':   float(np.std(hgcrs)),
        }
    }
    results_path = os.path.join(BASE_OUTPUT,
                                f'cv_results_{exp_tag}_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results_out, f, indent=2, default=str)

    print("\n" + "═"*60)
    print(f"  COMPLETE — Runtime: {elapsed/60:.1f} min")
    print(f"  Acc:   {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
    print(f"  Bal:   {np.mean(bals)*100:.1f}% ± {np.std(bals)*100:.1f}%")
    print(f"  HGC-R: {np.mean(hgcrs)*100:.1f}% ± {np.std(hgcrs)*100:.1f}%")
    print(f"  Results → {results_path}")
    print("═"*60)


if __name__ == '__main__':
    main()
