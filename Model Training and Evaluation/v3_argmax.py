"""
===========================================================================
v3-FINAL ARGMAX - Complete Single-Notebook Pipeline
===========================================================================
What this does (in order):
  1. Generate stylized 8x augmented training data
  2. Extract DINOv2 + DenseNet121 features (frozen backbones, cached)
  3. Train teacher (TinyCLAM, hidden=256, single-head per-class gated attn)
  4. Train 5 students with KL-distillation on val
  5. Evaluate with 8-view image-space TTA + ensemble (ARGMAX ONLY, no threshold tuning)
  6. Save ALL checkpoints (teacher + 5 students) to survive Kaggle wipe
  7. Modality-stratified reporting (WLI / NBI / Overall)
  8. Per-patient diagnostic breakdown
  9. UMAP + t-SNE bag-level feature visualizations (paper figures)
 10. Zip everything for one-click download

Removed from original v3-final.py:
  [x] GAN augmentation (collapsed LGC recall)
  [x] Val WLI filter (created distribution mismatch)
  [x] Constrained threshold tuning (broke on P2 OOD val HGC)

Splits (Split B - validated 87.65% bal acc / 98.65% HGC recall):
  Train: [4,5,7,8,10,13,14,16,21,22,23,24,25]
  Val:   [0,1,2,9,12]
  Test:  [6,11,17,18]
===========================================================================
"""

# ----------------------------------------------------------
# Cell 0: Install dependencies
# ----------------------------------------------------------
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn"], check=False)

# ----------------------------------------------------------
# Cell 1: Imports
# ----------------------------------------------------------
import os, re, copy, math, json, time, random, hashlib, warnings, shutil, zipfile
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

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, recall_score, precision_score, f1_score
)
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
IS_KAGGLE = os.path.exists('/kaggle')
IS_LIGHTNING = os.path.exists('/teamspace')
workspace = Path('.').resolve()


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

if IS_KAGGLE:
    AUG_TRAIN_DIR = '/kaggle/working/v3_augmented'
    AUG_TRAIN_MANIFEST = '/kaggle/working/v3_augmented_manifest.csv'
    DATA_SEARCH_ROOTS = ['/kaggle/input']
    ORIG_DATA_DIR = find_first_dir(DATA_SEARCH_ROOTS, 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations_fixed.csv')
    OUTPUT_DIR = '/kaggle/working/output_v3_argmax'
    CACHE_DIR = '/kaggle/working/feat_cache_v3'
    DIAG_DIR = '/kaggle/working/diagnostics'
    BUNDLE_PATH = '/kaggle/working/v3_argmax_bundle.zip'
elif IS_LIGHTNING:
    LIGHTNING_WORK_DIR = Path('/teamspace/studios/this_studio')
    if not LIGHTNING_WORK_DIR.exists():
        LIGHTNING_WORK_DIR = workspace
    DATA_SEARCH_ROOTS = [
        str(workspace),
        '/teamspace/datasets',
        '/teamspace/studios/this_studio',
    ]
    ORIG_DATA_DIR = find_first_dir(DATA_SEARCH_ROOTS, 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations_fixed.csv')
    if ANNOTATIONS_CSV is None:
        ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations.csv')
    AUG_TRAIN_DIR = str(LIGHTNING_WORK_DIR / 'v3_augmented')
    AUG_TRAIN_MANIFEST = str(LIGHTNING_WORK_DIR / 'v3_augmented_manifest.csv')
    OUTPUT_DIR = str(LIGHTNING_WORK_DIR / 'output_v3_argmax')
    CACHE_DIR = str(LIGHTNING_WORK_DIR / 'feat_cache_v3')
    DIAG_DIR = str(LIGHTNING_WORK_DIR / 'diagnostics')
    BUNDLE_PATH = str(LIGHTNING_WORK_DIR / 'v3_argmax_bundle.zip')
else:
    DATA_SEARCH_ROOTS = [str(workspace)]
    ORIG_DATA_DIR = find_first_dir(DATA_SEARCH_ROOTS, 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations_fixed.csv')
    if ANNOTATIONS_CSV is None:
        ANNOTATIONS_CSV = find_first_file(DATA_SEARCH_ROOTS, 'annotations.csv')
    AUG_TRAIN_DIR = str(workspace / 'v3_augmented')
    AUG_TRAIN_MANIFEST = str(workspace / 'v3_augmented_manifest.csv')
    OUTPUT_DIR = str(workspace / 'output_v3_argmax')
    CACHE_DIR = str(workspace / 'feat_cache_v3')
    DIAG_DIR = str(workspace / 'diagnostics')
    BUNDLE_PATH = str(workspace / 'v3_argmax_bundle.zip')

if ORIG_DATA_DIR is None:
    raise FileNotFoundError(f"Could not find EndoscopicBladderTissue under: {DATA_SEARCH_ROOTS}")
if ANNOTATIONS_CSV is None:
    raise FileNotFoundError(f"Could not find annotations CSV under: {DATA_SEARCH_ROOTS}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)


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


def remove_bad_cache(path):
    try:
        os.remove(path)
    except OSError:
        pass


# Splits (Split B)
TRAIN_PATIENTS = [4, 5, 7, 8, 10, 13, 14, 16, 21, 22, 23, 24, 25]
VAL_PATIENTS   = [0, 1, 2, 9, 12]
TEST_PATIENTS  = [6, 11, 17, 18]

# Classes
NUM_CLASSES = 3
CLASS_NAMES = ['HGC', 'LGC', 'Normal']
LABEL_MAP = {'HGC': 'HGC', 'LGC': 'LGC', 'NST': 'Normal', 'NTL': 'Normal'}
CLASS_TO_IDX = {'HGC': 0, 'LGC': 1, 'Normal': 2}
IDX_TO_CLASS = {0: 'HGC', 1: 'LGC', 2: 'Normal'}
CANCER_CLASSES = {0, 1}

# Image preprocessing (unchanged from v3-final)
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

FEAT_BATCH = 128
PATCH_BATCH_TARGET = 512
CACHE_VERSION = 'v3_argmax'

# CLAM
MIL_HIDDEN = 256
MIL_DROPOUT_TEACHER = 0.20
MIL_DROPOUT_STUDENTS = [0.30, 0.35, 0.40, 0.45, 0.50]
N_ATT_HEADS = 1
CLAM_K_SAMPLE = 8
FEAT_NOISE_STD = 0.015
FEAT_DROP_P = 0.10

# Training
LR = 1e-4
WD = 1e-4
TEACHER_EPOCHS = 30
STUDENT_EPOCHS = 25
WARMUP_EPOCHS = 3
PATIENCE = 8
GRAD_CLIP = 1.0

# Loss
FOCAL_GAMMA = 2.0
LABEL_SMOOTH = 0.05
BAG_LOSS_W = 1.0
INST_LOSS_W = 0.05
HIER_LOSS_W = 0.10
ORDINAL_LOSS_W = 0.10
DISTILL_W = 0.30
DISTILL_T = 4.0

HGC_WEIGHT_BOOST = 2.5
MAX_PATCHES_TRAIN = 200
MAX_PATCHES_TEST = 400

N_STUDENTS = 5
STUDENT_SEEDS = [42, 7, 123, 2024, 999]
N_TTA_VIEWS = 8
USE_PER_BAG_STD = True

# Augmentation
N_AUG_VARIANTS = 8
AUG_SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

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


def extract_patient_id(filename):
    s = str(filename)
    s = re.sub(r'_aug\d+\.png$', '', s)
    s = re.sub(r'_orig\.png$', '', s)
    for pat in [r'case_(\d+)', r'cys_case_(\d+)']:
        m = re.search(pat, s)
        if m:
            return int(m.group(1))
    return -1


# ============================================================
# STAGE 1: STYLIZED AUGMENTATION (inline)
# ============================================================
def aug_random_hue_sat_shift(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = random.uniform(-25, 25)
    s_mult = random.uniform(0.55, 1.45)
    v_mult = random.uniform(0.65, 1.35)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_mult, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_mult, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_random_gamma(img):
    gamma = random.uniform(0.65, 1.45)
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def aug_brightness_contrast(img):
    alpha = random.uniform(0.75, 1.30)
    beta = random.uniform(-30, 30)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def aug_motion_blur(img):
    if random.random() > 0.4:
        return img
    ksize = random.choice([3, 5, 7])
    angle = random.uniform(0, 180)
    kernel = np.zeros((ksize, ksize))
    kernel[ksize // 2, :] = np.ones(ksize) / ksize
    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel = kernel / (kernel.sum() + 1e-8)
    return cv2.filter2D(img, -1, kernel)


def aug_jpeg_compression(img):
    if random.random() > 0.5:
        return img
    quality = random.randint(35, 80)
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, enc_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def aug_vignette(img):
    if random.random() > 0.35:
        return img
    h, w = img.shape[:2]
    cx = w / 2 + random.uniform(-w * 0.08, w * 0.08)
    cy = h / 2 + random.uniform(-h * 0.08, h * 0.08)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    strength = random.uniform(0.20, 0.55)
    mask = np.clip(1 - strength * (dist / max_dist) ** 2, 0, 1)
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= mask
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_geometric(img):
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        img = cv2.flip(img, 0)
    if random.random() > 0.5:
        angle = random.uniform(-12, 12)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img


def stylized_augment(img):
    img = aug_geometric(img)
    img = aug_random_hue_sat_shift(img)
    img = aug_random_gamma(img)
    img = aug_brightness_contrast(img)
    img = aug_motion_blur(img)
    img = aug_vignette(img)
    img = aug_jpeg_compression(img)
    return img


def run_stage1_augmentation():
    print("\n" + "=" * 70)
    print("STAGE 1: IN-MEMORY AUG MANIFEST (no disk writes)")
    print("=" * 70)

    if os.path.exists(AUG_TRAIN_MANIFEST):
        existing = pd.read_csv(AUG_TRAIN_MANIFEST)
        if len(existing) > 0:
            if {'source_path', 'aug_seed'}.issubset(existing.columns):
                print(f"  [CACHE] Manifest already exists ({len(existing)} rows). Skipping.")
                return
            print("  [WARN] Existing manifest is an on-disk PNG manifest; rebuilding low-disk manifest.")

    random.seed(AUG_SEED)
    np.random.seed(AUG_SEED)

    df = pd.read_csv(ANNOTATIONS_CSV)
    df.columns = df.columns.str.strip()
    df['patient_id'] = df['HLY'].apply(extract_patient_id)
    train_df = df[df['patient_id'].isin(TRAIN_PATIENTS)].copy()
    print(f"  Train images to augment: {len(train_df)}")

    image_index = {}
    for search_root in DATA_SEARCH_ROOTS:
        if not os.path.exists(search_root):
            continue
        for root, _, files in os.walk(search_root):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_index[f.lower()] = os.path.join(root, f)
    print(f"  Indexed {len(image_index)} source images")

    manifest_rows = []
    skipped = 0

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building aug manifest"):
        fname = str(row['HLY']).strip()
        path = image_index.get(fname.lower())
        if path is None:
            skipped += 1
            continue
        base_name = os.path.splitext(fname)[0]
        tissue = str(row.get('tissue type', 'unknown'))

        # Original: aug_seed=-1 means no augmentation.
        manifest_rows.append({
            'HLY': f"{base_name}_orig",
            'tissue type': tissue,
            'patient_id': row['patient_id'],
            'is_augmented': False,
            'aug_seed': -1,
            'source_path': path,
        })

        # Variants store only the deterministic recipe; images are generated later.
        for v in range(N_AUG_VARIANTS):
            seed_key = f"{base_name}|{v}|{AUG_SEED}"
            aug_seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
            manifest_rows.append({
                'HLY': f"{base_name}_aug{v}",
                'tissue type': tissue,
                'patient_id': row['patient_id'],
                'is_augmented': True,
                'aug_seed': aug_seed,
                'source_path': path,
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(AUG_TRAIN_MANIFEST, index=False)
    print(f"  [OK] Manifest: {len(manifest)} entries ({skipped} sources skipped)")
    print(f"  [OK] Manifest: {AUG_TRAIN_MANIFEST}")
    print(f"  [OK] Disk saved: ~{len(manifest) * 50 / 1024:.1f} MB (no PNGs written)")
    for tt, cnt in manifest['tissue type'].value_counts().items():
        print(f"    {tt}: {cnt}")


def load_or_generate_image(row, norm=None):
    """Read source image and optionally apply deterministic in-memory augmentation."""
    img = cv2.imread(str(row['source_path']))
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > 768:
        s = 768 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    if bool(row.get('is_augmented', False)):
        seed = int(row['aug_seed'])
        rng_random = random.Random(seed)
        rng_np = np.random.RandomState(seed)
        old_random = random.getstate()
        old_np = np.random.get_state()
        random.setstate(rng_random.getstate())
        np.random.set_state(rng_np.get_state())
        try:
            img = stylized_augment(img)
        finally:
            random.setstate(old_random)
            np.random.set_state(old_np)

    h, w = img.shape[:2]
    s = IMAGE_RESIZE / max(h, w)
    if s != 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if norm:
        img = norm.transform(img)
    return img


# ============================================================
# STAGE 2: DATA LOADING
# ============================================================
IMAGE_PATH_INDEX = {}


def scan_for_images():
    global IMAGE_PATH_INDEX
    print("\n" + "=" * 70)
    print("SCANNING FILESYSTEM")
    print("=" * 70)
    search_dirs = DATA_SEARCH_ROOTS
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


def load_data():
    scan_for_images()
    print("\n" + "=" * 70)
    print("LOADING DATA - v3 ARGMAX")
    print("=" * 70)
    print(f"  Train patients: {TRAIN_PATIENTS}")
    print(f"  Val patients:   {VAL_PATIENTS}")
    print(f"  Test patients:  {TEST_PATIENTS}")

    if not os.path.exists(AUG_TRAIN_MANIFEST):
        raise FileNotFoundError(f"Augmented manifest not found: {AUG_TRAIN_MANIFEST}")
    train_df = pd.read_csv(AUG_TRAIN_MANIFEST)
    train_df['label'] = train_df['tissue type'].map(LABEL_MAP)
    train_df['target'] = train_df['label'].map(CLASS_TO_IDX)
    train_df['patient'] = train_df['patient_id']
    if 'source_path' in train_df.columns:
        train_df['path'] = train_df['source_path']
        print(f"  Loaded {len(train_df)} train recipes from in-memory augmentation manifest")
    else:
        train_df['path'] = train_df['full_path']
        print(f"  Loaded {len(train_df)} train images from augmented manifest")

    df_orig = pd.read_csv(ANNOTATIONS_CSV)
    df_orig.columns = df_orig.columns.str.strip()
    df_orig['patient_id'] = df_orig['HLY'].apply(extract_patient_id)
    df_orig['label'] = df_orig['tissue type'].map(LABEL_MAP)
    df_orig['target'] = df_orig['label'].map(CLASS_TO_IDX)
    df_orig['patient'] = df_orig['patient_id']
    df_orig['is_augmented'] = False
    df_orig['aug_mode'] = 'orig'

    def resolve_orig(row):
        fname = str(row['HLY']).strip()
        return IMAGE_PATH_INDEX.get(fname.lower())

    df_orig['path'] = df_orig.apply(resolve_orig, axis=1)
    df_orig = df_orig[df_orig['path'].notna()].copy()

    val_df = df_orig[df_orig['patient_id'].isin(VAL_PATIENTS)].copy()
    test_df = df_orig[df_orig['patient_id'].isin(TEST_PATIENTS)].copy()

    train_pids = set(train_df['patient_id'].unique())
    val_pids = set(val_df['patient_id'].unique())
    test_pids = set(test_df['patient_id'].unique())
    assert len(train_pids & val_pids) == 0
    assert len(train_pids & test_pids) == 0
    assert len(val_pids & test_pids) == 0
    print("  [OK] Patient disjointness verified")

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        dist = df['label'].value_counts().to_dict()
        print(f"{name}: {len(df)} images")
        for cls in CLASS_NAMES:
            cnt = dist.get(cls, 0)
            pct = 100 * cnt / max(len(df), 1)
            print(f"    {cls}: {cnt} ({pct:.1f}%)")

    # Save raw annotations CSV reference for modality lookup later
    return train_df, val_df, test_df, df_orig


# ============================================================
# STAGE 3: IMAGE PREPROCESSING + PATCH EXTRACTION
# ============================================================
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
    sample_paths = df.sample(min(50, len(df))).path.values
    for fp in sample_paths:
        try:
            img = cv2.imread(fp)
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
    print(f"  [OK] Normalizer fitted on {len(samples)} images")
    return norm


# ============================================================
# STAGE 4: FEATURE EXTRACTION
# ============================================================
dino_model = None
dense_model = None
feat_dim = 0


def load_backbones():
    global dino_model, dense_model, feat_dim, IMNET_MEAN, IMNET_STD
    print("\n" + "=" * 70)
    print("LOADING FROZEN BACKBONES")
    print("=" * 70)
    IMNET_MEAN = IMNET_MEAN.to(DEVICE)
    IMNET_STD = IMNET_STD.to(DEVICE)
    dino_dim = 0
    print("  Loading DINOv2...")
    try:
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dino_model.eval().to(DEVICE)
        for p in dino_model.parameters():
            p.requires_grad = False
        dino_dim = 768
        print(f"  [OK] dinov2_vitb14 - FROZEN, dim={dino_dim}")
    except Exception as e:
        print(f"  [WARN] DINOv2 failed: {e}")
    densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    dense_dim = densenet.classifier.in_features
    densenet.classifier = nn.Identity()
    dense_model = densenet.eval().to(DEVICE)
    for p in dense_model.parameters():
        p.requires_grad = False
    print(f"  [OK] DenseNet121 - FROZEN, dim={dense_dim}")
    feat_dim = (dino_dim if dino_model else 0) + dense_dim
    print(f"  Total feature dim: {feat_dim}")
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
        parts = []
        with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
            if dino_model is not None:
                dino_out = dino_model(batch_norm)
                if isinstance(dino_out, dict):
                    dino_feats = dino_out.get('x_norm_clstoken', next(iter(dino_out.values())))
                else:
                    dino_feats = dino_out
                if dino_feats.dim() > 2:
                    dino_feats = dino_feats[:, 0, :]
                parts.append(dino_feats.float().cpu())
            parts.append(dense_model(batch_norm).float().cpu())
        all_feats.append(torch.cat(parts, dim=1))
    return torch.cat(all_feats, 0).half()


def extract_features_for_split(df, desc="Extracting", norm=None, use_cache=True,
                               from_manifest=False):
    """
    If from_manifest=True, df is the augmentation manifest with source_path and
    aug_seed columns; images are generated in memory.
    Otherwise df has path column pointing to a real image file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    results, skipped, cache_hits = [], 0, 0
    pending_tensors, pending_meta = [], []
    cache_writes_enabled = use_cache

    def flush():
        nonlocal pending_tensors, pending_meta, cache_writes_enabled
        if not pending_tensors:
            return
        feats = extract_dual_features(pending_tensors)
        start = 0
        for meta in pending_meta:
            n = meta['n_patches']
            block = feats[start:start + n].clone()
            start += n
            if meta['cache_path'] and cache_writes_enabled:
                try:
                    safe_torch_save({'features': block}, meta['cache_path'])
                except Exception as e:
                    if 'No space' in str(e):
                        print("  [WARN] Disk full - disabling cache writes for remainder")
                        cache_writes_enabled = False
            results.append({
                'features': block,
                'label': meta['label'],
                'label_name': meta['label_name'],
                'patient': meta['patient'],
                'path': meta['path'],
                'filename': meta['filename'],
                'n_patches': block.shape[0],
            })
        pending_tensors.clear()
        pending_meta.clear()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        if from_manifest:
            cache_id = f"{row['source_path']}|{row.get('aug_seed', -1)}"
        else:
            cache_id = str(row.path)
        cache_path = None
        if use_cache:
            cache_key = _get_cache_key(cache_id)
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pt")
            if os.path.exists(cache_path):
                try:
                    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
                    if 'features' in cached and cached['features'].shape[0] > 0:
                        results.append({
                            'features': cached['features'],
                            'label': int(row.target),
                            'label_name': row.label,
                            'patient': int(row.patient),
                            'path': cache_id,
                            'filename': str(row.HLY),
                            'n_patches': cached['features'].shape[0],
                        })
                        cache_hits += 1
                        continue
                except Exception:
                    remove_bad_cache(cache_path)
        try:
            if from_manifest:
                img = load_or_generate_image(row, norm)
                if img is None:
                    skipped += 1
                    continue
            else:
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
            'path': cache_id,
            'filename': str(row.HLY),
            'n_patches': len(tensors),
            'cache_path': cache_path,
        })
        if len(pending_tensors) >= PATCH_BATCH_TARGET:
            flush()
    flush()
    if skipped:
        print(f"  [WARN] Skipped {skipped}")
    if cache_hits:
        print(f"  [CACHE] Cache hits: {cache_hits}/{len(df)}")
    print(f"  [OK] Extracted {len(results)} bags")
    return results


# ============================================================
# STAGE 5: TINY CLAM MODEL
# ============================================================
class TinyCLAM(nn.Module):
    def __init__(self, feat_dim_in, hidden=MIL_HIDDEN, n_classes=NUM_CLASSES,
                 dropout=0.25, k_sample=CLAM_K_SAMPLE):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.feat_noise = FEAT_NOISE_STD
        self.feat_drop = nn.Dropout(FEAT_DROP_P)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.att_branches = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 4), nn.Tanh())
            for _ in range(n_classes)
        ])
        self.gate_branches = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 4), nn.Sigmoid())
            for _ in range(n_classes)
        ])
        self.att_combine = nn.ModuleList([
            nn.Linear(hidden // 4, 1) for _ in range(n_classes)
        ])
        self.inst_classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 2))
            for _ in range(n_classes)
        ])
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

    def forward(self, x, label=None, return_bag_embedding=False):
        input_dtype = x.dtype
        if USE_PER_BAG_STD:
            x_f = x.float()
            x = ((x_f - x_f.mean(0, keepdim=True)) / (x_f.std(0, keepdim=True) + 1e-6)).to(input_dtype)
        if self.training:
            x = x.float()
            x = x + torch.randn_like(x) * self.feat_noise
            x = x.to(input_dtype)
            x = self.feat_drop(x)
        h = self.fc(x)
        logits = []
        bag_embeddings = []
        total_inst = torch.tensor(0.0, device=x.device)
        for c in range(self.n_classes):
            a = self.att_branches[c](h)
            g = self.gate_branches[c](h)
            scores = self.att_combine[c](a * g).squeeze(-1)
            weights = F.softmax(scores, dim=0)
            bag = torch.sum(weights.unsqueeze(-1) * h, dim=0)
            bag_embeddings.append(bag)
            logits.append(self.bag_classifiers[c](bag))
            if self.training and label is not None and label.item() == c:
                total_inst += self._inst_loss(scores.detach(), h, self.inst_classifiers[c], self.k_sample)
        out = {'logits': torch.cat(logits), 'inst_loss': total_inst}
        if return_bag_embedding:
            # Concatenate per-class bag embeddings as the bag representation
            out['bag_embedding'] = torch.cat(bag_embeddings, dim=0)
        return out


# ============================================================
# STAGE 6: LOSSES
# ============================================================
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


def kl_distill_loss(student_logits, teacher_logits, T=DISTILL_T):
    student_log_probs = F.log_softmax(student_logits / T, dim=0)
    teacher_probs = F.softmax(teacher_logits / T, dim=0)
    return F.kl_div(student_log_probs.unsqueeze(0), teacher_probs.unsqueeze(0),
                    reduction='batchmean') * (T * T)


# ============================================================
# STAGE 7: TRAINING UTILS
# ============================================================
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
    print(f"Class weights (HGC x{HGC_WEIGHT_BOOST}):")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls}: count={counts.get(i,0)} -> w={weights[i]:.3f}")
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


def warmup_cosine(optimizer, warmup, total):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# STAGE 8: TEACHER TRAINING
# ============================================================
def train_teacher(train_data, val_data, class_weights):
    print("\n" + "=" * 70)
    print("TRAINING TEACHER")
    print("=" * 70)
    set_seed(42)
    teacher = TinyCLAM(feat_dim_in=feat_dim, hidden=MIL_HIDDEN,
                       dropout=MIL_DROPOUT_TEACHER).to(DEVICE)
    n_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"  Teacher params: {n_params:,}")
    focal = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTH).to(DEVICE)
    optim = torch.optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WD)
    sched = warmup_cosine(optim, WARMUP_EPOCHS, TEACHER_EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(TEACHER_EPOCHS):
        teacher.train()
        epoch_data = class_balanced_sample(train_data)
        running, correct, total = 0.0, 0, 0
        for sample in epoch_data:
            feats = sample['features'].to(DEVICE)
            if feats.shape[0] > MAX_PATCHES_TRAIN:
                idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TRAIN]
                feats = feats[idx]
            label = torch.tensor(sample['label'], dtype=torch.long, device=DEVICE)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                out = teacher(feats, label=label)
                loss = supervised_loss(out, label, class_weights, focal)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(teacher.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
            running += loss.item()
            correct += (out['logits'].argmax().item() == sample['label'])
            total += 1
        sched.step()
        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)

        teacher.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for sample in val_data:
                feats = sample['features'].to(DEVICE)
                if feats.shape[0] > MAX_PATCHES_TEST:
                    idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TEST]
                    feats = feats[idx]
                label = torch.tensor(sample['label'], dtype=torch.long, device=DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    out = teacher(feats)
                loss = F.cross_entropy(out['logits'].float().unsqueeze(0),
                                       label.unsqueeze(0), weight=class_weights)
                v_loss += loss.item()
                if out['logits'].argmax().item() == sample['label']:
                    v_correct += 1
        v_loss /= len(val_data)
        v_acc = v_correct / len(val_data)

        improved = ""
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = copy.deepcopy(teacher.state_dict())
            patience = 0
            improved = " *"
        else:
            patience += 1

        if epoch < 3 or (epoch + 1) % 3 == 0 or improved or epoch == TEACHER_EPOCHS - 1:
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                  f"val_loss={v_loss:.4f} val_acc={v_acc:.3f}{improved}")
        if patience >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        teacher.load_state_dict(best_state)
        print(f"    [OK] Restored teacher (val_loss={best_val_loss:.4f})")

    # SAVE IMMEDIATELY
    teacher_path = os.path.join(OUTPUT_DIR, 'teacher.pt')
    safe_torch_save(teacher.state_dict(), teacher_path)
    print(f"    [OK] Teacher checkpoint saved: {teacher_path}")
    return teacher


# ============================================================
# STAGE 9: STUDENT TRAINING
# ============================================================
@torch.no_grad()
def compute_teacher_val_logits(teacher, val_data):
    teacher.eval()
    teacher_logits = []
    for sample in val_data:
        feats = sample['features'].to(DEVICE)
        if feats.shape[0] > MAX_PATCHES_TEST:
            idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TEST]
            feats = feats[idx]
        with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
            out = teacher(feats)
        teacher_logits.append(out['logits'].float().detach().cpu())
    return teacher_logits


def train_student(student_idx, train_data, val_data, val_teacher_logits, class_weights):
    seed = STUDENT_SEEDS[student_idx]
    dropout = MIL_DROPOUT_STUDENTS[student_idx]
    print(f"Student {student_idx+1}/{N_STUDENTS}: seed={seed}, dropout={dropout}")
    set_seed(seed)
    student = TinyCLAM(feat_dim_in=feat_dim, hidden=MIL_HIDDEN, dropout=dropout).to(DEVICE)
    focal = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTH).to(DEVICE)
    optim = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WD)
    sched = warmup_cosine(optim, WARMUP_EPOCHS, STUDENT_EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        epoch_data = class_balanced_sample(train_data)
        running_sup, running_distill, correct, total = 0.0, 0.0, 0, 0
        for sample in epoch_data:
            feats = sample['features'].to(DEVICE)
            if feats.shape[0] > MAX_PATCHES_TRAIN:
                idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TRAIN]
                feats = feats[idx]
            label = torch.tensor(sample['label'], dtype=torch.long, device=DEVICE)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                out = student(feats, label=label)
                sup_loss = supervised_loss(out, label, class_weights, focal)
            scaler.scale(sup_loss).backward()
            running_sup += sup_loss.item()

            if random.random() < 0.5:
                val_idx = random.randrange(len(val_data))
                val_sample = val_data[val_idx]
                val_feats = val_sample['features'].to(DEVICE)
                if val_feats.shape[0] > MAX_PATCHES_TRAIN:
                    idx = torch.randperm(val_feats.shape[0])[:MAX_PATCHES_TRAIN]
                    val_feats = val_feats[idx]
                t_logits = val_teacher_logits[val_idx].to(DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    val_out = student(val_feats)
                    distill_loss = DISTILL_W * kl_distill_loss(val_out['logits'].float(), t_logits)
                scaler.scale(distill_loss).backward()
                running_distill += distill_loss.item()

            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
            correct += (out['logits'].argmax().item() == sample['label'])
            total += 1
        sched.step()
        train_loss = running_sup / max(total, 1)
        distill_avg = running_distill / max(total, 1)
        train_acc = correct / max(total, 1)

        student.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for sample in val_data:
                feats = sample['features'].to(DEVICE)
                if feats.shape[0] > MAX_PATCHES_TEST:
                    idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TEST]
                    feats = feats[idx]
                label = torch.tensor(sample['label'], dtype=torch.long, device=DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    out = student(feats)
                loss = F.cross_entropy(out['logits'].float().unsqueeze(0),
                                       label.unsqueeze(0), weight=class_weights)
                v_loss += loss.item()
                if out['logits'].argmax().item() == sample['label']:
                    v_correct += 1
        v_loss /= len(val_data)
        v_acc = v_correct / len(val_data)

        improved = ""
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = copy.deepcopy(student.state_dict())
            patience = 0
            improved = " *"
        else:
            patience += 1

        if epoch < 3 or (epoch + 1) % 3 == 0 or improved or epoch == STUDENT_EPOCHS - 1:
            print(f"    Epoch {epoch+1:3d}: sup={train_loss:.4f} distill={distill_avg:.4f} "
                  f"train_acc={train_acc:.3f} | val_loss={v_loss:.4f} val_acc={v_acc:.3f}{improved}")
        if patience >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        student.load_state_dict(best_state)
        print(f"    [OK] Restored student (val_loss={best_val_loss:.4f})")

    # SAVE IMMEDIATELY
    s_path = os.path.join(OUTPUT_DIR, f'student_{student_idx}.pt')
    safe_torch_save(student.state_dict(), s_path)
    print(f"    [OK] Student checkpoint saved: {s_path}")
    return student


# ============================================================
# STAGE 10: TTA + INFERENCE
# ============================================================
def tta_transform(img_bgr, tta_idx):
    if tta_idx == 0:
        return img_bgr
    elif tta_idx == 1:
        return cv2.flip(img_bgr, 1)
    elif tta_idx == 2:
        return cv2.flip(img_bgr, 0)
    elif tta_idx == 3:
        return cv2.flip(cv2.flip(img_bgr, 1), 0)
    elif tta_idx == 4:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif tta_idx == 5:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif tta_idx == 6:
        return cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=10)
    elif tta_idx == 7:
        return cv2.convertScaleAbs(img_bgr, alpha=0.9, beta=-10)
    return img_bgr


def extract_features_with_tta(df, norm, desc="TTA features"):
    """Compute TTA features per image and keep them in RAM without disk cache."""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        per_image_features = []
        try:
            img_orig = load_image(str(row.path), norm)
        except Exception:
            results.append(None)
            continue
        for tta_idx in range(N_TTA_VIEWS):
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
            per_image_features.append(feats)
        results.append({
            'tta_features': per_image_features,
            'label': int(row.target),
            'patient': int(row.patient),
            'path': str(row.path),
            'filename': str(row.HLY),
        })
    print(f"  [OK] Extracted TTA features for {len(results)} images x {N_TTA_VIEWS} views (RAM only)")
    return results


def cleanup_train_cache():
    """Remove feature cache to keep enough disk headroom for checkpoints."""
    if os.path.exists(CACHE_DIR):
        size = sum(os.path.getsize(os.path.join(root, f))
                   for root, _, files in os.walk(CACHE_DIR) for f in files)
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"  [OK] Freed {size / 1e9:.2f} GB from train feature cache")


@torch.no_grad()
def predict_with_tta_ensemble(students, tta_data_item, return_embedding=False):
    if tta_data_item is None:
        if return_embedding:
            return np.array([1/3, 1/3, 1/3]), None
        return np.array([1/3, 1/3, 1/3])
    all_probs = []
    all_embeds = []
    for tta_feats in tta_data_item['tta_features']:
        if tta_feats is None or tta_feats.shape[0] == 0:
            continue
        feats = tta_feats.to(DEVICE)
        if feats.shape[0] > MAX_PATCHES_TEST:
            idx = torch.randperm(feats.shape[0])[:MAX_PATCHES_TEST]
            feats = feats[idx]
        for student in students:
            student.eval()
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                out = student(feats, return_bag_embedding=return_embedding)
            probs = F.softmax(out['logits'].float(), dim=0).cpu().numpy()
            all_probs.append(probs)
            if return_embedding and 'bag_embedding' in out:
                all_embeds.append(out['bag_embedding'].float().cpu().numpy())
    if not all_probs:
        if return_embedding:
            return np.array([1/3, 1/3, 1/3]), None
        return np.array([1/3, 1/3, 1/3])
    avg_probs = np.mean(all_probs, axis=0)
    avg_probs = avg_probs / (avg_probs.sum() + 1e-8)
    if return_embedding:
        avg_embed = np.mean(all_embeds, axis=0) if all_embeds else None
        return avg_probs, avg_embed
    return avg_probs


def evaluate_full(students, tta_data, desc="Evaluating", collect_embeddings=False):
    all_preds, all_labels, all_probs, all_embeds, all_meta = [], [], [], [], []
    for item in tqdm(tta_data, desc=desc):
        if item is None:
            continue
        if collect_embeddings:
            probs, embed = predict_with_tta_ensemble(students, item, return_embedding=True)
            all_embeds.append(embed)
        else:
            probs = predict_with_tta_ensemble(students, item)
        all_preds.append(int(np.argmax(probs)))
        all_labels.append(item['label'])
        all_probs.append(probs)
        all_meta.append({'patient': item['patient'], 'filename': item['filename'], 'path': item['path']})
    if collect_embeddings:
        return all_preds, all_labels, all_probs, all_embeds, all_meta
    return all_preds, all_labels, all_probs, all_meta


# ============================================================
# STAGE 11: REPORTING
# ============================================================
def report_metrics(preds, labels, name):
    preds = np.array(preds)
    labels = np.array(labels)
    if len(preds) == 0:
        print(f"  [WARN] {name}: no samples")
        return None
    acc = accuracy_score(labels, preds)
    bal = balanced_accuracy_score(labels, preds)
    print(f"{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:          {acc*100:.2f}%")
    print(f"  Balanced Accuracy: {bal*100:.2f}%")
    # Only show classification report if all classes present
    present = sorted(set(labels.tolist()) | set(preds.tolist()))
    target_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if i in present]
    print(classification_report(labels, preds, labels=present,
                                target_names=target_names, digits=4, zero_division=0))
    hgc_idx = CLASS_TO_IDX['HGC']
    hgc_true = (labels == hgc_idx)
    hgc_pred = (preds == hgc_idx)
    hgc_tp = (hgc_pred & hgc_true).sum()
    hgc_r = hgc_tp / max(hgc_true.sum(), 1)
    hgc_p = hgc_tp / max(hgc_pred.sum(), 1)
    print(f"  HGC: Recall={hgc_r*100:.2f}%, Precision={hgc_p*100:.2f}%")
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print(f"Confusion Matrix:")
    print(f"  {'':>10} {'  '.join(f'{c:>8s}' for c in CLASS_NAMES)}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:>10} {'  '.join(f'{cm[i,j]:8d}' for j in range(NUM_CLASSES))}")
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal),
        'hgc_recall': float(hgc_r),
        'hgc_precision': float(hgc_p),
        'confusion_matrix': cm.tolist(),
    }


def save_cm(preds, labels, fname, title):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


# ============================================================
# STAGE 12: DIAGNOSTICS - modality, per-patient, UMAP, t-SNE
# ============================================================
def build_modality_lookup(df_orig):
    """Map filename -> 'WLI' / 'NBI' (case-insensitive)."""
    lookup = {}
    col = None
    for candidate in ['imaging type', 'imaging_type', 'modality', 'image type']:
        if candidate in df_orig.columns:
            col = candidate
            break
    if col is None:
        # Try to find any column whose name contains "imag"
        for c in df_orig.columns:
            if 'imag' in c.lower():
                col = c
                break
    if col is None:
        print("  [WARN] No imaging type column found; modality stratification disabled")
        return lookup, None
    for _, row in df_orig.iterrows():
        fname = str(row['HLY']).strip().lower()
        mod = str(row[col]).strip().upper()
        if 'NBI' in mod:
            lookup[fname] = 'NBI'
        elif 'WLI' in mod:
            lookup[fname] = 'WLI'
        else:
            lookup[fname] = mod
    return lookup, col


def modality_stratified_report(preds, labels, meta, modality_lookup):
    """Report metrics for WLI, NBI, Overall."""
    print("\n" + "=" * 70)
    print("  MODALITY-STRATIFIED TEST RESULTS")
    print("=" * 70)

    if not modality_lookup:
        print("  Modality info unavailable.")
        return {}

    bins = {'WLI': ([], []), 'NBI': ([], []), 'Overall': (list(preds), list(labels))}
    for p, l, m in zip(preds, labels, meta):
        mod = modality_lookup.get(str(m['filename']).strip().lower(), None)
        if mod in bins:
            bins[mod][0].append(p)
            bins[mod][1].append(l)

    rows = []
    results = {}
    for name in ['WLI', 'NBI', 'Overall']:
        p_list, l_list = bins[name]
        if len(p_list) == 0:
            continue
        p_arr = np.array(p_list)
        l_arr = np.array(l_list)
        acc = accuracy_score(l_arr, p_arr)
        bal = balanced_accuracy_score(l_arr, p_arr)
        hgc_idx = CLASS_TO_IDX['HGC']
        hgc_true = (l_arr == hgc_idx)
        hgc_pred = (p_arr == hgc_idx)
        hgc_tp = (hgc_pred & hgc_true).sum()
        hgc_r = hgc_tp / max(hgc_true.sum(), 1)
        hgc_p = hgc_tp / max(hgc_pred.sum(), 1)
        # Per-class recall
        class_recalls = {}
        for ci, cname in enumerate(CLASS_NAMES):
            mask = (l_arr == ci)
            if mask.sum() > 0:
                class_recalls[cname] = ((p_arr == ci) & mask).sum() / mask.sum()
            else:
                class_recalls[cname] = float('nan')
        rows.append({
            'modality': name, 'n': len(p_arr),
            'accuracy': acc, 'bal_acc': bal,
            'hgc_recall': hgc_r, 'hgc_precision': hgc_p,
            **{f'recall_{c}': class_recalls[c] for c in CLASS_NAMES}
        })
        results[name] = {
            'n': int(len(p_arr)),
            'accuracy': float(acc), 'bal_acc': float(bal),
            'hgc_recall': float(hgc_r), 'hgc_precision': float(hgc_p),
            'class_recalls': {k: float(v) for k, v in class_recalls.items()},
        }

    # Print clean table
    print(f"{'Modality':<10} {'N':>5} {'Acc':>8} {'BalAcc':>8} {'HGC_R':>8} {'HGC_P':>8} "
          f"{'R_HGC':>8} {'R_LGC':>8} {'R_Norm':>8}")
    print("  " + "-" * 80)
    for r in rows:
        print(f"  {r['modality']:<10} {r['n']:>5d} {r['accuracy']*100:>7.2f}% {r['bal_acc']*100:>7.2f}% "
              f"{r['hgc_recall']*100:>7.2f}% {r['hgc_precision']*100:>7.2f}% "
              f"{r['recall_HGC']*100:>7.2f}% {r['recall_LGC']*100:>7.2f}% {r['recall_Normal']*100:>7.2f}%")

    # Save CSV
    pd.DataFrame(rows).to_csv(os.path.join(DIAG_DIR, 'modality_breakdown.csv'), index=False)
    return results


def per_patient_report(preds, labels, meta):
    print("\n" + "=" * 70)
    print("  PER-PATIENT TEST BREAKDOWN")
    print("=" * 70)
    rows = []
    pids = sorted(set(m['patient'] for m in meta))
    for pid in pids:
        idxs = [i for i, m in enumerate(meta) if m['patient'] == pid]
        p_lab = np.array([labels[i] for i in idxs])
        p_prd = np.array([preds[i] for i in idxs])
        acc = accuracy_score(p_lab, p_prd)
        dist = Counter(p_lab.tolist())
        dist_str = ', '.join(f"{IDX_TO_CLASS[k]}={v}" for k, v in sorted(dist.items()))
        cm = confusion_matrix(p_lab, p_prd, labels=list(range(NUM_CLASSES)))
        print(f"  Patient {pid}: {len(idxs)} imgs ({dist_str}) -> acc={acc*100:.2f}%")
        rows.append({'patient': pid, 'n': len(idxs), 'accuracy': acc, 'distribution': dist_str})
    pd.DataFrame(rows).to_csv(os.path.join(DIAG_DIR, 'per_patient.csv'), index=False)
    return rows


def confidence_distribution_plot(probs, labels, fname):
    probs = np.array(probs)
    labels = np.array(labels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ci, cname in enumerate(CLASS_NAMES):
        ax = axes[ci]
        for true_c in range(NUM_CLASSES):
            mask = (labels == true_c)
            if mask.sum() == 0:
                continue
            ax.hist(probs[mask, ci], bins=20, alpha=0.5,
                    label=f'True={CLASS_NAMES[true_c]}', density=True)
        ax.set_title(f'P({cname}) distribution')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def feature_space_visualizations(embeddings, labels, modalities, fname_prefix):
    """UMAP + t-SNE on bag embeddings, colored by class (and shape by modality)."""
    print("\n" + "=" * 70)
    print("  FEATURE SPACE VISUALIZATIONS")
    print("=" * 70)
    valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
    if len(valid_idx) < 5:
        print("  [WARN] Not enough valid embeddings for visualization")
        return
    X = np.stack([embeddings[i] for i in valid_idx])
    y = np.array([labels[i] for i in valid_idx])
    mods = [modalities[i] for i in valid_idx] if modalities else ['?'] * len(valid_idx)
    print(f"  Embedding matrix: {X.shape}")

    # t-SNE
    try:
        print("  Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(X) // 4)))
        X_tsne = tsne.fit_transform(X)
        plot_embedding(X_tsne, y, mods, "t-SNE of bag embeddings",
                       os.path.join(DIAG_DIR, f'{fname_prefix}_tsne.png'))
    except Exception as e:
        print(f"  [WARN] t-SNE failed: {e}")

    # UMAP
    try:
        import umap
        print("  Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=min(15, max(3, len(X) // 5)))
        X_umap = reducer.fit_transform(X)
        plot_embedding(X_umap, y, mods, "UMAP of bag embeddings",
                       os.path.join(DIAG_DIR, f'{fname_prefix}_umap.png'))
    except Exception as e:
        print(f"  [WARN] UMAP failed: {e}")


def plot_embedding(X2d, y, mods, title, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#27ae60'}
    markers = {'WLI': 'o', 'NBI': 's', '?': 'x'}
    for cls in range(NUM_CLASSES):
        for mod in set(mods):
            mask = np.array([(y[i] == cls and mods[i] == mod) for i in range(len(y))])
            if mask.sum() == 0:
                continue
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       c=colors[cls], marker=markers.get(mod, 'x'),
                       label=f'{CLASS_NAMES[cls]} ({mod})',
                       alpha=0.7, s=80, edgecolors='black', linewidths=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"    [OK] Saved {fname}")


# ============================================================
# STAGE 13: BUNDLE OUTPUTS
# ============================================================
def bundle_outputs():
    print("\n" + "=" * 70)
    print("  BUNDLING OUTPUTS")
    print("=" * 70)
    if os.path.exists(BUNDLE_PATH):
        os.remove(BUNDLE_PATH)
    with zipfile.ZipFile(BUNDLE_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in [OUTPUT_DIR, DIAG_DIR]:
            if not os.path.exists(folder):
                continue
            for root, _, files in os.walk(folder):
                for f in files:
                    full = os.path.join(root, f)
                    arc = os.path.relpath(full, os.path.dirname(BUNDLE_PATH))
                    zf.write(full, arc)
    size_mb = os.path.getsize(BUNDLE_PATH) / 1024 / 1024
    print(f"  [OK] Bundle saved: {BUNDLE_PATH} ({size_mb:.1f} MB)")
    print(f"  [CACHE] Download this file BEFORE the notebook becomes inactive!")


# ============================================================
# MAIN
# ============================================================
def main():
    start = time.time()
    print("+======================================================================+")
    print("|  Bladder Classification - v3 ARGMAX (single-notebook)               |")
    print("|  Frozen DINOv2 + DenseNet -> TinyCLAM (256/1head)                    |")
    print("|  Teacher -> 5 Students with KL-distill -> 8-view TTA -> ARGMAX         |")
    print("|  + Modality stratification + UMAP/t-SNE diagnostics                 |")
    print("+======================================================================+")
    print(f"  Device: {DEVICE} | AMP: {USE_AMP}")
    print(f"  Platform: {'Kaggle' if IS_KAGGLE else 'Lightning AI' if IS_LIGHTNING else 'Local'}")
    print(f"  Data dir: {ORIG_DATA_DIR}")
    print(f"  Annotations: {ANNOTATIONS_CSV}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Stage 1
    run_stage1_augmentation()

    # Stage 2
    train_df, val_df, test_df, df_orig_full = load_data()

    # Modality lookup
    modality_lookup, mod_col = build_modality_lookup(df_orig_full)
    if mod_col:
        print(f"  [OK] Modality column: '{mod_col}', mapped {len(modality_lookup)} entries")

    # Stage 3-4
    global feat_dim
    feat_dim = load_backbones()
    norm = fit_normalizer(train_df.sample(min(len(train_df), 200), random_state=42))

    print("\n" + "=" * 70)
    print("EXTRACTING TRAIN FEATURES")
    print("=" * 70)
    train_data = extract_features_for_split(
        train_df, "Train", norm, use_cache=True, from_manifest=True)

    print("\n" + "=" * 70)
    print("EXTRACTING VAL FEATURES (single view, for training)")
    print("=" * 70)
    val_data = extract_features_for_split(
        val_df, "Val (clean)", norm, use_cache=True)

    print(f"Train bags: {len(train_data)} | Val bags: {len(val_data)}")
    class_weights = compute_class_weights(train_data)

    # Features are already in RAM; make room for teacher/student checkpoints.
    # cleanup_train_cache()  # Commented out to preserve cache

    # Stage 5: Teacher
    teacher = train_teacher(train_data, val_data, class_weights)

    # Stage 6: Teacher logits + Students
    print("Computing teacher's val logits...")
    teacher_val_logits = compute_teacher_val_logits(teacher, val_data)
    print(f"  [OK] Cached {len(teacher_val_logits)} teacher logits")

    print("\n" + "=" * 70)
    print(f"TRAINING {N_STUDENTS} STUDENTS WITH DISTILLATION")
    print("=" * 70)
    students = []
    for s_idx in range(N_STUDENTS):
        student = train_student(s_idx, train_data, val_data, teacher_val_logits, class_weights)
        student.eval()
        students.append(student)

    # Free disk before TTA (TTA features stay in RAM, but we still need headroom)
    # cleanup_train_cache()  # Commented out to preserve cache
    free_gb = shutil.disk_usage(os.path.dirname(BUNDLE_PATH)).free / 1e9
    print(f"\n  Disk free before TTA: {free_gb:.2f} GB")

    # Stage 7: TTA features
    print("\n" + "=" * 70)
    print("EXTRACTING TTA FEATURES FOR VAL + TEST")
    print("=" * 70)
    val_tta = extract_features_with_tta(val_df, norm, "Val TTA")
    test_tta = extract_features_with_tta(test_df, norm, "Test TTA")

    # Stage 8: Argmax evaluation (ensemble + TTA)
    print("\n" + "=" * 70)
    print("EVALUATION (ensemble + TTA, ARGMAX)")
    print("=" * 70)

    val_preds, val_labels, val_probs, val_meta = evaluate_full(students, val_tta, "Val")
    val_metrics = report_metrics(val_preds, val_labels, "VAL - ARGMAX")

    test_preds, test_labels, test_probs, test_embeds, test_meta = evaluate_full(
        students, test_tta, "Test", collect_embeddings=True)
    test_metrics = report_metrics(test_preds, test_labels, "TEST - ARGMAX (HEADLINE)")

    # Stage 9: Diagnostics
    modality_results = modality_stratified_report(test_preds, test_labels, test_meta, modality_lookup)
    per_patient = per_patient_report(test_preds, test_labels, test_meta)

    save_cm(test_preds, test_labels, os.path.join(OUTPUT_DIR, 'cm_test_argmax.png'), "v3-Argmax Test")
    save_cm(val_preds, val_labels, os.path.join(OUTPUT_DIR, 'cm_val_argmax.png'), "v3-Argmax Val")

    # Confidence histograms
    confidence_distribution_plot(test_probs, test_labels,
                                 os.path.join(DIAG_DIR, 'confidence_dist_test.png'))

    # UMAP / t-SNE
    test_modalities = [modality_lookup.get(str(m['filename']).strip().lower(), '?') for m in test_meta]
    feature_space_visualizations(test_embeds, test_labels, test_modalities, 'test')

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'patient': [m['patient'] for m in test_meta],
        'filename': [m['filename'] for m in test_meta],
        'modality': test_modalities,
        'true_class': [IDX_TO_CLASS[l] for l in test_labels],
        'pred_class': [IDX_TO_CLASS[p] for p in test_preds],
        'prob_HGC': [p[0] for p in test_probs],
        'prob_LGC': [p[1] for p in test_probs],
        'prob_Normal': [p[2] for p in test_probs],
    })
    pred_df.to_csv(os.path.join(DIAG_DIR, 'test_predictions.csv'), index=False)

    # Final summary JSON
    elapsed = time.time() - start
    summary = {
        'experiment': 'v3_argmax',
        'timestamp': datetime.now().isoformat(),
        'runtime_minutes': elapsed / 60,
        'split': {'train': TRAIN_PATIENTS, 'val': VAL_PATIENTS, 'test': TEST_PATIENTS},
        'config': {
            'mil_hidden': MIL_HIDDEN, 'n_att_heads': N_ATT_HEADS,
            'n_students': N_STUDENTS, 'student_dropouts': MIL_DROPOUT_STUDENTS,
            'n_tta_views': N_TTA_VIEWS, 'distill_w': DISTILL_W, 'distill_T': DISTILL_T,
            'hgc_weight_boost': HGC_WEIGHT_BOOST,
            'frozen_backbone': True, 'per_bag_std': USE_PER_BAG_STD,
            'threshold_tuning': False,
        },
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'modality_results': modality_results,
    }
    with open(os.path.join(OUTPUT_DIR, 'v3_argmax_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Diagnostic markdown report
    with open(os.path.join(DIAG_DIR, 'diagnostic_report.md'), 'w') as f:
        f.write(f"# v3-Argmax Diagnostic Report")
        f.write(f"**Run time:** {elapsed/60:.1f} min")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}")
        f.write(f"## Test (Headline)")
        f.write(f"- Accuracy: **{test_metrics['accuracy']*100:.2f}%**")
        f.write(f"- Balanced Accuracy: **{test_metrics['balanced_accuracy']*100:.2f}%**")
        f.write(f"- HGC Recall: **{test_metrics['hgc_recall']*100:.2f}%**")
        f.write(f"- HGC Precision: **{test_metrics['hgc_precision']*100:.2f}%**")
        f.write(f"## Modality Breakdown")
        for name, r in modality_results.items():
            f.write(f"### {name} (N={r['n']})")
            f.write(f"- Acc: {r['accuracy']*100:.2f}%, Bal: {r['bal_acc']*100:.2f}%")
            f.write(f"- HGC R/P: {r['hgc_recall']*100:.2f}% / {r['hgc_precision']*100:.2f}%")

    # Stage 10: Bundle
    bundle_outputs()

    # FINAL HEADLINE BOX
    print("")
    print("#" * 72)
    print("#" + " " * 70 + "#")
    print("#" + f"  v3-ARGMAX FINAL - Runtime: {elapsed/60:.1f} min".ljust(70) + "#")
    print("#" + " " * 70 + "#")
    print("#" + f"  TEST  acc={test_metrics['accuracy']*100:.2f}%  "f"bal={test_metrics['balanced_accuracy']*100:.2f}%  "f"HGC_R={test_metrics['hgc_recall']*100:.2f}%".ljust(70) + "#")
    if 'WLI' in modality_results:
        r = modality_results['WLI']
        print("#" + f"  WLI   acc={r['accuracy']*100:.2f}%  bal={r['bal_acc']*100:.2f}%  "f"HGC_R={r['hgc_recall']*100:.2f}%  (N={r['n']})".ljust(70) + "#")
    if 'NBI' in modality_results:
        r = modality_results['NBI']
        print("#" + f"  NBI   acc={r['accuracy']*100:.2f}%  bal={r['bal_acc']*100:.2f}%  "f"HGC_R={r['hgc_recall']*100:.2f}%  (N={r['n']})".ljust(70) + "#")
    print("#" + " " * 70 + "#")
    print("#" + f"  Bundle: {BUNDLE_PATH}".ljust(70) + "#")
    print("#" + " " * 70 + "#")
    print("#" * 72)


if __name__ == '__main__':
    main()
