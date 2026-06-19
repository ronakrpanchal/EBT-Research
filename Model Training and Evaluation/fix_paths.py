import os
import re

with open("/Applications/Projects/Bladder Research/Model Training and Evaluation/bladder_stage1_finetune.py", "r") as f:
    content = f.read()

old_paths = """if IS_LIGHTNING:
    LIGHTNING_WORK_DIR = Path('/teamspace/studios/this_studio')
    ORIG_DATA_DIR = '/teamspace/studios/this_studio/ebt-22/EndoscopicBladderTissue/EndoscopicBladderTissue'
    ANNOTATIONS_CSV = '/teamspace/studios/this_studio/ebt-22/EndoscopicBladderTissue/EndoscopicBladderTissue/annotations_fixed.csv'
    AUG_TRAIN_DIR = str(LIGHTNING_WORK_DIR / 'v3_augmented_all')
    AUG_TRAIN_MANIFEST = str(LIGHTNING_WORK_DIR / 'v3_augmented_all_manifest.csv')
    OUTPUT_DIR = str(LIGHTNING_WORK_DIR / 'output_stage1')
else:
    ORIG_DATA_DIR = str(workspace.parent / 'Data' / 'EndoscopicBladderTissue')
    ANNOTATIONS_CSV = str(workspace.parent / 'Data' / 'annotations_fixed.csv')
    AUG_TRAIN_DIR = str(workspace / 'augmented_data_22' / 'augmented_data_22')
    AUG_TRAIN_MANIFEST = str(workspace / 'augmented_data_22' / 'augmented_data_22' / 'combined_manifest.csv')
    if not os.path.exists(AUG_TRAIN_MANIFEST):
        AUG_TRAIN_DIR = str(workspace / 'v3_augmented_all')
        AUG_TRAIN_MANIFEST = str(workspace / 'v3_augmented_all_manifest.csv')
    OUTPUT_DIR = str(workspace / 'output_stage1')"""

new_paths = """def find_first_dir(search_roots, dirname):
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
    OUTPUT_DIR = str(workspace / 'output_stage1')"""

content = content.replace(old_paths, new_paths)

with open("/Applications/Projects/Bladder Research/Model Training and Evaluation/bladder_stage1_finetune.py", "w") as f:
    f.write(content)
