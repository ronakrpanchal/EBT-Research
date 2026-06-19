import re

with open("/Applications/Projects/Bladder Research/Model Training and Evaluation/bladder_stage1_finetune.py", "r") as f:
    content = f.read()

def resolve_manifest_path(row, orig_dir, aug_dir):
    import os
    if row['is_augmented'] == 1:
        fname = row['full_path'].split('/')[-1]
        p = os.path.join(aug_dir, row['tissue type'], fname)
        return p if os.path.exists(p) else None
    else:
        p = os.path.join(orig_dir, row['tissue type'], row['source_filename'])
        return p if os.path.exists(p) else None

# We need to inject resolve_manifest_path into the file.
if "def resolve_manifest_path" not in content:
    content = content.replace("def scan_for_images():", """def resolve_manifest_path(row, orig_dir, aug_dir):
    import os
    if row['is_augmented'] == 1:
        fname = str(row['full_path']).split('/')[-1]
        p = os.path.join(aug_dir, row['tissue type'], fname)
        return p if os.path.exists(p) else None
    else:
        p = os.path.join(orig_dir, row['tissue type'], row['source_filename'])
        return p if os.path.exists(p) else None

def scan_for_images():""")

# Replace the data loading logic in main()
old_loading = """    orig_df = pd.read_csv(ANNOTATIONS_CSV)
    orig_df.columns = orig_df.columns.str.strip()
    records = []
    for _, row in orig_df.iterrows():
        fname = str(row['HLY']).strip()
        path = IMAGE_PATH_INDEX.get(fname.lower())
        if path:
            records.append({
                'filename': fname,
                'path': path,
                'label': LABEL_MAP.get(row['tissue type'], 2),
                'patient_id': extract_patient_id(fname)
            })
    base_df = pd.DataFrame(records)
    
    aug_df = pd.read_csv(AUG_TRAIN_MANIFEST)
    aug_records = []
    for _, row in aug_df.iterrows():
        fname = row['filename']
        path = IMAGE_PATH_INDEX.get(fname.lower())
        if path:
            aug_records.append({
                'filename': fname,
                'path': path,
                'label': row['label'],
                'patient_id': extract_patient_id(fname)
            })
    aug_full_df = pd.DataFrame(aug_records)"""

new_loading = """    df_orig = pd.read_csv(ANNOTATIONS_CSV)
    df_orig.columns = df_orig.columns.str.strip()
    df_orig['patient_id'] = df_orig['HLY'].apply(extract_patient_id)
    df_orig['label']  = df_orig['tissue type'].map(LABEL_MAP).fillna(2).astype(int)
    df_orig['path']   = df_orig.apply(lambda r: IMAGE_PATH_INDEX.get(str(r['HLY']).strip().lower()), axis=1)
    base_df = df_orig[df_orig['path'].notna()].copy()
    
    full_manifest = pd.read_csv(AUG_TRAIN_MANIFEST)
    full_manifest['label'] = full_manifest['tissue type'].map(LABEL_MAP).fillna(2).astype(int)
    full_manifest['path'] = full_manifest.apply(lambda r: resolve_manifest_path(r, ORIG_DATA_DIR, AUG_TRAIN_DIR), axis=1)
    aug_full_df = full_manifest[full_manifest['path'].notna()].copy()"""

content = content.replace(old_loading, new_loading)

with open("/Applications/Projects/Bladder Research/Model Training and Evaluation/bladder_stage1_finetune.py", "w") as f:
    f.write(content)
