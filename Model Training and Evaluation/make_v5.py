import os
import re

with open("/Applications/Projects/Bladder Research/Model Training and Evaluation/bladder_5fold_cv_v4_convnext.py", "r") as f:
    content = f.read()

# 1. Config strings
content = content.replace("'v4_convnext'", "'v5_2stage'")
content = content.replace("output_5foldcv_v4_convnext", "output_5foldcv_v5_2stage")
content = content.replace("feat_cache_v4", "feat_cache_v5")
content = content.replace("bladder_5fold_cv_v4_convnext.py", "bladder_5fold_cv_v5_2stage.py")
content = content.replace("v4 5-Fold CV CONVNEXT", "v5 5-Fold CV (2-STAGE FINETUNED)")

# 2. Update backbone loading logic
old_load = """convnext_model = None
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
    
    return feat_dim"""

new_load = """convnext_model = None
feat_dim = 0

def load_backbones(fold_idx):
    global convnext_model, feat_dim, IMNET_MEAN, IMNET_STD
    print("" + "=" * 60)
    print(f"LOADING FINETUNED CONVNEXT BACKBONE (Fold {fold_idx})")
    print("=" * 60)
    
    IMNET_MEAN = IMNET_MEAN.to(DEVICE)
    IMNET_STD = IMNET_STD.to(DEVICE)
    
    print(f"  Loading convnext_tiny for Fold {fold_idx}...")
    try:
        # Load the base model with 3 classes as it was fine-tuned
        convnext_model = timm.create_model('convnext_tiny', pretrained=False, num_classes=3)
        
        # Load finetuned weights
        stage1_dir = OUTPUT_DIR.replace('output_5foldcv_v5_2stage', 'output_stage1')
        weights_path = os.path.join(stage1_dir, f'convnext_finetuned_fold{fold_idx}.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing Stage 1 weights for fold {fold_idx}: {weights_path}\\nPLEASE RUN bladder_stage1_finetune.py FIRST.")
            
        convnext_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        # Strip the classification head to output 768-dim features
        convnext_model.reset_classifier(0)
        
        convnext_model.eval().to(DEVICE)
        for p in convnext_model.parameters():
            p.requires_grad = False
            
        feat_dim = 768
        print(f"  ✓ convnext_tiny (Finetuned) — FROZEN, dim={feat_dim}")
    except Exception as e:
        print(f"  ⚠ ConvNeXt failed: {e}")
        raise e
    
    return feat_dim"""

content = content.replace(old_load, new_load)

# 3. Update main loop to pass fold_idx to load_backbones
old_main_loop = """        feat_dim_in = load_backbones()
        
        # ── 5. Feature Extraction (Cache to disk) ──"""

new_main_loop = """        feat_dim_in = load_backbones(fold_idx)
        
        # ── 5. Feature Extraction (Cache to disk) ──"""

content = content.replace(old_main_loop, new_main_loop)

# 4. Also fix the TTA extraction part
old_tta_load = """        load_backbones()
        extract_features_for_split(val_df, desc=f"Fold{fold_idx} Val TTA", norm=norm, use_cache=True, tta_views=N_TTA_VIEWS)"""

new_tta_load = """        load_backbones(fold_idx)
        extract_features_for_split(val_df, desc=f"Fold{fold_idx} Val TTA", norm=norm, use_cache=True, tta_views=N_TTA_VIEWS)"""

content = content.replace(old_tta_load, new_tta_load)

with open("/Applications/Projects/Bladder Research/Model Training and Evaluation/bladder_5fold_cv_v5_2stage.py", "w") as f:
    f.write(content)

print("Created bladder_5fold_cv_v5_2stage.py")
