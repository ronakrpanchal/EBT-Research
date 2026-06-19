import os

path = "/Applications/Projects/Bladder Research/Model Training and Evaluation/bladder_5fold_cv_v3_fixed.py"
with open(path, "r") as f:
    content = f.read()

# 1. Update extract_features_for_split to store paths instead of tensors
old_extract = """                    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
                    results.append({
                        'features': cached['features'],
                        'label': int(row.target),
                        'label_name': row.label,
                        'patient': int(row.patient),
                        'path': row.path,
                        'n_patches': cached['features'].shape[0],
                    })"""

new_extract = """                    results.append({
                        'features': cache_path,
                        'label': int(row.target),
                        'label_name': row.label,
                        'patient': int(row.patient),
                        'path': row.path,
                        'n_patches': -1,
                    })"""
content = content.replace(old_extract, new_extract)

old_flush = """            if meta['cache_path']:
                try:
                    torch.save({'features': block}, meta['cache_path'])
                except Exception:
                    pass
            results.append({
                'features': block,
                'label': meta['label'],
                'label_name': meta['label_name'],
                'patient': meta['patient'],
                'path': meta['path'],
                'n_patches': block.shape[0],
            })"""

new_flush = """            if meta['cache_path']:
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
            })"""
content = content.replace(old_flush, new_flush)


# 2. Update mixup_features to load paths
old_mixup = """            bag_a = items[i]['features'].float()
            bag_b = items[j]['features'].float()"""

new_mixup = """            bag_a = items[i]['features']
            if isinstance(bag_a, str):
                bag_a = torch.load(bag_a, map_location='cpu', weights_only=False)['features']
            bag_a = bag_a.float()
            
            bag_b = items[j]['features']
            if isinstance(bag_b, str):
                bag_b = torch.load(bag_b, map_location='cpu', weights_only=False)['features']
            bag_b = bag_b.float()"""
content = content.replace(old_mixup, new_mixup)


# 3. Update train_model loops to load paths
old_train_loop = """        for sample in epoch_data:
            feats = sample['features'].to(DEVICE)"""

new_train_loop = """        for sample in epoch_data:
            feats = sample['features']
            if isinstance(feats, str):
                feats = torch.load(feats, map_location='cpu', weights_only=False)['features']
            feats = feats.to(DEVICE)"""
content = content.replace(old_train_loop, new_train_loop)

old_val_loop = """            for sample in val_data:
                feats = sample['features'].to(DEVICE)"""

new_val_loop = """            for sample in val_data:
                feats = sample['features']
                if isinstance(feats, str):
                    feats = torch.load(feats, map_location='cpu', weights_only=False)['features']
                feats = feats.to(DEVICE)"""
content = content.replace(old_val_loop, new_val_loop)


# 4. Update extract_features_with_tta to store paths
old_tta_extract = """                try:
                    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
                    per_image_features.append(cached['features'])
                    continue
                except Exception:
                    pass"""

new_tta_extract = """                if os.path.exists(cache_path):
                    per_image_features.append(cache_path)
                    continue"""
content = content.replace(old_tta_extract, new_tta_extract)

old_tta_save = """            try:
                torch.save({'features': feats}, cache_path)
            except Exception:
                pass
            per_image_features.append(feats)"""

new_tta_save = """            try:
                torch.save({'features': feats}, cache_path)
            except Exception:
                pass
            per_image_features.append(cache_path if cache_path else feats)"""
content = content.replace(old_tta_save, new_tta_save)


# 5. Update predict_with_tta to load paths
old_predict = """    for tta_feats in tta_data_item['tta_features']:
        if tta_feats is None or tta_feats.shape[0] == 0:
            continue
        feats = tta_feats.to(DEVICE)"""

new_predict = """    for tta_feats in tta_data_item['tta_features']:
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
        feats = feats.to(DEVICE)"""
content = content.replace(old_predict, new_predict)

# Write out the updated fixed file
with open(path, "w") as f:
    f.write(content)

print("Memory optimizations applied to fixed file.")
