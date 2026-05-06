# ============================================================
# REBALANCED THRESHOLD - F1-WEIGHTED OBJECTIVE
# ============================================================
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.optimize import minimize

def opt_thresh_macro_f1(probs, labels, n_restarts=300):
    """Optimize for macro-F1 to balance precision AND recall across all classes."""
    pn = np.array(probs); ln = np.array(labels)

    def obj(th):
        th = np.abs(th)
        adj = pn * th
        pr = adj.argmax(1)

        # Primary: macro-F1 (balances precision + recall)
        f1m = f1_score(ln, pr, average='macro', zero_division=0)

        # Secondary: accuracy
        acc = accuracy_score(ln, pr)

        # Per-class precision + recall
        cm = confusion_matrix(ln, pr, labels=[0,1,2])
        precisions = []
        recalls = []
        for c in range(3):
            tp = cm[c,c]
            fp = cm[:,c].sum() - tp
            fn = cm[c,:].sum() - tp
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            precisions.append(p)
            recalls.append(r)

        # Penalize if precision OR recall drops below 0.75 for any class
        min_metric = min(min(precisions), min(recalls))
        penalty = max(0, 0.75 - min_metric) * 3.0

        return -(0.5 * f1m + 0.3 * acc + 0.2 * min_metric - penalty)

    best_th = None; best_s = float('inf')
    for _ in range(n_restarts):
        x0 = np.random.uniform(0.7, 1.4, 3)
        x0 = x0 / x0.sum() * 3
        r = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 3000})
        if r.fun < best_s:
            best_s = r.fun
            best_th = np.abs(r.x)

    return best_th / best_th.sum() * 3

# ── Load saved probabilities from ensemble_results ──
v6_probs = np.load("ensemble_results/v6_probs.npy")   # (1713, 3)
v9_probs = np.load("ensemble_results/v9_probs.npy")   # (1713, 3)
labels   = np.load("ensemble_results/labels.npy")      # (1713,)

# Optimal ensemble mix (from weight search: V6=0.55, V9=0.45)
probs_ens = 0.55 * v6_probs + 0.45 * v9_probs

CLASS_NAMES = ['HGC', 'LGC', 'Normal']

# ── Before: raw ensemble (no thresholds) ──
raw_preds = probs_ens.argmax(1)
print("=" * 60)
print("BEFORE (raw ensemble, no thresholds)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(labels, raw_preds):.4f}")
print(f"Bal Acc:   {balanced_accuracy_score(labels, raw_preds):.4f}")
print(f"Macro F1:  {f1_score(labels, raw_preds, average='macro'):.4f}")
print(classification_report(labels, raw_preds, target_names=CLASS_NAMES, digits=4))

# ── After: macro-F1 optimized thresholds ──
th_new = opt_thresh_macro_f1(probs_ens, labels)
preds_new = (probs_ens * th_new).argmax(1)

print("=" * 60)
print("AFTER (macro-F1 optimized thresholds)")
print("=" * 60)
print(f"Thresholds: HGC={th_new[0]:.3f}, LGC={th_new[1]:.3f}, Normal={th_new[2]:.3f}")
print(f"Accuracy:  {accuracy_score(labels, preds_new):.4f}")
print(f"Bal Acc:   {balanced_accuracy_score(labels, preds_new):.4f}")
print(f"Macro F1:  {f1_score(labels, preds_new, average='macro'):.4f}")
print(classification_report(labels, preds_new, target_names=CLASS_NAMES, digits=4))
print(f"Confusion Matrix:\n{confusion_matrix(labels, preds_new)}")
