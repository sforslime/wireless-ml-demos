"""
Open-set modulation recognition demo.

1. Train a small 1D CNN on 6 KNOWN modulation types.
2. Hold out 3 UNKNOWN types the model never sees in training.
3. Detect unknowns with three scores:
     - max softmax probability (baseline)
     - energy score
     - Mahalanobis distance in the learned embedding space
4. Few-shot incremental step: add the 3 new classes using only
   K labelled examples each (class prototypes, no retraining),
   and check that old classes are not forgotten.

Run:  python run.py          (about 5-10 min on a laptop CPU)
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_auc_score

from signals import make_dataset

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(os.cpu_count() or 2)

KNOWN = ["BPSK", "QPSK", "8PSK", "16QAM", "PAM4", "2FSK"]
UNKNOWN = ["64QAM", "4FSK", "AM-DSB"]
SNRS = list(range(-4, 20, 2))     # -4 dB ... 18 dB
K_SHOT = 10                       # labelled examples per new class
OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ---------------- data ----------------
print("Generating data...")
Xtr, ytr, _ = make_dataset(KNOWN, SNRS, n_per=300, seed=1)
Xva, yva, _ = make_dataset(KNOWN, SNRS, n_per=50, seed=2)
Xte_k, yte_k, ste_k = make_dataset(KNOWN, SNRS, n_per=100, seed=3)
Xte_u, yte_u, ste_u = make_dataset(UNKNOWN, SNRS, n_per=100, seed=4)
Xshot, yshot, _ = make_dataset(UNKNOWN, SNRS, n_per=K_SHOT, seed=5)
print(f"train {len(Xtr)}, test known {len(Xte_k)}, test unknown {len(Xte_u)}")


# ---------------- model ----------------
class SignalCNN(nn.Module):
    def __init__(self, n_classes, emb_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.embed = nn.Linear(128, emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        z = self.embed(self.features(x))
        return self.head(F.relu(z)), z


model = SignalCNN(len(KNOWN))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 20
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)


def batches(X, y, bs, shuffle):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield torch.from_numpy(X[j]), torch.from_numpy(y[j])


@torch.no_grad()
def infer(X):
    model.eval()
    logits, embs = [], []
    for xb, _ in batches(X, np.zeros(len(X), dtype=np.int64), 1024, False):
        lo, z = model(xb)
        logits.append(lo)
        embs.append(z)
    return torch.cat(logits).numpy(), torch.cat(embs).numpy()


print("Training...")
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    for xb, yb in batches(Xtr, ytr, 256, True):
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()
    va_acc = (infer(Xva)[0].argmax(1) == yva).mean()
    print(f"  epoch {ep + 1:2d}  loss {loss.item():.3f}  val acc {va_acc:.3f}")
print(f"Training took {time.time() - t0:.0f}s")
torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))


# ---------------- closed-set accuracy ----------------
lo_tr, z_tr = infer(Xtr)
lo_va, z_va = infer(Xva)
lo_k, z_k = infer(Xte_k)
lo_u, z_u = infer(Xte_u)
pred_k = lo_k.argmax(1)
acc_by_snr = {s: float((pred_k[ste_k == s] == yte_k[ste_k == s]).mean()) for s in SNRS}
closed_acc = float((pred_k == yte_k).mean())
print(f"\nClosed-set accuracy (known classes): {closed_acc:.3f}")


# ---------------- unknown-detection scores (higher = more unknown) ----------------
def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def score_msp(lo):
    return 1 - softmax(lo).max(1)


def score_energy(lo):
    m = lo.max(1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(lo - m).sum(1)))


# Mahalanobis: class means + shared covariance from training embeddings
means = np.stack([z_tr[ytr == c].mean(0) for c in range(len(KNOWN))])
centered = z_tr - means[ytr]
cov = centered.T @ centered / len(z_tr) + 1e-4 * np.eye(z_tr.shape[1])
prec = np.linalg.inv(cov)


def score_maha(z):
    d = np.stack([np.einsum("ij,jk,ik->i", z - m, prec, z - m) for m in means], 1)
    return d.min(1)


methods = {
    "Max softmax (baseline)": (score_msp(lo_va), score_msp(lo_k), score_msp(lo_u)),
    "Energy": (score_energy(lo_va), score_energy(lo_k), score_energy(lo_u)),
    "Mahalanobis (embedding)": (score_maha(z_va), score_maha(z_k), score_maha(z_u)),
}

labels_all = np.r_[np.zeros(len(z_k)), np.ones(len(z_u))]
snr_all = np.r_[ste_k, ste_u]
results = {"closed_set_accuracy": closed_acc, "closed_acc_by_snr": acc_by_snr, "open_set": {}}

print("\nUnknown detection:")
for name, (s_va, s_k, s_u) in methods.items():
    s_all = np.r_[s_k, s_u]
    auroc = roc_auc_score(labels_all, s_all)
    auroc_snr = {s: float(roc_auc_score(labels_all[snr_all == s], s_all[snr_all == s])) for s in SNRS}
    per_class = {
        UNKNOWN[c]: float(roc_auc_score(np.r_[np.zeros(len(s_k)), np.ones((yte_u == c).sum())],
                                        np.r_[s_k, s_u[yte_u == c]]))
        for c in range(len(UNKNOWN))
    }
    # threshold chosen on VALIDATION knowns so 95% of known signals are accepted
    thr = np.quantile(s_va, 0.95)
    known_ok = float(((s_k <= thr) & (pred_k == yte_k)).mean())
    unk_rej = float((s_u > thr).mean())
    unk_rej_hi = float((s_u[ste_u >= 6] > thr).mean())
    results["open_set"][name] = {
        "auroc": float(auroc), "auroc_by_snr": auroc_snr, "auroc_per_unknown_class": per_class,
        "known_accepted_and_correct": known_ok, "unknown_rejected": unk_rej,
        "unknown_rejected_snr>=6": unk_rej_hi,
    }
    print(f"  {name:26s} AUROC {auroc:.3f}  unknown rejected {unk_rej:.2%} "
          f"(SNR>=6: {unk_rej_hi:.2%})  per-class {', '.join(f'{k} {v:.2f}' for k, v in per_class.items())}")


# ---------------- few-shot incremental learning ----------------
def l2n(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


_, z_shot = infer(Xshot)
proto_old = l2n(np.stack([z_tr[ytr == c].mean(0) for c in range(len(KNOWN))]))
proto_new = l2n(np.stack([z_shot[yshot == c].mean(0) for c in range(len(UNKNOWN))]))

# before: nearest-class-mean with 6 prototypes (old classes only)
ncm_before = (l2n(z_k) @ proto_old.T).argmax(1)
old_before = float((ncm_before == yte_k).mean())

# after: 9 prototypes, model weights untouched
protos = np.r_[proto_old, proto_new]
z_all = np.r_[z_k, z_u]
y_all = np.r_[yte_k, yte_u + len(KNOWN)]
pred_all = (l2n(z_all) @ protos.T).argmax(1)
old_mask = y_all < len(KNOWN)
old_after = float((pred_all[old_mask] == y_all[old_mask]).mean())
new_after = float((pred_all[~old_mask] == y_all[~old_mask]).mean())
new_by_snr = {s: float((pred_all[~old_mask & (snr_all == s)] == y_all[~old_mask & (snr_all == s)]).mean())
              for s in SNRS}
old_by_snr = {s: float((pred_all[old_mask & (snr_all == s)] == y_all[old_mask & (snr_all == s)]).mean())
              for s in SNRS}
results["incremental"] = {
    "k_shot": K_SHOT, "old_class_acc_before": old_before, "old_class_acc_after": old_after,
    "new_class_acc_after": new_after, "new_acc_by_snr": new_by_snr, "old_acc_by_snr": old_by_snr,
}
print(f"\nFew-shot incremental ({K_SHOT}-shot): old classes {old_before:.3f} -> {old_after:.3f}, "
      f"new classes {new_after:.3f}")

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)


# ---------------- figures ----------------
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})

# 1. closed-set accuracy vs SNR
plt.figure(figsize=(6, 4))
plt.plot(SNRS, [acc_by_snr[s] for s in SNRS], "o-")
plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy"); plt.ylim(0, 1.02)
plt.title("Closed-set accuracy on 6 known modulations")
plt.tight_layout(); plt.savefig(f"{OUT}/1_closed_set_accuracy.png"); plt.close()

# 2. AUROC vs SNR
plt.figure(figsize=(6, 4))
for name, r in results["open_set"].items():
    plt.plot(SNRS, [r["auroc_by_snr"][s] for s in SNRS], "o-", label=f"{name} ({r['auroc']:.2f})")
plt.axhline(0.5, color="gray", ls="--", lw=1)
plt.xlabel("SNR (dB)"); plt.ylabel("AUROC (known vs unknown)"); plt.ylim(0.4, 1.02)
plt.title("Detecting unseen modulations"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/2_unknown_detection_auroc.png"); plt.close()

# 3. t-SNE of embeddings at high SNR
hi_k = np.where(ste_k >= 10)[0]
hi_u = np.where(ste_u >= 10)[0]
rng = np.random.default_rng(0)
hi_k = rng.choice(hi_k, min(1200, len(hi_k)), replace=False)
hi_u = rng.choice(hi_u, min(600, len(hi_u)), replace=False)
emb2 = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(np.r_[z_k[hi_k], z_u[hi_u]])
plt.figure(figsize=(7, 5.5))
nk = len(hi_k)
for c, name in enumerate(KNOWN):
    m = yte_k[hi_k] == c
    plt.scatter(emb2[:nk][m, 0], emb2[:nk][m, 1], s=6, label=name)
markers = ["x", "+", "^"]
for c, name in enumerate(UNKNOWN):
    m = yte_u[hi_u] == c
    plt.scatter(emb2[nk:][m, 0], emb2[nk:][m, 1], s=14, c="black", marker=markers[c], label=f"{name} (unknown)")
plt.legend(fontsize=7, markerscale=2, ncol=2); plt.xticks([]); plt.yticks([])
plt.title("t-SNE of learned embeddings (SNR >= 10 dB)")
plt.tight_layout(); plt.savefig(f"{OUT}/3_embedding_tsne.png"); plt.close()

# 4. confusion matrix at SNR >= 10
m = ste_k >= 10
cm = confusion_matrix(yte_k[m], pred_k[m], normalize="true")
plt.figure(figsize=(5.5, 4.5))
plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
plt.xticks(range(len(KNOWN)), KNOWN, rotation=45); plt.yticks(range(len(KNOWN)), KNOWN)
for i in range(len(KNOWN)):
    for j in range(len(KNOWN)):
        plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                 color="white" if cm[i, j] > 0.5 else "black")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.grid(False)
plt.title("Confusion matrix (SNR >= 10 dB)")
plt.tight_layout(); plt.savefig(f"{OUT}/4_confusion_matrix.png"); plt.close()

# 5. incremental learning
plt.figure(figsize=(6, 4))
plt.plot(SNRS, [old_by_snr[s] for s in SNRS], "o-", label="Old classes (6)")
plt.plot(SNRS, [new_by_snr[s] for s in SNRS], "s-", label=f"New classes (3, {K_SHOT}-shot)")
plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy"); plt.ylim(0, 1.02)
plt.title("Adding new classes without retraining"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/5_incremental_fewshot.png"); plt.close()

print(f"\nSaved figures and metrics to ./{OUT}/")
