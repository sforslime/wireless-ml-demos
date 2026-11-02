"""
Trustworthy spectrum-occupancy forecasting demo.

Forecast the Spectrum Occupancy Ratio (SOR) 1 hour ahead and attach a
90% prediction interval with a distribution-free coverage guarantee.

Compared:
  - point forecasts: persistence, seasonal-naive, MLP (median)
  - intervals: raw quantile regression, split conformal, conformalized
    quantile regression (CQR)
  - under a traffic surge (distribution shift): static CQR vs
    adaptive conformal inference (ACI) with a rolling calibration window

Run:  python run.py        (about 3-5 min on a laptop CPU)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from spectrum import SLOTS_PER_DAY, simulate

torch.manual_seed(0)
np.random.seed(0)

N_DAYS = 70
TRAIN_END, CAL_END = 42, 56          # days: train 0-41, calibrate 42-55, test 56-69
SURGE_DAY = 63                       # surge scenario: traffic jumps from day 63
LOOKBACK = SLOTS_PER_DAY             # use the past 24 h
HORIZON = 4                          # forecast 4 slots = 1 hour ahead
ALPHA = 0.10                         # target 90% coverage
QUANTILES = [ALPHA / 2, 0.5, 1 - ALPHA / 2]
OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ---------------- data ----------------
print("Simulating spectrum occupancy...")
sor = simulate(N_DAYS, seed=7)
sor_surge = simulate(N_DAYS, surge_from_day=SURGE_DAY, seed=7)   # same randomness, plus surge


def make_xy(series):
    """Features: last 24h of SOR + time-of-day and weekend flag of the target slot."""
    X, y, t_idx = [], [], []
    for t in range(LOOKBACK, len(series) - HORIZON + 1):
        target = t + HORIZON - 1
        day, slot = divmod(target, SLOTS_PER_DAY)
        tod = 2 * np.pi * slot / SLOTS_PER_DAY
        extra = [np.sin(tod), np.cos(tod), float(day % 7 in (5, 6))]
        X.append(np.r_[series[t - LOOKBACK:t], extra])
        y.append(series[target])
        t_idx.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(t_idx)


X, y, tix = make_xy(sor)
Xs, ys, _ = make_xy(sor_surge)
day_of = tix // SLOTS_PER_DAY
tr, ca, te = day_of < TRAIN_END, (day_of >= TRAIN_END) & (day_of < CAL_END), day_of >= CAL_END
print(f"train {tr.sum()}, calibration {ca.sum()}, test {te.sum()} samples")


# ---------------- quantile-regression MLP ----------------
model = nn.Sequential(
    nn.Linear(X.shape[1], 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, len(QUANTILES)),
)
qs = torch.tensor(QUANTILES)


def pinball(pred, target):
    err = target[:, None] - pred
    return torch.mean(torch.maximum(qs * err, (qs - 1) * err))


opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
Xtr_t, ytr_t = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
print("Training...")
for ep in range(150):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), 128):
        j = perm[i:i + 128]
        loss = pinball(model(Xtr_t[j]), ytr_t[j])
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 30 == 0:
        print(f"  epoch {ep + 1:3d}  pinball loss {loss.item():.4f}")


@torch.no_grad()
def predict(Xa):
    model.eval()
    p = model(torch.from_numpy(Xa)).numpy()
    p.sort(axis=1)                       # avoid quantile crossing
    return p[:, 0], p[:, 1], p[:, 2]


lo, med, hi = predict(X)
lo_s, med_s, hi_s = predict(Xs)


# ---------------- point forecast accuracy ----------------
persist = X[:, LOOKBACK - 1]                                   # last observed value
seasonal = np.array([sor[t - SLOTS_PER_DAY] for t in tix])     # same slot yesterday
mae = {
    "Persistence (last value)": float(np.abs(persist[te] - y[te]).mean()),
    "Seasonal naive (yesterday)": float(np.abs(seasonal[te] - y[te]).mean()),
    "MLP median": float(np.abs(med[te] - y[te]).mean()),
}
print("\nTest MAE:", {k: round(v, 4) for k, v in mae.items()})


# ---------------- intervals ----------------
def conformal_q(scores, alpha):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, method="higher")


def summarize(l, h, yy):
    return {"coverage": float(((yy >= l) & (yy <= h)).mean()), "mean_width": float((h - l).mean())}


# split conformal around the median (constant width)
q_split = conformal_q(np.abs(y[ca] - med[ca]), ALPHA)
# conformalized quantile regression (adaptive width)
cqr_scores = np.maximum(lo[ca] - y[ca], y[ca] - hi[ca])
q_cqr = conformal_q(cqr_scores, ALPHA)

intervals = {
    "Raw quantile regression": (lo, hi),
    "Split conformal": (med - q_split, med + q_split),
    "CQR (conformalized quantiles)": (lo - q_cqr, hi + q_cqr),
}
interval_results = {k: summarize(l[te], h[te], y[te]) for k, (l, h) in intervals.items()}
print("\nTest intervals (target 90%):")
for k, v in interval_results.items():
    print(f"  {k:30s} coverage {v['coverage']:.3f}  mean width {v['mean_width']:.4f}")

# calibration curve: does empirical coverage track the target for every alpha?
alphas = np.linspace(0.02, 0.4, 12)
cal_curve = {"nominal": (1 - alphas).tolist(), "split": [], "cqr": []}
for a in alphas:
    qa = conformal_q(np.abs(y[ca] - med[ca]), a)
    cal_curve["split"].append(summarize(med[te] - qa, med[te] + qa, y[te])["coverage"])
    # CQR re-uses the 90% quantile heads; conformal step adjusts them to level a
    qa = conformal_q(cqr_scores, a)
    cal_curve["cqr"].append(summarize(lo[te] - qa, hi[te] + qa, y[te])["coverage"])


# ---------------- distribution shift: traffic surge ----------------
test_idx = np.where(te)[0]
surge_mask = day_of[test_idx] >= SURGE_DAY

# static CQR: calibration fixed, threshold fixed
cover_static = (ys[test_idx] >= lo_s[test_idx] - q_cqr) & (ys[test_idx] <= hi_s[test_idx] + q_cqr)

# ACI (Gibbs & Candes, 2021) + rolling calibration window
GAMMA, WINDOW = 0.02, 14 * SLOTS_PER_DAY
window = list(cqr_scores[-WINDOW:])
alpha_t = ALPHA
cover_aci, widths_aci, alpha_trace = [], [], []
for i in test_idx:
    a = float(np.clip(alpha_t, 1e-3, 0.999))
    qt = conformal_q(np.array(window), a)
    covered = lo_s[i] - qt <= ys[i] <= hi_s[i] + qt
    cover_aci.append(covered)
    widths_aci.append(hi_s[i] - lo_s[i] + 2 * qt)
    alpha_trace.append(alpha_t)
    alpha_t += GAMMA * (ALPHA - (0.0 if covered else 1.0))          # miss -> widen, hit -> tighten
    window.append(max(lo_s[i] - ys[i], ys[i] - hi_s[i]))            # score revealed after observing y
    window = window[-WINDOW:]
cover_aci = np.array(cover_aci)

shift_results = {
    "static_cqr_before_surge": float(cover_static[~surge_mask].mean()),
    "static_cqr_during_surge": float(cover_static[surge_mask].mean()),
    "aci_before_surge": float(cover_aci[~surge_mask].mean()),
    "aci_during_surge": float(cover_aci[surge_mask].mean()),
}
print("\nTraffic surge (coverage):", {k: round(v, 3) for k, v in shift_results.items()})

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump({"test_mae": mae, "intervals_90": interval_results,
               "calibration_curve": cal_curve, "surge": shift_results}, f, indent=2)


# ---------------- figures ----------------
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
hours = lambda idx: (tix[idx] - tix[idx][0]) / 4

# 1. forecast with CQR band over 3 test days
w = test_idx[:3 * SLOTS_PER_DAY]
l_c, h_c = intervals["CQR (conformalized quantiles)"]
plt.figure(figsize=(9, 3.8))
plt.fill_between(hours(w), l_c[w], h_c[w], alpha=0.3, label="90% CQR interval")
plt.plot(hours(w), y[w], "k", lw=1, label="Actual SOR")
plt.plot(hours(w), med[w], "C0", lw=1.3, label="Forecast (1 h ahead)")
plt.xlabel("Hours into test period"); plt.ylabel("Spectrum occupancy ratio")
plt.title("1-hour-ahead SOR forecast with conformal interval"); plt.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/1_forecast_with_interval.png"); plt.close()

# 2. coverage and width per method
names = list(interval_results)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].bar(range(3), [interval_results[n]["coverage"] for n in names], color=["C3", "C1", "C2"])
ax[0].axhline(1 - ALPHA, color="k", ls="--", lw=1, label="target 90%")
ax[0].set_ylim(0.6, 1); ax[0].set_title("Test coverage"); ax[0].legend(fontsize=8)
ax[1].bar(range(3), [interval_results[n]["mean_width"] for n in names], color=["C3", "C1", "C2"])
ax[1].set_title("Mean interval width (smaller = sharper)")
for a in ax:
    a.set_xticks(range(3)); a.set_xticklabels(["Raw QR", "Split conf.", "CQR"])
plt.tight_layout(); plt.savefig(f"{OUT}/2_coverage_vs_width.png"); plt.close()

# 3. calibration curve
plt.figure(figsize=(4.8, 4.5))
plt.plot(cal_curve["nominal"], cal_curve["nominal"], "k--", lw=1, label="perfect")
plt.plot(cal_curve["nominal"], cal_curve["split"], "o-", label="Split conformal")
plt.plot(cal_curve["nominal"], cal_curve["cqr"], "s-", label="CQR")
plt.xlabel("Target coverage"); plt.ylabel("Empirical test coverage")
plt.title("Coverage guarantee holds at every level"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/3_calibration_curve.png"); plt.close()

# 4. surge: rolling 1-day coverage
roll = lambda c: np.convolve(c.astype(float), np.ones(SLOTS_PER_DAY) / SLOTS_PER_DAY, mode="valid")
d_axis = CAL_END + (np.arange(len(roll(cover_static))) + SLOTS_PER_DAY) / SLOTS_PER_DAY
plt.figure(figsize=(8, 3.8))
plt.axvspan(SURGE_DAY, N_DAYS, color="red", alpha=0.08, label="traffic surge")
plt.plot(d_axis, roll(cover_static), label="Static CQR")
plt.plot(d_axis, roll(cover_aci), label="Adaptive conformal (ACI)")
plt.axhline(1 - ALPHA, color="k", ls="--", lw=1, label="target 90%")
plt.xlabel("Day"); plt.ylabel("Rolling 24 h coverage"); plt.ylim(0, 1.02)
plt.title("Staying reliable under distribution shift"); plt.legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(f"{OUT}/4_surge_adaptive_coverage.png"); plt.close()

print(f"\nSaved figures and metrics to ./{OUT}/")
