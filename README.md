# Wireless AI Demos

Two small research demos on machine learning for wireless systems, built in Python and PyTorch. Each runs on a laptop CPU in a few minutes with no dataset download.

| Demo | Question | Headline result |
|---|---|---|
| [01 – Open-set modulation recognition](01-open-set-signals/) | Can a classifier recognize known signal types *and* flag never-seen ones? | Embedding-distance detection reaches **0.85 AUROC**, vs 0.58 for softmax confidence. Two of the three unseen types are detected at **≥0.96 AUROC**. |
| [02 – Trustworthy spectrum forecasting](02-spectrum-forecast/) | Can we forecast spectrum occupancy with intervals that are actually reliable? | Conformal calibration lifts coverage from **77.5% to 88%**. Adaptive conformal inference holds **~90% coverage** through a traffic surge, where static intervals fall to 68.5%. |

## Run

```bash
pip install -r requirements.txt
cd 01-open-set-signals && python run.py
cd ../02-spectrum-forecast && python run.py
```

Results (figures and `metrics.json`) are written to each demo's `results/` folder. All data is simulated, so runs are fully reproducible.
