# CIFAR-100 distributed training over BOINC

## Overview
- **Services**: `mysql`, `apache` (BOINC server + daemons), ten clients `client1..client10` on the same image.
- **App**: BOINC app `cifar100` (version 1.0) based on PyTorch + torchvision; trains ConvNeXt-Tiny on CIFAR-100.
- **Templates**: `templates/cifar100_in` (one `job.json` input), `templates/cifar100_out` (one `metrics.json` output).
- **Pipeline**: `run-cifar100.ps1` -> `bin/create_work` -> clients compute -> validator/assimilator -> `/results/cifar100/<wu>/metrics_<result_id>.json`.

## Dataset and model
- **Dataset**: CIFAR-100 (50k train, 10k test), normalized with standard CIFAR-100 mean/std.
- **Transforms**: random crop + horizontal flip for train, deterministic for validation.
- **Model**: `convnext_tiny` with optional ImageNet weights; classifier head replaced for 100 classes.

## Training defaults (run-cifar100.ps1)
- `sample_count=50000`, `val_count=10000`
- `train_epochs=10`, `batch_size=128`
- `learning_rate=5e-4`, `weight_decay=0.05`, `label_smoothing=0.0`
- `input_size=32`
- `use_pretrained=true`, `freeze_backbone=false`
- `lr_schedule=cosine`
- `num_workers=2`, `num_threads=2`

## Distributed setup
- **Clients**: 10 services in `docker-compose.yml`.
- **Workunit fanout**: `target_nresults=10` to keep all clients busy.
- **Determinism vs diversity**: each client gets a unique `client_seed` derived from base seed + hostname hash.
  This avoids identical metrics between local and distributed runs while keeping the same job payload.

## Validation and metrics
Each client writes:
- `train_acc`, `val_acc`, `val_top5`, `train_loss`, `val_loss`
- per-epoch histories
- optional `confusion_matrix` and `per_class_accuracy` on the validation set

Aggregated metrics and plots can be generated via:
```bash
python reports/cifar100/aggregate_metrics.py \
  --results-root results_local/cifar100 \
  --local-root results_local \
  --out-dir reports/cifar100 \
  --min-epochs 4 \
  --min-sample-count 5000
```
Outputs include `summary.json`, `per_class_accuracy.csv`, and (if matplotlib is installed)
`acc_curves.png`, `loss_curves.png`, `final_val_acc.png`, `distributed_vs_local_val_acc.png`,
`train_vs_val_acc.png`, `val_acc_hist.png`. Setting `compute_confusion=true` in jobs will also
produce per-class plots and a confusion heatmap.

### Current results (highgpu run, 20k/5k samples, 4 epochs, batch 32, pretrained)
- Distributed rounds (client1 GPU): val_acc round1–4 = 0.688 / 0.680 / 0.683 / 0.679; val_top5 ≈ 0.92.
- Local baseline (same hyperparams on client1): val_acc 0.659, val_top5 0.915, train_acc 0.831.
- Plots in `reports/cifar100`:
  - `acc_curves.png` / `loss_curves.png`: per-epoch curves (round 1).
  - `final_val_acc.png`: val_acc per round.
  - `distributed_vs_local_val_acc.png`: distributed rounds vs local baseline bar chart.
  - `val_acc_hist.png` / `train_vs_val_acc.png`: distribution and scatter across runs.

## How to run
1) Register the app and templates (`CIFAR100.md`).
2) Submit jobs:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run-cifar100.ps1 `
     -Rounds 3 -TargetResults 10 -MinQuorum 2
   ```
3) Inspect results under `results/cifar100/<wu>/metrics_<result_id>.json`.
4) Generate aggregated metrics with the script above.

## Notes and troubleshooting
- If you update the app code, rerun `bin/update_versions --appname cifar100` and reset clients.
- If metrics look identical across clients, set `use_client_hash=true` (default) or override
  `client_seed_offset` per job to force different data subsets.
