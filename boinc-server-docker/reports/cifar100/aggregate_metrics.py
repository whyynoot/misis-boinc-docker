#!/usr/bin/env python3
import argparse
import json
import pathlib
import statistics
import numpy as np


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def stdev(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def load_metrics(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    if data.get("dataset") != "cifar100":
        return None
    return data


def collect_metrics(root, pattern):
    if not root.exists():
        return []
    results = []
    for path in root.rglob(pattern):
        data = load_metrics(path)
        if data:
            data["_path"] = str(path)
            results.append(data)
    return results


def eligible(run, min_epochs, min_sample_count, name_contains):
    if run.get("sample_count", 0) < min_sample_count:
        return False
    hist = run.get("train_acc_history") or []
    if len(hist) < min_epochs:
        return False
    if name_contains and name_contains not in run.get("_path", ""):
        return False
    return True


def aggregate_histories(runs, key):
    histories = [r.get(key) for r in runs if isinstance(r.get(key), list) and r.get(key)]
    if not histories:
        return None
    min_len = min(len(h) for h in histories)
    avg = []
    low = []
    high = []
    for idx in range(min_len):
        values = [h[idx] for h in histories]
        avg.append(mean(values))
        low.append(min(values))
        high.append(max(values))
    return {"avg": avg, "min": low, "max": high}


def aggregate_confusion(runs):
    matrices = [r.get("confusion_matrix") for r in runs if r.get("confusion_matrix")]
    if not matrices:
        return None
    size = len(matrices[0])
    agg = [[0 for _ in range(size)] for _ in range(size)]
    for matrix in matrices:
        for i in range(size):
            row = matrix[i]
            for j in range(size):
                agg[i][j] += row[j]
    return agg


def get_class_names(runs):
    for r in runs:
        names = r.get("class_names")
        if isinstance(names, list) and names:
            return names
    return None


def aggregate_per_class_accuracy(runs):
    per_class = [r.get("per_class_accuracy") for r in runs if r.get("per_class_accuracy")]
    if not per_class:
        return None
    size = len(per_class[0])
    avg = []
    for idx in range(size):
        values = [vec[idx] for vec in per_class]
        avg.append(mean(values))
    return avg


def write_csv(path, header, rows):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(item) for item in row))
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=pathlib.Path, default=pathlib.Path("results/cifar100"))
    parser.add_argument("--local-root", type=pathlib.Path, default=pathlib.Path("results_local"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("reports/cifar100"))
    parser.add_argument("--min-epochs", type=int, default=1, help="Drop runs with fewer epochs logged.")
    parser.add_argument("--min-sample-count", type=int, default=0, help="Drop runs with fewer training samples.")
    parser.add_argument("--name-contains", type=str, default="", help="If set, keep only runs whose file path contains the substring.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "*metrics*.json"
    all_runs = collect_metrics(args.results_root, pattern)
    distributed_runs = [
        r for r in all_runs
        if "local" not in pathlib.Path(r.get("_path", "")).name.lower()
        and eligible(r, args.min_epochs, args.min_sample_count, args.name_contains)
    ]
    local_runs = [
        r for r in collect_metrics(args.local_root, pattern)
        if "local" in pathlib.Path(r.get("_path", "")).name.lower()
        and eligible(r, args.min_epochs, args.min_sample_count, args.name_contains)
    ]

    rounds = {}
    for run in distributed_runs:
        rounds.setdefault(int(run.get("round", 0)), []).append(run)

    summary = {
        "total_distributed_runs": len(distributed_runs),
        "total_local_runs": len(local_runs),
        "filters": {
            "min_epochs": args.min_epochs,
            "min_sample_count": args.min_sample_count,
            "name_contains": args.name_contains,
        },
        "rounds": {},
    }
    for round_id, runs in sorted(rounds.items()):
        summary["rounds"][round_id] = {
            "runs": len(runs),
            "val_acc_mean": mean([r.get("val_acc") for r in runs]),
            "val_acc_std": stdev([r.get("val_acc") for r in runs]),
            "val_top5_mean": mean([r.get("val_top5") for r in runs]),
            "train_acc_mean": mean([r.get("train_acc") for r in runs]),
            "train_loss_mean": mean([r.get("train_loss") for r in runs]),
            "val_loss_mean": mean([r.get("val_loss") for r in runs]),
            "histories": {
                "train_acc": aggregate_histories(runs, "train_acc_history"),
                "val_acc": aggregate_histories(runs, "val_acc_history"),
                "val_top5": aggregate_histories(runs, "val_top5_history"),
                "train_loss": aggregate_histories(runs, "train_loss_history"),
                "val_loss": aggregate_histories(runs, "val_loss_history"),
            },
        }

    summary["local"] = {
        "runs": len(local_runs),
        "val_acc_mean": mean([r.get("val_acc") for r in local_runs]),
        "val_acc_std": stdev([r.get("val_acc") for r in local_runs]),
        "val_top5_mean": mean([r.get("val_top5") for r in local_runs]),
        "train_acc_mean": mean([r.get("train_acc") for r in local_runs]),
        "train_loss_mean": mean([r.get("train_loss") for r in local_runs]),
        "val_loss_mean": mean([r.get("val_loss") for r in local_runs]),
    }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    agg_conf = aggregate_confusion(distributed_runs)
    if agg_conf:
        (args.out_dir / "confusion_matrix.json").write_text(json.dumps(agg_conf))

    per_class = aggregate_per_class_accuracy(distributed_runs)
    class_names = get_class_names(distributed_runs)
    if per_class:
        rows = []
        for idx, acc in enumerate(per_class):
            label = class_names[idx] if class_names and idx < len(class_names) else idx
            rows.append((idx, label, acc))
        write_csv(args.out_dir / "per_class_accuracy.csv", ["class_id", "label", "accuracy"], rows)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping plots.")
        return

    if summary["rounds"]:
        round_ids = list(summary["rounds"].keys())
        val_acc = [summary["rounds"][rid]["val_acc_mean"] for rid in round_ids]
        val_top5 = [summary["rounds"][rid]["val_top5_mean"] for rid in round_ids]
        plt.figure(figsize=(8, 4))
        plt.plot(round_ids, val_acc, marker="o", label="val_acc")
        plt.plot(round_ids, val_top5, marker="o", label="val_top5")
        plt.xlabel("round (each round 4 epochs)")
        plt.ylabel("accuracy")
        plt.title("Final validation accuracy per round")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "final_val_acc.png", dpi=160)
        plt.close()

        if any(summary["rounds"][rid]["val_acc_std"] for rid in round_ids):
            plt.figure(figsize=(8, 4))
            means = [summary["rounds"][rid]["val_acc_mean"] for rid in round_ids]
            stds = [summary["rounds"][rid]["val_acc_std"] for rid in round_ids]
            plt.bar(round_ids, means, yerr=stds, capsize=5, color="#4f81bd")
            plt.xlabel("round")
            plt.ylabel("val_acc (mean ± std)")
            plt.title("Validation accuracy per round (with std)")
            plt.tight_layout()
            plt.savefig(args.out_dir / "final_val_acc_err.png", dpi=160)
            plt.close()

        local_val = summary.get("local", {}).get("val_acc_mean")
        if local_val is not None:
            plt.figure(figsize=(8, 4))
            xs = list(range(len(round_ids))) + [len(round_ids)]
            labels = [f"r{rid}" for rid in round_ids] + ["local"]
            vals = val_acc + [local_val]
            colors = ["#4f81bd"] * len(round_ids) + ["#c0504d"]
            plt.bar(xs, vals, color=colors, tick_label=labels, alpha=0.85)
            local_std = summary.get("local", {}).get("val_acc_std") or 0.0
            if local_std > 0:
                plt.errorbar([xs[-1]], [local_val], yerr=[local_std], fmt="none", ecolor="#c0504d", capsize=6)
            plt.ylabel("val_acc")
            plt.title("Distributed rounds vs local baseline")
            plt.tight_layout()
            plt.savefig(args.out_dir / "distributed_vs_local_val_acc.png", dpi=160)
            plt.close()

    if summary["rounds"]:
        first_round = summary["rounds"][sorted(summary["rounds"].keys())[0]]
        hist = first_round.get("histories", {})
        if hist.get("train_acc") and hist.get("val_acc"):
            epochs = list(range(1, len(hist["train_acc"]["avg"]) + 1))
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, hist["train_acc"]["avg"], label="train_acc")
            plt.plot(epochs, hist["val_acc"]["avg"], label="val_acc")
            plt.plot(epochs, hist["val_top5"]["avg"], label="val_top5")
            plt.xlabel("rounds (each round 4 epochs)")
            plt.ylabel("accuracy")
            plt.title("Accuracy curves (avg across clients)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.out_dir / "acc_curves.png", dpi=160)
            plt.close()

        if hist.get("train_loss") and hist.get("val_loss"):
            epochs = list(range(1, len(hist["train_loss"]["avg"]) + 1))
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, hist["train_loss"]["avg"], label="train_loss")
            plt.plot(epochs, hist["val_loss"]["avg"], label="val_loss")
            plt.xlabel("rounds (each round 4 epochs)")
            plt.ylabel("loss")
            plt.title("Loss curves (avg across clients)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.out_dir / "loss_curves.png", dpi=160)
            plt.close()

        # Overlay distributed vs local histories (MNIST-style comparison)
        if local_runs:
            local_hist_src = local_runs[0]
            l_train = local_hist_src.get("train_acc_history")
            l_val = local_hist_src.get("val_acc_history")
            l_train_loss = local_hist_src.get("train_loss_history")
            l_val_loss = local_hist_src.get("val_loss_history")
            if l_train and l_val:
                epochs = list(range(1, min(len(hist["val_acc"]["avg"]), len(l_val)) + 1))
                plt.figure(figsize=(8, 5))
                plt.plot(epochs, hist["val_acc"]["avg"][: len(epochs)], label="distributed val_acc (r1)")
                plt.plot(epochs, hist["train_acc"]["avg"][: len(epochs)], label="distributed train_acc (r1)")
                plt.plot(epochs, l_val[: len(epochs)], label="local val_acc", linestyle="--")
                plt.plot(epochs, l_train[: len(epochs)], label="local train_acc", linestyle="--")
                plt.xlabel("rounds (each round 4 epochs)")
                plt.ylabel("accuracy")
                plt.title("Accuracy curves: distributed vs local")
                plt.legend()
                plt.tight_layout()
                plt.savefig(args.out_dir / "acc_curves_compare.png", dpi=160)
                plt.close()
            if l_train_loss and l_val_loss and hist.get("train_loss") and hist.get("val_loss"):
                epochs = list(range(1, min(len(hist["train_loss"]["avg"]), len(l_train_loss)) + 1))
                plt.figure(figsize=(8, 5))
                plt.plot(epochs, hist["train_loss"]["avg"][: len(epochs)], label="distributed train_loss (r1)")
                plt.plot(epochs, hist["val_loss"]["avg"][: len(epochs)], label="distributed val_loss (r1)")
                plt.plot(epochs, l_train_loss[: len(epochs)], label="local train_loss", linestyle="--")
                plt.plot(epochs, l_val_loss[: len(epochs)], label="local val_loss", linestyle="--")
                plt.xlabel("rounds (each round 4 epochs)")
                plt.ylabel("loss")
                plt.title("Loss curves: distributed vs local")
                plt.legend()
                plt.tight_layout()
                plt.savefig(args.out_dir / "loss_curves_compare.png", dpi=160)
                plt.close()

    val_acc_all = [r.get("val_acc") for r in distributed_runs if r.get("val_acc") is not None]
    train_acc_all = [r.get("train_acc") for r in distributed_runs if r.get("train_acc") is not None]
    rounds_all = [int(r.get("round", 0)) for r in distributed_runs]
    if val_acc_all:
        plt.figure(figsize=(8, 4))
        plt.hist(val_acc_all, bins=15, color="#6fa8dc", edgecolor="black")
        plt.xlabel("val_acc")
        plt.ylabel("count")
        plt.title("Validation accuracy distribution (all runs)")
        plt.tight_layout()
        plt.savefig(args.out_dir / "val_acc_hist.png", dpi=160)
        plt.close()

    if val_acc_all and train_acc_all and len(train_acc_all) == len(val_acc_all):
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(train_acc_all, val_acc_all, c=rounds_all, cmap="viridis", alpha=0.85)
        plt.xlabel("train_acc")
        plt.ylabel("val_acc")
        plt.title("Train vs validation accuracy")
        cbar = plt.colorbar(scatter)
        cbar.set_label("round")
        plt.tight_layout()
        plt.savefig(args.out_dir / "train_vs_val_acc.png", dpi=160)
        plt.close()

    if per_class:
        indices = list(range(len(per_class)))
        ranked = sorted(zip(indices, per_class), key=lambda x: x[1], reverse=True)
        top_k = ranked[:10]
        bottom_k = ranked[-10:]
        labels_top = [class_names[i] if class_names and i < len(class_names) else str(i) for i, _ in top_k]
        labels_bot = [class_names[i] if class_names and i < len(class_names) else str(i) for i, _ in bottom_k]

        plt.figure(figsize=(10, 4))
        plt.bar(range(len(top_k)), [v for _, v in top_k], tick_label=labels_top, color="#82b366")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("accuracy")
        plt.title("Top 10 classes by accuracy")
        plt.tight_layout()
        plt.savefig(args.out_dir / "per_class_top10.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.bar(range(len(bottom_k)), [v for _, v in bottom_k], tick_label=labels_bot, color="#d46a6a")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("accuracy")
        plt.title("Bottom 10 classes by accuracy")
        plt.tight_layout()
        plt.savefig(args.out_dir / "per_class_bottom10.png", dpi=160)
        plt.close()

    if agg_conf is not None:
        cm = np.array(agg_conf, dtype=float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, np.maximum(row_sum, 1.0))
        plt.figure(figsize=(7, 6))
        plt.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(label="normalized count")
        plt.title("Aggregated confusion matrix (normalized rows)")
        plt.xlabel("predicted class")
        plt.ylabel("true class")
        plt.tight_layout()
        plt.savefig(args.out_dir / "confusion_matrix_heatmap.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
