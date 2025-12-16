#!/usr/bin/env python3
import json
import pathlib
import urllib.request
import numpy as np
import socket
import time


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def download_mnist(dest_dir: pathlib.Path) -> pathlib.Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "mnist.npz"
    if not dest.exists():
        urllib.request.urlretrieve(MNIST_URL, dest)
    return dest


def run(job_file: pathlib.Path, output_file: pathlib.Path):
    job = json.loads(job_file.read_text())
    sample_count = int(job.get("sample_count", 5000))
    seed = int(job.get("seed", 42))
    round_id = int(job.get("round", 1))
    client_label = str(job.get("client_label", "")) or socket.gethostname()
    train_epochs = int(job.get("train_epochs", 5))
    batch_size = int(job.get("batch_size", 256))
    learning_rate = float(job.get("learning_rate", 0.1))

    dataset_path = download_mnist(job_file.parent)
    data = np.load(dataset_path)
    x_train = data["x_train"]
    y_train = data["y_train"]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_train), size=sample_count, replace=False)
    x = x_train[idx].astype("float32") / 255.0
    y = y_train[idx]

    # split into train/val
    split = int(0.8 * sample_count)
    x_train_s, y_train_s = x[:split], y[:split]
    x_val_s, y_val_s = x[split:], y[split:]

    # flatten
    x_train_s = x_train_s.reshape((x_train_s.shape[0], -1))
    x_val_s = x_val_s.reshape((x_val_s.shape[0], -1))

    # initialize weights
    num_classes = 10
    num_features = x_train_s.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.01, size=(num_features, num_classes))
    b = np.zeros((num_classes,), dtype=np.float32)

    def softmax(z):
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def one_hot(labels):
        oh = np.zeros((labels.size, num_classes), dtype=np.float32)
        oh[np.arange(labels.size), labels] = 1.0
        return oh

    def accuracy(xd, yd):
        probs = softmax(xd @ W + b)
        preds = probs.argmax(axis=1)
        return float((preds == yd).mean())

    y_train_oh = one_hot(y_train_s)

    start = time.time()
    n_batches = max(1, int(np.ceil(x_train_s.shape[0] / batch_size)))
    for epoch in range(train_epochs):
        perm = rng.permutation(x_train_s.shape[0])
        x_train_s = x_train_s[perm]
        y_train_s = y_train_s[perm]
        y_train_oh = y_train_oh[perm]
        for bidx in range(n_batches):
            start_idx = bidx * batch_size
            end_idx = min((bidx + 1) * batch_size, x_train_s.shape[0])
            xb = x_train_s[start_idx:end_idx]
            yb = y_train_s[start_idx:end_idx]
            yb_oh = y_train_oh[start_idx:end_idx]
            logits = xb @ W + b
            probs = softmax(logits)
            grad_logits = (probs - yb_oh) / xb.shape[0]
            grad_W = xb.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b
    runtime_sec = time.time() - start

    train_acc = accuracy(x_train_s, y_train_s)
    val_acc = accuracy(x_val_s, y_val_s)

    output = {
        "sample_count": sample_count,
        "seed": seed,
        "round": round_id,
        "client": client_label,
        "train_epochs": train_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "runtime_sec": runtime_sec,
    }
    output_file.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    import sys

    job_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "job.json")
    out_path = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "metrics.json")
    run(job_path, out_path)
