#!/usr/bin/env python3
import hashlib
import json
import pathlib
import random
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, models, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def stable_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(input_size: int, train: bool) -> transforms.Compose:
    ops = []
    if input_size != 32:
        ops.append(transforms.Resize(input_size))
    if train:
        pad = max(2, int(round(input_size * 0.125)))
        ops.extend(
            [
                transforms.RandomCrop(input_size, padding=pad),
                transforms.RandomHorizontalFlip(),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return transforms.Compose(ops)


def build_model(num_classes: int, use_pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = None
    if use_pretrained:
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")
    return model


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    return int(correct.sum().item())


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, compute_confusion=False):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    total_loss = 0.0
    confusion = None
    if compute_confusion:
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_size = labels.size(0)
        total += batch_size
        total_loss += loss.item() * batch_size
        correct1 += accuracy_topk(outputs, labels, 1)
        correct5 += accuracy_topk(outputs, labels, 5)
        if confusion is not None:
            preds = outputs.argmax(dim=1)
            for tgt, pred in zip(labels.view(-1), preds.view(-1)):
                confusion[int(tgt), int(pred)] += 1
    avg_loss = total_loss / max(1, total)
    acc1 = correct1 / max(1, total)
    acc5 = correct5 / max(1, total)
    return avg_loss, acc1, acc5, confusion


def run(job_file: pathlib.Path, output_file: pathlib.Path):
    job = json.loads(job_file.read_text())
    sample_count = int(job.get("sample_count", 50000))
    val_count = int(job.get("val_count", 10000))
    base_seed = int(job.get("seed", 42))
    round_id = int(job.get("round", 1))
    client_label = str(job.get("client_label", "")) or socket.gethostname()
    use_client_hash = bool(job.get("use_client_hash", True))
    client_seed_offset = int(job.get("client_seed_offset", 0))
    train_epochs = int(job.get("train_epochs", 10))
    batch_size = int(job.get("batch_size", 128))
    learning_rate = float(job.get("learning_rate", 5e-4))
    weight_decay = float(job.get("weight_decay", 0.05))
    label_smoothing = float(job.get("label_smoothing", 0.0))
    input_size = int(job.get("input_size", 32))
    num_workers = int(job.get("num_workers", 2))
    num_threads = int(job.get("num_threads", 2))
    use_pretrained = bool(job.get("use_pretrained", True))
    freeze_backbone = bool(job.get("freeze_backbone", False))
    lr_schedule = str(job.get("lr_schedule", "cosine")).lower()
    compute_confusion = bool(job.get("compute_confusion", True))
    force_cpu = bool(job.get("force_cpu", False))

    client_hash = stable_hash(client_label) if use_client_hash else 0
    client_seed = base_seed + client_seed_offset + (client_hash % 10000)
    set_seed(client_seed)

    if num_threads > 0:
        torch.set_num_threads(num_threads)

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")

    log(
        f"Starting CIFAR-100 run on {device} | samples train/val={sample_count}/{val_count} | "
        f"epochs={train_epochs} batch={batch_size} lr={learning_rate} wd={weight_decay} "
        f"pretrained={use_pretrained} freeze_backbone={freeze_backbone} seed={client_seed}"
    )

    data_root = job_file.parent / "cifar100"
    train_tf = build_transforms(input_size, train=True)
    val_tf = build_transforms(input_size, train=False)
    train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)
    val_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=val_tf)

    rng = np.random.default_rng(client_seed)
    if sample_count < len(train_dataset):
        train_idx = rng.choice(len(train_dataset), size=sample_count, replace=False)
        train_dataset = Subset(train_dataset, train_idx)
    if val_count < len(val_dataset):
        val_idx = rng.choice(len(val_dataset), size=val_count, replace=False)
        val_dataset = Subset(val_dataset, val_idx)

    generator = torch.Generator()
    generator.manual_seed(client_seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if device.type == "cuda" and num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(num_classes=100, use_pretrained=use_pretrained, freeze_backbone=freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    log(f"Model ready: ConvNeXt-Tiny, parameters={sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, train_epochs))
    elif lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, train_epochs // 3), gamma=0.5)

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    val_top5_hist = []
    lr_hist = []
    epoch_times = []

    start = time.time()
    for epoch in range(train_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        log(f"Epoch {epoch + 1}/{train_epochs} start | lr={optimizer.param_groups[0]['lr']:.3e}")
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += accuracy_topk(outputs, labels, 1)
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc, val_top5, confusion = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=100,
            compute_confusion=(compute_confusion and epoch == train_epochs - 1),
        )
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        val_top5_hist.append(val_top5)
        lr_hist.append(optimizer.param_groups[0]["lr"])
        epoch_times.append(time.time() - epoch_start)
        if scheduler is not None:
            scheduler.step()
        log(
            f"Epoch {epoch + 1} done in {epoch_times[-1]:.1f}s | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_top1={val_acc:.3f} val_top5={val_top5:.3f}"
        )

    runtime_sec = time.time() - start
    conf_matrix = None
    per_class_acc = None
    if compute_confusion and confusion is not None:
        conf_matrix = confusion.cpu().tolist()
        conf_diag = confusion.diag().float()
        conf_sum = confusion.sum(dim=1).clamp(min=1).float()
        per_class_acc = (conf_diag / conf_sum).cpu().tolist()

    class_names = None
    if hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "classes"):
        class_names = train_dataset.dataset.classes
    elif hasattr(train_dataset, "classes"):
        class_names = train_dataset.classes

    output = {
        "dataset": "cifar100",
        "model": "convnext_tiny",
        "sample_count": int(len(train_dataset)),
        "val_count": int(len(val_dataset)),
        "seed": base_seed,
        "client_seed": client_seed,
        "client_hash": client_hash,
        "round": round_id,
        "client": client_label,
        "train_epochs": train_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "input_size": input_size,
        "use_pretrained": use_pretrained,
        "freeze_backbone": freeze_backbone,
        "lr_schedule": lr_schedule,
        "num_workers": num_workers,
        "num_threads": num_threads,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "train_acc": train_acc_hist[-1] if train_acc_hist else None,
        "val_acc": val_acc_hist[-1] if val_acc_hist else None,
        "val_top5": val_top5_hist[-1] if val_top5_hist else None,
        "train_loss": train_loss_hist[-1] if train_loss_hist else None,
        "val_loss": val_loss_hist[-1] if val_loss_hist else None,
        "train_acc_history": train_acc_hist,
        "val_acc_history": val_acc_hist,
        "val_top5_history": val_top5_hist,
        "train_loss_history": train_loss_hist,
        "val_loss_history": val_loss_hist,
        "lr_history": lr_hist,
        "epoch_times_sec": epoch_times,
        "runtime_sec": runtime_sec,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": conf_matrix,
        "class_names": class_names,
    }
    log(
        f"Finished in {runtime_sec/60:.1f}m | "
        f"final train_acc={output['train_acc']:.3f} val_top1={output['val_acc']:.3f} val_top5={output['val_top5']:.3f}"
    )
    output_file.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    import sys

    job_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "job.json")
    out_path = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "metrics.json")
    run(job_path, out_path)
