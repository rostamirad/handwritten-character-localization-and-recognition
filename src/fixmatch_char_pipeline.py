from __future__ import annotations
import os, re, json, argparse, random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import Levenshtein
except Exception:
    Levenshtein = None

from sklearn.metrics import confusion_matrix, classification_report

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


STRICT_DETERMINISM = True      
DISABLE_AMP        = True      
DATALOADER_WORKERS = 0         


def get_out_dirs(script_stem: str, out_root: str = "results") -> Dict[str, Path]:
    base = Path(out_root) / script_stem
    dirs = {
        "base": base,
        "crops": base / "char_crops",
        "models": base / "models",
        "eval": base / "eval",
        "figs": base / "figs",
        "infer": base / "infer",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_split_paths(root_dir: str, split: str) -> Tuple[str, str]:
    """Returns (images_dir, labels_dir) for a given split. For validation, tries 'val' then 'valid'."""
    candidates = [split]
    if split == "val":
        candidates = ["val", "valid"]
    for name in candidates:
        img_dir = os.path.join(root_dir, name, "images")
        lbl_dir = os.path.join(root_dir, name, "labels")
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            return img_dir, lbl_dir
    raise FileNotFoundError(
        f"Could not find split directories for '{split}'. "
        f"Tried: {', '.join(candidates)} under {root_dir}."
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int = 42) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if STRICT_DETERMINISM:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    g = torch.Generator()
    g.manual_seed(seed)
    print(f"[Determinism] seed={seed}, strict={STRICT_DETERMINISM}, workers={DATALOADER_WORKERS}, amp={'off' if DISABLE_AMP else 'on'}")
    return g

def safe_char(ch: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", ch)

def fit_on_canvas(img: Image.Image, target: int = 256, fill: int = 255) -> Image.Image:
    w, h = img.size
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target, target), color=(fill, fill, fill))
    off_x = (target - new_w) // 2
    off_y = (target - new_h) // 2
    canvas.paste(img_resized, (off_x, off_y))
    return canvas

def save_fig(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class PadToSquare:
    """Pad a PIL image to a square shape with given fill."""
    def __init__(self, fill=0):
        self.fill = fill
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_side = max(w, h)
        pl = (max_side - w) // 2
        pt = (max_side - h) // 2
        pr = max_side - w - pl
        pb = max_side - h - pt
        return ImageOps.expand(img, (pl, pt, pr, pb), self.fill)

def get_transforms(img_size: int = 32):
    weak = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    strong = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandAugment(num_ops=2, magnitude=10),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return weak, strong


def scan_train_split(train_img_dir: str, train_lbl_dir: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    labeled, unlabeled = [], []
    for label_file in tqdm(sorted(os.listdir(train_lbl_dir)), desc="Analyzing labels"):  
        if not label_file.endswith(".json"):
            continue
        label_path = os.path.join(train_lbl_dir, label_file)
        with open(label_path, "r") as f:
            label_data = json.load(f)
        expr = label_data.get("expression", "").strip()
        base = label_file[:-5]
        candidate_imgs = [f"{base}.png", f"{base}.jpg", f"{base}.jpeg"]
        image_path = None
        for c in candidate_imgs:
            cp = os.path.join(train_img_dir, c)
            if os.path.exists(cp):
                image_path = cp; break
        if image_path:
            if expr:
                labeled.append((image_path, expr))
            else:
                unlabeled.append(image_path)
    return labeled, unlabeled

def crop_all_chars(labeled_samples: List[Tuple[str, str]],
                   unlabeled_samples: List[str],
                   crop_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    labeled_info, unlabeled_info = [], []

    for img_path, expr in tqdm(labeled_samples, desc="Cropping Labeled Chars"):
        label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".json"
        image = Image.open(img_path)
        with open(label_path, "r") as f:
            label_data = json.load(f)
        boxes = sorted(label_data["annotations"], key=lambda b: b["boundingBox"]["x"])
        if len(boxes) != len(expr):
            continue
        for idx, (ann, char_label) in enumerate(zip(boxes, expr)):
            bb = ann["boundingBox"]
            x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"])
            crop = image.crop((x, y, x + w, y + h))
            crop_filename = f"labeled_{Path(img_path).stem}_{idx}_{safe_char(char_label)}.png"
            crop_path = crop_dir / crop_filename
            crop.save(crop_path)
            labeled_info.append({"path": str(crop_path), "label": char_label})

    for img_path in tqdm(unlabeled_samples, desc="Cropping Unlabeled Chars"):
        label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".json"
        image = Image.open(img_path)
        with open(label_path, "r") as f:
            label_data = json.load(f)
        boxes = sorted(label_data["annotations"], key=lambda b: b["boundingBox"]["x"])
        for idx, ann in enumerate(boxes):
            bb = ann["boundingBox"]
            x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"])
            crop = image.crop((x, y, x + w, y + h))
            crop_filename = f"unlabeled_{Path(img_path).stem}_{idx}.png"
            crop_path = crop_dir / crop_filename
            crop.save(crop_path)
            unlabeled_info.append({"path": str(crop_path), "label": -1})

    return labeled_info, unlabeled_info

class FixMatchDataset(Dataset):
    def __init__(self, labeled_info, unlabeled_info, weak_transform, strong_transform):
        self.labeled_info = labeled_info
        self.unlabeled_info = unlabeled_info
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.label_set = sorted(list(set(info["label"] for info in self.labeled_info)))
        self.label2idx = {c: i for i, c in enumerate(self.label_set)}
        self.idx2label = {i: c for c, i in self.label2idx.items()}

    def __len__(self):
        return max(len(self.labeled_info), len(self.unlabeled_info))

    def __getitem__(self, idx):
        labeled_idx = idx % len(self.labeled_info)
        l_info = self.labeled_info[labeled_idx]
        l_img = Image.open(l_info["path"]).convert("L")
        l_label = self.label2idx[l_info["label"]]

        unlabeled_idx = idx % len(self.unlabeled_info)
        u_info = self.unlabeled_info[unlabeled_idx]
        u_img = Image.open(u_info["path"]).convert("L")

        return {
            "labeled_img": self.weak_transform(l_img),
            "label": torch.tensor(l_label, dtype=torch.long),
            "unlabeled_img_weak": self.weak_transform(u_img),
            "unlabeled_img_strong": self.strong_transform(u_img),
        }


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x


def train_fixmatch_across_thresholds(
    train_loader: DataLoader,
    num_classes: int,
    thresholds: List[float],
    out_models: Path,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    lambda_u: float = 1.0,
) -> List[Path]:
    out_models.mkdir(parents=True, exist_ok=True)
    saved = []

    for thr in thresholds:
        print(f"\n===== START FixMatch TRAINING @ threshold={thr} =====")
        model = ImprovedCNN(num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            total_loss_l = total_loss_u = total_mask = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} | Thresh={thr}")
            for batch in pbar:
                labeled_imgs = batch["labeled_img"].to(device)
                labels = batch["label"].to(device)
                unlabeled_weak = batch["unlabeled_img_weak"].to(device)
                unlabeled_strong = batch["unlabeled_img_strong"].to(device)

                labeled_outputs = model(labeled_imgs)
                loss_l = F.cross_entropy(labeled_outputs, labels, reduction="mean")

                with torch.no_grad():
                    weak_outputs = model(unlabeled_weak)
                    probs = torch.softmax(weak_outputs, dim=1)
                    max_probs, pseudo_labels = torch.max(probs, dim=1)
                    mask = max_probs.ge(thr).float()

                strong_outputs = model(unlabeled_strong)
                loss_u = (F.cross_entropy(strong_outputs, pseudo_labels, reduction="none") * mask).mean()

                total_loss = loss_l + lambda_u * loss_u
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss_l += loss_l.item()
                total_loss_u += loss_u.item()
                total_mask += mask.mean().item()

            avg_loss_l = total_loss_l / len(train_loader)
            avg_loss_u = total_loss_u / len(train_loader)
            avg_mask = total_mask / len(train_loader)
            print(f"Epoch {epoch+1} | L-Loss: {avg_loss_l:.4f} | U-Loss: {avg_loss_u:.4f} | Mask: {avg_mask:.2%}")

        model_path = out_models / f"fixmatch_model_thresh_{int(thr*100)}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved: {model_path}")
        saved.append(model_path)

    return saved


def load_validation(val_img_dir: str, val_lbl_dir: str):
    val_records = []
    val_files = sorted([f for f in os.listdir(val_lbl_dir) if f.endswith(".json")])  
    for label_file in tqdm(val_files, desc="Loading Validation Samples"):
        label_path = os.path.join(val_lbl_dir, label_file)
        img_base = label_file[:-5]
        for ext in (".png", ".jpg", ".jpeg"):
            img_path = os.path.join(val_img_dir, img_base + ext)
            if os.path.exists(img_path): break
        else:
            continue
        with open(label_path, "r") as f:
            label_data = json.load(f)
        true_expression = label_data.get("expression", "").strip()
        if not true_expression:
            continue
        boxes = sorted(label_data["annotations"], key=lambda b: b["boundingBox"]["x"])
        if len(boxes) != len(true_expression):
            continue
        val_records.append({
            "img_path": img_path,
            "true_expression": true_expression,
            "boxes": [v["boundingBox"] for v in boxes],
        })
    return val_records

def evaluate_threshold_models(
    model_paths: List[Path],
    num_classes: int,
    val_records: List[Dict],
    weak_transform,
    idx2label: Dict[int, str],
    label2idx: Dict[str, int],
    device: torch.device,
) -> Tuple[Optional[float], Dict[float, Dict[str, float]], List[int], List[int]]:
    if Levenshtein is None:
        raise ImportError("Levenshtein is required for validation similarity. Please install it.")

    results: Dict[float, Dict[str, float]] = {}
    best_threshold = None
    best_similarity = -1.0
    best_model_preds: List[int] = []
    best_model_labels: List[int] = []

    def parse_thr(p: Path) -> float:
        try:
            return int(p.stem.split("_")[-1]) / 100.0
        except Exception:
            return -1.0

    for mp in sorted(model_paths, key=parse_thr):
        thr = parse_thr(mp)
        if thr < 0:
            continue
        print(f"--- Evaluating {mp.name} (thr={thr}) ---")

        model = ImprovedCNN(num_classes).to(device)
        model.load_state_dict(torch.load(mp, map_location=device))
        model.eval()

        all_similarities = []
        all_char_preds: List[int] = []
        all_char_labels: List[int] = []

        with torch.no_grad():
            for record in tqdm(val_records, desc=f"Predicting @ {thr}"):
                true_expression = record["true_expression"]
                img = Image.open(record["img_path"])
                predicted_chars = []

                for i, bb in enumerate(record["boxes"]):
                    x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"])
                    crop = img.crop((x, y, x + w, y + h)).convert("L")
                    crop_tensor = weak_transform(crop).unsqueeze(0).to(device)
                    output = model(crop_tensor)
                    pred_idx = output.argmax(dim=1).item()
                    predicted_chars.append(idx2label.get(pred_idx, "?"))

                    true_char = true_expression[i]
                    if true_char in label2idx:
                        all_char_preds.append(pred_idx)
                        all_char_labels.append(label2idx[true_char])

                predicted_expression = "".join(predicted_chars)
                dist = Levenshtein.distance(true_expression, predicted_expression)
                norm = max(len(true_expression), len(predicted_expression), 1)
                similarity = 1 - (dist / norm)
                all_similarities.append(similarity)

        mean_similarity = float(np.mean(all_similarities)) if all_similarities else 0.0
        char_acc = float(np.mean(np.array(all_char_preds) == np.array(all_char_labels))) if all_char_preds else 0.0

        results[thr] = {"similarity": mean_similarity, "char_acc": char_acc}
        print(f"→ thr={thr}: Levenshtein Sim={mean_similarity:.2%} | Char Acc={char_acc:.2%}")

        if mean_similarity > best_similarity:
            best_similarity = mean_similarity
            best_threshold = thr
            best_model_preds = all_char_preds
            best_model_labels = all_char_labels

    return best_threshold, results, best_model_preds, best_model_labels

def save_eval_summary(out_eval: Path, results: Dict[float, Dict[str, float]], best_thr: Optional[float]):
    out_eval.mkdir(parents=True, exist_ok=True)
    (out_eval / "summary.json").write_text(json.dumps({
        "by_threshold": results,
        "best_threshold": best_thr,
    }, indent=2))

def plot_confusion_and_report(out_eval: Path, best_thr: float,
                              preds: List[int], labels: List[int],
                              label2idx: Dict[str, int]):
    out_eval.mkdir(parents=True, exist_ok=True)
    class_labels = sorted(list(label2idx.keys()))
    idx_order = [label2idx[k] for k in class_labels]

    report_txt = classification_report(y_true=labels, y_pred=preds,
                                       target_names=class_labels, digits=3)
    (out_eval / "classification_report.txt").write_text(report_txt)
    print("\n" + "="*60)
    print("Classification Report for Best Model")
    print("="*60)
    print(report_txt)
    print("="*60 + "\n")

    cm = confusion_matrix(y_true=labels, y_pred=preds, labels=idx_order)
    fig = plt.figure(figsize=(16, 14))
    if _HAS_SEABORN:
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                         xticklabels=class_labels, yticklabels=class_labels,
                         linewidths=.5, linecolor='lightgray')
        ax.set_title(f'Confusion Matrix (Threshold = {best_thr})')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    else:
        plt.imshow(cm, interpolation="nearest")
        plt.title(f'Confusion Matrix (Threshold = {best_thr})')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels)
    plt.tight_layout()
    save_fig(fig, out_eval / "confusion_matrix.png")


def run_test_inference(boxes_csv: Path, test_img_dir: str, model_weights: Path,
                       weak_transform, num_classes: int, idx2label: Dict[int, str],
                       out_dir: Path, device: torch.device, batch_size: int = 256):
    df = pd.read_csv(boxes_csv)
    required_cols = {"image_id", "x", "y", "width", "height"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"boxes CSV missing columns: {missing}")
    try:
        df["image_id"] = df["image_id"].astype(int)
    except Exception:
        pass

    tensors, meta = [], []
    for i, row in df.iterrows():
        img_id = int(row["image_id"])
        x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])
        for ext in (".png", ".jpg", ".jpeg"):
            img_path = os.path.join(test_img_dir, f"{img_id}{ext}")
            if os.path.exists(img_path): break
        else:
            continue
        im = Image.open(img_path).convert("L")
        W, H = im.size
        left = max(0, int(np.floor(x)))
        top = max(0, int(np.floor(y)))
        right = min(W, int(np.ceil(x + w)))
        bottom = min(H, int(np.ceil(y + h)))
        crop = im.crop((left, top, right, bottom))
        tensors.append(weak_transform(crop))
        meta.append((i, img_id, x))

    X = torch.stack(tensors, dim=0) if len(tensors) else torch.empty((0,1,32,32))
    model = ImprovedCNN(num_classes=num_classes).to(device)
    state = torch.load(model_weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    pred_idx = np.empty(len(X), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            batch = X[start:end].to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            pred_idx[start:end] = preds

    pred_chars = [idx2label[int(k)] for k in pred_idx]

    pred_box_df = df.copy()
    pred_box_df["pred_char"] = pred_chars
    pred_per_box_path = out_dir / "pred_per_box.csv"
    pred_box_df.to_csv(pred_per_box_path, index=False)

    by_image = defaultdict(list)
    for (row_idx, img_id, x), ch in zip(meta, pred_chars):
        by_image[img_id].append((x, ch))
    sub_rows = []
    for img_id, lst in by_image.items():
        lst.sort(key=lambda t: t[0])
        expr = "".join(ch for _, ch in lst)
        sub_rows.append((img_id, expr))
    submission = pd.DataFrame(sub_rows, columns=["image_id", "expression"]).sort_values("image_id")
    submission_path = out_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved per-box predictions -> {pred_per_box_path}")
    print(f"Saved submission -> {submission_path}")
    return pred_per_box_path, submission_path

def visualize_random_test_predictions(submission_csv: Path, test_img_dir: str,
                                      out_dir: Path, n: int = 3, seed: int = 42):
    if not submission_csv.exists():
        return
    df = pd.read_csv(submission_csv)
    if df.empty:
        return
    rng = random.Random(seed)  
    picks = df["image_id"].tolist()
    if not picks:
        return
    chosen = rng.sample(picks, k=min(n, len(picks)))

    fig, axes = plt.subplots(len(chosen), 1, figsize=(7, 6 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]
    for ax, img_id in zip(axes, chosen):
        expr = df.loc[df.image_id == img_id, "expression"].values[0]
        for ext in (".png", ".jpg", ".jpeg"):
            img_path = os.path.join(test_img_dir, f"{int(img_id)}{ext}")
            if os.path.exists(img_path): break
        else:
            continue
        img = Image.open(img_path).convert("RGB")
        canvas = fit_on_canvas(img, target=384)
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, 8), f"Predicted: {expr}", fill=(255, 0, 0), font=font)
        ax.imshow(canvas); ax.axis("off")
    out_path = out_dir / "random_test_predictions.png"
    save_fig(fig, out_path)
    print(f"Saved random test predictions figure -> {out_path}")


def visualize_random_labeled_sample(train_img_dir: str, train_lbl_dir: str, out_figs: Path):
    json_files = sorted([f for f in os.listdir(train_lbl_dir) if f.endswith(".json")])
    if not json_files:
        return
    label_file = json_files[0]  
    label_path = os.path.join(train_lbl_dir, label_file)
    base = label_file[:-5]
    for ext in (".png", ".jpg", ".jpeg"):
        img_path = os.path.join(train_img_dir, base + ext)
        if os.path.exists(img_path): break
    else:
        return
    with open(label_path, "r") as f:
        label_data = json.load(f)
    expr = label_data.get("expression", "")
    image = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for ann in label_data["annotations"]:
        bb = ann["boundingBox"]
        rect = patches.Rectangle((bb["x"], bb["y"]), bb["width"], bb["height"],
                                 linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
    ax.set_title(f"Expression: {expr}")
    ax.axis("off")
    save_fig(fig, out_figs / "random_labeled_sample.png")

def visualize_val_samples(val_records: List[Dict], model: nn.Module, weak_transform, idx2label: Dict[int, str],
                          out_figs: Path, title: str, n: int = 3, device: Optional[torch.device] = None):
    if len(val_records) == 0:
        return
    n = min(n, len(val_records))
    rows = []
    with torch.no_grad():
        for rec in val_records[:n]:  
            true_expr = rec["true_expression"]
            img = Image.open(rec["img_path"])
            pred_chars = []
            for bb in rec["boxes"]:
                x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"])
                crop = img.crop((x, y, x + w, y + h)).convert("L")
                tensor = weak_transform(crop).unsqueeze(0).to(device if device else "cpu")
                pred_idx = model(tensor).argmax(dim=1).item()
                pred_chars.append(idx2label[pred_idx])
            pred_expr = "".join(pred_chars)
            rows.append((rec["img_path"], true_expr, pred_expr))

    fig, axes = plt.subplots(len(rows), 1, figsize=(6, 6 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, (img_path, t, p) in zip(axes, rows):
        img = Image.open(img_path).convert("RGB")
        img_canvas = fit_on_canvas(img, target=256)
        draw = ImageDraw.Draw(img_canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((5, 5), f"T: {t} | P: {p}", fill=(255, 0, 0), font=font)
        ax.imshow(img_canvas); ax.axis("off")
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    save_fig(fig, out_figs / f"{title.replace(' ', '_').lower()}.png")


def parse_args():
    p = argparse.ArgumentParser("fixmatch_char_pipeline")
    p.add_argument("--data_dir", default="./dataset",
                   help="Dataset root (expects train/val/test subfolders)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--thresholds", default="0.8,0.85,0.9,0.95",
                   help="Comma-separated confidence thresholds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--test_boxes_csv", default="./results/faster_rcnn/output.csv",
                   help="CSV with columns image_id,x,y,width,height")
    return p.parse_args()

def main():
    args = parse_args()
    g = set_seed(args.seed)

    script_stem = Path(__file__).stem
    OUT = get_out_dirs(script_stem, out_root="results")
    print(f"[OUT] Base directory: {OUT['base']}")

    # Paths
    TRAIN_IMG_DIR, TRAIN_LABEL_DIR = get_split_paths(args.data_dir, "train")
    VAL_IMG_DIR,   VAL_LABEL_DIR   = get_split_paths(args.data_dir, "val")
    TEST_IMG_DIR = os.path.join(args.data_dir, "test", "images")

    weak_transform, strong_transform = get_transforms(img_size=args.img_size)

    # Scan & visualize
    labeled_samples, unlabeled_samples = scan_train_split(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
    print(f"Total labeled samples: {len(labeled_samples)}")
    print(f"Total unlabeled samples: {len(unlabeled_samples)}")
    visualize_random_labeled_sample(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, OUT["figs"])

    # Crop chars
    labeled_crop_info, unlabeled_crop_info = crop_all_chars(labeled_samples, unlabeled_samples, OUT["crops"])
    print(f"Total labeled crops: {len(labeled_crop_info)}")
    print(f"Total unlabeled crops: {len(unlabeled_crop_info)}")

    # Dataset/loader (deterministic shuffle via generator=g)
    fixmatch_ds = FixMatchDataset(labeled_crop_info, unlabeled_crop_info, weak_transform, strong_transform)
    num_classes = len(fixmatch_ds.label_set)
    train_loader = DataLoader(
        fixmatch_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=DATALOADER_WORKERS, worker_init_fn=seed_worker, generator=g
    )
    print(f"FixMatch Dataset len={len(fixmatch_ds)} | classes={num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    model_paths = train_fixmatch_across_thresholds(
        train_loader=train_loader,
        num_classes=num_classes,
        thresholds=thresholds,
        out_models=OUT["models"],
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_u=args.lambda_u,
    )

    # Validation
    val_records = load_validation(VAL_IMG_DIR, VAL_LABEL_DIR)
    print(f"Validation samples: {len(val_records)}")
    if not val_records:
        print("No validation records found; skipping evaluation and inference.")
        return

    best_thr, results, best_preds, best_labels = evaluate_threshold_models(
        model_paths=model_paths,
        num_classes=num_classes,
        val_records=val_records,
        weak_transform=weak_transform,
        idx2label=fioxmatch_ds.idx2label if False else fixmatch_ds.idx2label,  
        label2idx=fixmatch_ds.label2idx,
        device=device,
    )
    print("\n==================== Evaluation Summary ====================")
    print("Threshold | Levenshtein Sim. | Character Acc.")
    print("----------------------------------------------------------")
    for t in sorted(results.keys()):
        r = results[t]
        print(f"  {t:.2f}    |      {r['similarity']:.2%}      |    {r['char_acc']:.2%}")
    print("==========================================================")
    print(f"Best threshold: {best_thr}")

    save_eval_summary(OUT["eval"], results, best_thr)
    if best_thr is not None and best_preds and best_labels:
        plot_confusion_and_report(OUT["eval"], best_thr, best_preds, best_labels, fixmatch_ds.label2idx)

    if best_thr is not None:
        best_model_path = OUT["models"] / f"fixmatch_model_thresh_{int(best_thr*100)}.pth"
        if best_model_path.exists():
            model = ImprovedCNN(num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            visualize_val_samples(val_records, model, weak_transform, fixmatch_ds.idx2label,
                                  OUT["figs"], title="Validation Samples (T vs P)", n=3, device=device)

    boxes_csv = Path(args.test_boxes_csv)
    if boxes_csv.exists():
        chosen = None
        if best_thr is not None:
            candidate = OUT["models"] / f"fixmatch_model_thresh_{int(best_thr*100)}.pth"
            if candidate.exists():
                chosen = candidate
        if chosen is None and model_paths:
            chosen = sorted(model_paths, key=lambda p: int(p.stem.split("_")[-1]))[-1]

        if chosen is None:
            print("No model weights available for test inference.")
        else:
            print(f"Using weights for test inference: {chosen.name}")
            per_box_csv, submission_csv = run_test_inference(
                boxes_csv=boxes_csv,
                test_img_dir=TEST_IMG_DIR,
                model_weights=chosen,
                weak_transform=weak_transform,
                num_classes=num_classes,
                idx2label=fixmatch_ds.idx2label,
                out_dir=OUT["infer"],
                device=device,
            )
            visualize_random_test_predictions(submission_csv, TEST_IMG_DIR, OUT["infer"], n=3, seed=args.seed)
    else:
        print(f"(Info) Test boxes CSV not found at: {boxes_csv} — skipping test inference.")

if __name__ == "__main__":
    main()
