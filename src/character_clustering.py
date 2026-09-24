from __future__ import annotations
import argparse
import itertools
import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import skimage.color as skcolor
import skimage.transform as sktransform
import torch
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from skimage.feature import hog, local_binary_pattern
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

SAVE_DIR: Optional[Path] = None
FIG_COUNT = 0
SHOW_FIGS = False  

def _save_and_show(fig: plt.Figure, title: str):
    """Save as PNG (if SAVE_DIR is set). Do NOT show by default in notebooks."""
    global FIG_COUNT, SAVE_DIR, SHOW_FIGS
    if SAVE_DIR is not None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{FIG_COUNT:02d}_{title}.png"
        fig.savefig(SAVE_DIR / filename, dpi=150, bbox_inches="tight")
    FIG_COUNT += 1
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)  

def _reset_fig_counter():
    global FIG_COUNT
    FIG_COUNT = 0

DATA_ROOT = Path("dataset")
TRAIN_IMAGES = DATA_ROOT / "train/images"
TRAIN_LABELS = DATA_ROOT / "train/labels"

class CharacterDataset(Dataset):
    def __init__(self, img_dir: Path, lbl_dir: Path):
        self.files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        filename = self.files[idx]
        image = Image.open(self.img_dir / filename).convert("RGB")
        with open(self.lbl_dir / filename.replace(".png", ".json")) as f:
            annotations = json.load(f)["annotations"]
        boxes = [[x, y, x + w, y + h]
                 for x, y, w, h in (ann["boundingBox"].values() for ann in annotations)]
        tensor_img = self.transform(image)
        return tensor_img, {"boxes": torch.tensor(boxes, dtype=torch.int64)}

IMAGE_SIZE = 32
LBP_P, LBP_R = 8, 1

def _gray32(img: Image.Image) -> np.ndarray:
    gray = skcolor.rgb2gray(np.asarray(img))
    resized = sktransform.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
    return resized.astype(np.float32)

def feat_raw(img: Image.Image) -> np.ndarray:
    return _gray32(img).flatten()

def feat_hog(img: Image.Image) -> np.ndarray:
    return hog(_gray32(img), pixels_per_cell=(8,8), cells_per_block=(2,2),
               orientations=9, feature_vector=True).astype(np.float32)

def feat_lbp(img: Image.Image) -> np.ndarray:
    lbp = local_binary_pattern(_gray32(img), P=LBP_P, R=LBP_R, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_P+2, range=(0,LBP_P+2), density=True)
    return hist.astype(np.float32)

def feat_hog_lbp(img: Image.Image) -> np.ndarray:
    h = feat_hog(img)
    l = feat_lbp(img)
    return np.concatenate([h, l], axis=0)

FEATURES: Dict[str, Callable[[Image.Image], np.ndarray]] = {
    "raw": feat_raw,
    "hog": feat_hog,
    "lbp": feat_lbp,
    "hog_lbp": feat_hog_lbp,
}

def extract_crops(ds: Dataset) -> List[Image.Image]:
    crops: List[Image.Image] = []
    loader = DataLoader(ds, batch_size=1)
    for img_t, target in loader:
        pil_img = to_pil_image(img_t.squeeze())
        for box in target["boxes"][0]:
            crops.append(pil_img.crop(tuple(box.tolist())))
    return crops

def l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-9)

def mean_kmeans_score(feats: np.ndarray, metric_fun) -> float:
    scores = []
    for k_val in range(5, 16):
        lbls = KMeans(k_val, n_init="auto", random_state=42).fit_predict(feats)
        scores.append(metric_fun(feats, lbls))
    return float(np.mean(scores))

def choose_feature_space(x: np.ndarray, dims: Tuple[int, ...], metric_fun) -> np.ndarray:
    base = l2_normalise(x)
    best_feats = base
    best_val = mean_kmeans_score(base, metric_fun)
    best_label = "RAW"
    print(f"[PCA] test RAW → {metric_fun.__name__}={best_val:.3f}")
    n_samples, n_feats = x.shape
    for d in dims:
        if d >= min(n_samples - 1, n_feats):
            continue
        proj = PCA(d, random_state=0).fit_transform(x)
        proj = l2_normalise(proj)
        val = mean_kmeans_score(proj, metric_fun)
        print(f"[PCA] test {d} → {metric_fun.__name__}={val:.3f}")
        if val > best_val:
            best_feats, best_val, best_label = proj, val, str(d)
    print(f"[PCA] >>> selected {best_label}")
    return best_feats

def plot_pca(feats: np.ndarray, labels: np.ndarray, title: str):
    coords = PCA(2, random_state=0).fit_transform(feats) if feats.shape[1] > 2 else feats[:, :2]
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab10", alpha=0.7)
    count = len(set(labels)) - (1 if -1 in labels else 0)
    plt.title(f"{title} (k={count})")
    plt.axis("equal"); plt.grid(lw=0.3); plt.tight_layout()
    _save_and_show(fig, "pca_scatter")

def show_samples(crops: List[Image.Image], labels: np.ndarray, n_per: int = 6):
    rng = random.Random(42)
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idxs = np.where(labels == cid)[0]
        sel = rng.sample(list(idxs), min(n_per, len(idxs)))
        fig = plt.figure(figsize=(1.2 * len(sel), 1.6))
        for i, idx in enumerate(sel, start=1):
            plt.subplot(1, len(sel), i)
            plt.imshow(crops[idx]); plt.axis("off")
        plt.suptitle(f"Cluster {cid} – {len(idxs)} samples")
        plt.tight_layout()
        _save_and_show(fig, f"cluster_{cid}")

MetricFun = {"silhouette": silhouette_score, "db": davies_bouldin_score}
MetricLbl  = {"silhouette": "Silhouette", "db": "Davies-Bouldin"}

def run_single_config(
    feature: str,
    cluster: str,
    metric: str,
    k: int = 16,
    pca_dim: Optional[int] = None,
    save_dir: Optional[str] = "results/character_clustering",
    show: bool = False,
):
    """
    Run a single configuration end-to-end and SAVE figures (no inline show unless show=True).
    """
    global SAVE_DIR, SHOW_FIGS
    SAVE_DIR = None if (save_dir is None or str(save_dir).lower() == "none") else Path(str(save_dir))
    SHOW_FIGS = bool(show)
    _reset_fig_counter()

    dataset = CharacterDataset(TRAIN_IMAGES, TRAIN_LABELS)
    crops = extract_crops(dataset)

    if feature not in FEATURES:
        raise ValueError(f"Unknown feature '{feature}'. Options: {list(FEATURES.keys())}")
    X = np.stack([FEATURES[feature](crop) for crop in crops])

    metric_fun = MetricFun[metric]
    if pca_dim is not None:
        print(f"[PCA] fixed dimension: {pca_dim}")
        feats = PCA(pca_dim, random_state=0).fit_transform(X)
        feats = l2_normalise(feats)
    else:
        feats = choose_feature_space(X, dims=(16, 32, 48, 64, 96, 128), metric_fun=metric_fun)

    m_label = MetricLbl[metric]
    if cluster == "kmeans":
        print(f"[KMeans] k={k}")
        labels = KMeans(k, n_init="auto", random_state=42).fit_predict(feats)
        score = metric_fun(feats, labels)
        print(f"→ {m_label}: {score:.3f}")
    elif cluster == "agglomerative":
        print(f"[Agglomerative] k={k}")
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(feats)
        score = metric_fun(feats, labels)
        print(f"→ {m_label}: {score:.3f}")
    elif cluster == "gmm":
        print(f"[GMM] n_components={k}")
        gmm = GaussianMixture(n_components=k, random_state=0).fit(feats)
        labels = gmm.predict(feats)
        score = metric_fun(feats, labels)
        bic = gmm.bic(feats)
        print(f"→ {m_label}: {score:.3f}, BIC: {bic:.1f}")
    else:
        raise ValueError(f"Unknown cluster '{cluster}'. Options: kmeans, agglomerative, gmm")

    title = f"{cluster.upper()} | {feature.upper()} | {m_label}"
    plot_pca(feats, labels, title)
    show_samples(crops, labels)

def run_grid(
    features: List[str],
    clusters: List[str],
    metrics: List[str],
    k: int = 16,
    pca_dim: Optional[int] = None,
    save_dir: Optional[str] = "results/character_clustering",
    show: bool = False,
):
  
    for feat, clus, met in itertools.product(features, clusters, metrics):
        header = f"\n=== feature={feat} | cluster={clus} | metric={met} ==="
        print(header)

        tag = f"{feat}_{clus}_{met}_k{k}_" + (f"pca{pca_dim}" if pca_dim is not None else "pcaauto")
        subdir = None if (save_dir is None or str(save_dir).lower() == "none") else Path(save_dir) / tag

        try:
            run_single_config(
                feature=feat,
                cluster=clus,
                metric=met,
                k=k,
                pca_dim=pca_dim,
                save_dir=subdir,   
                show=show          
            )
        except Exception as e:
            print(f"Error during run feature={feat}, cluster={clus}, metric={met}: {e}")
            break

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("unified_character_clustering")

    parser.add_argument("--feature", choices=list(FEATURES.keys()), default="hog")
    parser.add_argument("--cluster", choices=["kmeans", "agglomerative", "gmm"], default="kmeans")
    parser.add_argument("--metric", choices=list(MetricFun.keys()), default="silhouette")
    parser.add_argument("--k", type=int, default=16, help="number of clusters/components")
    parser.add_argument("--pca_dim", type=int, help="fixed PCA dimension (skip auto PCA search)")
    parser.add_argument("--save_dir", default="results/character_clustering",
                        help="directory to save figures or 'none' to disable")
    parser.add_argument("--show", action="store_true",
                        help="show figures inline (default: only save & close)")

    parser.add_argument("--run_grid", action="store_true",
                        help="run the Cartesian product of configs")
    parser.add_argument("--grid_features", nargs="+", default=["raw", "hog", "lbp", "hog_lbp"],
                        help="subset of features for grid")
    parser.add_argument("--grid_clusters", nargs="+", default=["kmeans", "agglomerative", "gmm"],
                        help="subset of clusters for grid")
    parser.add_argument("--grid_metrics",  nargs="+", default=["silhouette", "db"],
                        help="subset of metrics for grid")

    return parser.parse_args()

def main():
    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.run_grid:
        run_grid(
            features=args.grid_features,
            clusters=args.grid_clusters,
            metrics=args.grid_metrics,
            k=args.k,
            pca_dim=args.pca_dim,
            save_dir=args.save_dir,
            show=args.show,
        )
    else:
        run_single_config(
            feature=args.feature,
            cluster=args.cluster,
            metric=args.metric,
            k=args.k,
            pca_dim=args.pca_dim,
            save_dir=args.save_dir,
            show=args.show,
        )

if __name__ == "__main__":
    main()
