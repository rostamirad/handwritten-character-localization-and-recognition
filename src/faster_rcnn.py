
import os
import random
import json
import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.ops import roi_align, nms, box_iou, clip_boxes_to_image
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList
from tqdm import tqdm

STRICT_DETERMINISM = True    
CPU_ROI_ALIGN       = True    
DISABLE_AMP         = True    
DATALOADER_WORKERS  = 0       

def seed_worker(worker_id):
    """
    Helper function to seed each DataLoader worker for reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int = 42) -> torch.Generator:
    """Sets seed for all relevant libraries for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

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
    print(f"Seed set to {seed}. Deterministic algorithms {'strict' if STRICT_DETERMINISM else 'warn_only'} mode.")
    return g


class CharacterBoxDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None, max_size=1024):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.max_size = max_size
        self.img_files = []

        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            base = os.path.splitext(img_name)[0]
            jpath = os.path.join(label_dir, base + '.json')
            if not os.path.exists(jpath):
                continue
            try:
                with open(jpath, 'r', encoding='utf-8') as f:
                    jd = json.load(f)
                if jd.get('annotations'):
                    self.img_files.append(img_name)
            except (json.JSONDecodeError, KeyError):
                print(f"Warning: Could not process {img_name}, skipping.")

        print(f"Found {len(self.img_files)} images with valid annotations in {img_dir}.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        base = os.path.splitext(img_name)[0]
        jpath = os.path.join(self.label_dir, base + '.json')
        with open(jpath, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        boxes = []
        for ann in label_data.get('annotations', []):
            if 'boundingBox' in ann:
                bb = ann['boundingBox']
                x1, y1 = float(bb['x']), float(bb['y'])
                x2, y2 = x1 + float(bb['width']), y1 + float(bb['height'])
                boxes.append([x1, y1, x2, y2])

        w0, h0 = image.size
        if self.max_size is not None:
            longest = max(h0, w0)
            if longest > self.max_size:
                s = self.max_size / float(longest)
                image = image.resize((max(1, int(round(w0*s))), max(1, int(round(h0*s)))), Image.Resampling.BILINEAR)
                if boxes:
                    boxes = (torch.tensor(boxes, dtype=torch.float32) * s).tolist()

        labels = torch.ones((len(boxes),), dtype=torch.int64)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32),
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "size": torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        }

        if self.transforms:
            image = self.transforms(image)
        return image, target

class TestCharacterDataset(Dataset):
    def __init__(self, img_dir, transform=None, max_size=1024):
        self.img_dir = img_dir
        self.transform = transform
        self.max_size = max_size
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        path = os.path.join(self.img_dir, img_name)
        im = Image.open(path).convert("RGB")

        w0, h0 = im.size
        scale = 1.0
        if self.max_size is not None:
            longest = max(h0, w0)
            if longest > self.max_size:
                scale = self.max_size / float(longest)
                new_w = max(1, int(round(w0 * scale)))
                new_h = max(1, int(round(h0 * scale)))
                im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)

        if self.transform:
            im = self.transform(im)

        image_id = os.path.splitext(img_name)[0]  
        return im, image_id, torch.tensor(scale, dtype=torch.float32), torch.tensor([h0, w0], dtype=torch.int64)

def collate_fn(batch):
    return tuple(zip(*batch))

def pad_to_common(images, stride):
    sizes = [img.shape[-2:] for img in images]
    max_h = max(h for h, w in sizes)
    max_w = max(w for h, w in sizes)
    pad_h = ((max_h + stride - 1) // stride) * stride
    pad_w = ((max_w + stride - 1) // stride) * stride
    padded = [F.pad(img, (0, pad_w - img.shape[-1], 0, pad_h - img.shape[-2])) for img in images]
    return torch.stack(padded, dim=0), sizes

def dict_tensors_to_cpu(d):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in d.items()}


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        groups = 32 if c_out % 32 == 0 else (16 if c_out % 16 == 0 else 8)
        self.bn = nn.GroupNorm(groups, c_out)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class TinyBackbone(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(3,    base,   s=2), ConvBlock(base, base),
            ConvBlock(base, base*2, s=2), ConvBlock(base*2, base*2),
            ConvBlock(base*2, base*4, s=2), ConvBlock(base*4, base*4),
            ConvBlock(base*4, 256,    s=2), ConvBlock(256,    256),
        )
        self.out_channels = 256
        self.out_stride = 16
    def forward(self, x): return self.body(x)

class FastRCNNHead(nn.Module):
    def __init__(self, in_channels=256, pool_size=7, hidden=1024, num_classes=2):
        super().__init__()
        self.pool_size = pool_size
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cls_score = nn.Linear(hidden, num_classes)
        self.bbox_pred = nn.Linear(hidden, 4)

        for l in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        prior = 0.01
        with torch.no_grad():
            if self.cls_score.bias.numel() > 1:
                self.cls_score.bias[1] = -math.log((1 - prior) / prior)

    def forward(self, roi_feats):
        x = roi_feats.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.cls_score(x)
        deltas = self.bbox_pred(x)
        return scores, deltas

def encode_boxes(ex_rois, gt_rois, weights):
    ex_w = ex_rois[:, 2] - ex_rois[:, 0]; ex_h = ex_rois[:, 3] - ex_rois[:, 1]
    ex_cx = ex_rois[:, 0] + 0.5 * ex_w;   ex_cy = ex_rois[:, 1] + 0.5 * ex_h
    gt_w = gt_rois[:, 2] - gt_rois[:, 0]; gt_h = gt_rois[:, 3] - gt_rois[:, 1]
    gt_cx = gt_rois[:, 0] + 0.5 * gt_w;   gt_cy = gt_rois[:, 1] + 0.5 * gt_h
    dx = (gt_cx - ex_cx) / ex_w.clamp(min=1e-6)
    dy = (gt_cy - ex_cy) / ex_h.clamp(min=1e-6)
    dw = torch.log(gt_w.clamp(min=1e-6) / ex_w.clamp(min=1e-6))
    dh = torch.log(gt_h.clamp(min=1e-6) / ex_h.clamp(min=1e-6))
    deltas = torch.stack([dx, dy, dw, dh], dim=1)
    return deltas * weights.to(deltas.device)

def decode_boxes(boxes, deltas):
    w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + 0.5 * w;     cy = boxes[:, 1] + 0.5 * h
    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    pred_cx = cx + dx * w;      pred_cy = cy + dy * h
    pred_w  = torch.exp(dw) * w;  pred_h  = torch.exp(dh) * h
    x1 = pred_cx - 0.5 * pred_w;  y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w;  y2 = pred_cy + 0.5 * pred_h
    return torch.stack([x1, y1, x2, y2], dim=1)


class MyFasterRCNN(nn.Module):
    def __init__(self, num_classes=2,
                 anchor_sizes=(8, 12, 16, 24, 32, 48, 64, 96, 128),
                 anchor_ratios=(0.5, 1.0, 2.0),
                 rpn_nms_thresh=0.7, box_score_thresh=0.05, box_nms_thresh=0.3,
                 box_batch_size_per_image=256, box_positive_fraction=0.5, pool_size=7):
        super().__init__()
        self.backbone = TinyBackbone()
        self.out_channels = self.backbone.out_channels
        self.feature_stride = self.backbone.out_stride

        anchor_generator = AnchorGenerator(sizes=(anchor_sizes,), aspect_ratios=(anchor_ratios,))
        rpn_head = RPNHead(self.out_channels, anchor_generator.num_anchors_per_location()[0])
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator, head=rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=2000),
            post_nms_top_n=dict(training=1000, testing=300),
            nms_thresh=rpn_nms_thresh
        )

        self.box_head = FastRCNNHead(in_channels=self.out_channels, pool_size=pool_size, hidden=1024, num_classes=num_classes)
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction
        self.pool_size = pool_size
        self.box_weights = torch.tensor([10., 10., 5., 5.])

    def _images_to_image_list(self, images):
        batch, sizes = pad_to_common(images, stride=self.feature_stride)
        return ImageList(batch, [img.shape[-2:] for img in images]), sizes

    def forward(self, images, targets=None):
        device = images[0].device if isinstance(images, list) else images.device
        image_list, sizes_hw = self._images_to_image_list(images)
        feats = self.backbone(image_list.tensors)
        features = OrderedDict([("0", feats)])
        proposals, rpn_losses = self.rpn(image_list, features, targets)

        if self.training:
            losses_box = self._roi_forward_train(features, proposals, targets, device)
            losses = {**rpn_losses, **losses_box}
            return losses
        else:
            return self._roi_forward_infer(features, proposals, sizes_hw, device)

    def _sample_proposals(self, proposals, targets, device):
        sampled_rois, matched_gt, cls_targets = [], [], []
        for props, target in zip(proposals, targets):
            gt_boxes = target["boxes"].to(device)
            if gt_boxes.numel() > 0:
                props = torch.cat([props, gt_boxes], dim=0)

            if gt_boxes.numel() == 0:
                num = min(self.box_batch_size_per_image, len(props))
                idx = torch.randperm(len(props), device=device)[:num] if len(props) > 0 else torch.empty(0, dtype=torch.long, device=device)
                sampled_rois.append(props[idx])
                matched_gt.append(torch.zeros_like(props[idx]))
                cls_targets.append(torch.zeros(len(idx), dtype=torch.int64, device=device))
                continue

            ious = box_iou(props, gt_boxes)
            max_iou, max_id = ious.max(dim=1)
            pos_idx = torch.where(max_iou >= 0.5)[0]
            neg_idx = torch.where((max_iou < 0.5) & (max_iou >= 0.1))[0]
            num_pos = min(int(self.box_batch_size_per_image * self.box_positive_fraction), len(pos_idx))
            num_neg = min(self.box_batch_size_per_image - num_pos, len(neg_idx))
            perm_pos = torch.randperm(len(pos_idx), device=device)[:num_pos]
            perm_neg = torch.randperm(len(neg_idx), device=device)[:num_neg]
            keep_idx = torch.cat([pos_idx[perm_pos], neg_idx[perm_neg]])
            sampled_rois.append(props[keep_idx])
            cls_t = torch.zeros(keep_idx.numel(), dtype=torch.int64, device=device)
            cls_t[:num_pos] = 1
            cls_targets.append(cls_t)
            matched_gt.append(gt_boxes[max_id[keep_idx]])
        return sampled_rois, matched_gt, cls_targets

    def _roi_forward_train(self, features, proposals, targets, device):
        sampled_proposals, matched_gts, cls_targets_list = self._sample_proposals(proposals, targets, device)
        if not any(p.numel() > 0 for p in sampled_proposals):
            return {"loss_classifier": torch.tensor(0.0, device=device), "loss_box_reg": torch.tensor(0.0, device=device)}

        roi_list = [torch.cat([torch.full((rois.shape[0], 1), i, dtype=rois.dtype, device=device), rois], dim=1) for i, rois in enumerate(sampled_proposals) if rois.numel() > 0]
        if not roi_list:
            return {"loss_classifier": torch.tensor(0.0, device=device), "loss_box_reg": torch.tensor(0.0, device=device)}

        rois_all = torch.cat(roi_list, dim=0)

        if CPU_ROI_ALIGN:
            pooled_feats_cpu = roi_align(
                features['0'].to('cpu'), rois_all.to('cpu'),
                (self.pool_size, self.pool_size),
                1.0/self.feature_stride, aligned=True
            )
            pooled_feats = pooled_feats_cpu.to(features['0'].device)
        else:
            pooled_feats = roi_align(
                features['0'], rois_all,
                (self.pool_size, self.pool_size),
                1.0/self.feature_stride, aligned=True
            )

        class_logits, box_regression = self.box_head(pooled_feats)
        cls_targets = torch.cat(cls_targets_list, dim=0)
        loss_cls = F.cross_entropy(class_logits, cls_targets)

        pos_mask = cls_targets > 0
        if pos_mask.any():
            target_deltas = encode_boxes(torch.cat(sampled_proposals, dim=0)[pos_mask], torch.cat(matched_gts, dim=0)[pos_mask], self.box_weights)
            loss_reg = F.smooth_l1_loss(box_regression[pos_mask], target_deltas, reduction='sum') / pos_mask.sum().clamp(min=1)
        else:
            loss_reg = torch.tensor(0.0, device=device)
        return {"loss_classifier": loss_cls, "loss_box_reg": loss_reg}

    @torch.no_grad()
    def _roi_forward_infer(self, features, proposals, sizes_hw, device):
        results = []
        feat = features["0"]

        for i, (props, (H, W)) in enumerate(zip(proposals, sizes_hw)):
            props = clip_boxes_to_image(props, (H, W))

            if props.numel() == 0:
                results.append({"boxes": torch.zeros((0,4), device=device), "labels": torch.zeros((0,), dtype=torch.int64, device=device), "scores": torch.zeros((0,), device=device)})
                continue

            props = props.contiguous()
            finite = torch.isfinite(props).all(dim=1)
            wh = props[:, 2:4] - props[:, 0:2]
            pos_area = (wh[:, 0] > 1e-6) & (wh[:, 1] > 1e-6)
            keep_mask = finite & pos_area

            if not keep_mask.any():
                results.append({"boxes": torch.zeros((0,4), device=device), "labels": torch.zeros((0,), dtype=torch.int64, device=device), "scores": torch.zeros((0,), device=device)})
                continue

            props = props[keep_mask].contiguous()
            rois = torch.cat([torch.full((props.shape[0], 1), 0, dtype=props.dtype, device=device), props], dim=1).contiguous()

            if CPU_ROI_ALIGN:
                roi_feats_cpu = roi_align(
                    feat[i:i+1].to('cpu'), rois.to('cpu'),
                    (self.pool_size, self.pool_size),
                    spatial_scale=1.0 / float(self.feature_stride),
                    aligned=True
                )
                roi_feats = roi_feats_cpu.to(feat.device)
            else:
                roi_feats = roi_align(
                    feat[i:i+1], rois,
                    (self.pool_size, self.pool_size),
                    spatial_scale=1.0 / float(self.feature_stride),
                    aligned=True
                )

            scores, deltas = self.box_head(roi_feats)
            fg_scores = F.softmax(scores, dim=1)[:, 1]

            keep = fg_scores >= self.box_score_thresh
            if not keep.any():
                results.append({"boxes": torch.zeros((0,4), device=device), "labels": torch.zeros((0,), dtype=torch.int64, device=device), "scores": torch.zeros((0,), device=device)})
                continue

            props_kept, deltas_kept, scores_kept = props[keep], deltas[keep], fg_scores[keep]
            boxes_dec = decode_boxes(props_kept, deltas_kept / self.box_weights.to(deltas_kept.device))
            boxes_dec = clip_boxes_to_image(boxes_dec, (H, W)).contiguous()
            keep_nms = nms(boxes_dec, scores_kept, self.box_nms_thresh)
            boxes_final, scores_final = boxes_dec[keep_nms], scores_kept[keep_nms]

            results.append({"boxes": boxes_final, "labels": torch.ones(boxes_final.size(0), dtype=torch.int64, device=device), "scores": scores_final})
        return results


@torch.no_grad()
def evaluate_f1(model, data_loader, device, iou_thresh=0.5, score_thresh=0.5):
    model.eval()
    TP, FP, FN = 0, 0, 0
    orig_thresh = model.box_score_thresh
    model.box_score_thresh = score_thresh
    try:
        for images, targets in tqdm(data_loader, desc="Evaluating F1"):
            images = [im.to(device) for im in images]
            preds = model(images)
            for pred, tgt in zip(preds, targets):
                pb, ps = pred['boxes'].cpu(), pred['scores'].cpu()
                gb = tgt['boxes'].cpu()
                if gb.numel() == 0:
                    FP += pb.shape[0]
                    continue
                if pb.numel() == 0:
                    FN += gb.shape[0]
                    continue

                order = torch.argsort(ps, descending=True)
                pb = pb[order]
                assigned = torch.zeros(gb.shape[0], dtype=torch.bool)
                for i in range(pb.shape[0]):
                    ious = box_iou(pb[i].unsqueeze(0), gb).squeeze(0)
                    ious[assigned] = 0.0
                    max_iou, max_idx = ious.max(dim=0)
                    if max_iou >= iou_thresh:
                        TP += 1
                        assigned[max_idx] = True
                    else:
                        FP += 1
                FN += (~assigned).sum().item()
        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1
    finally:
        model.box_score_thresh = orig_thresh

@torch.no_grad()
def calculate_and_plot_map_vs_iou(model, data_loader, device, output_dir):
    print("\n--- Calculating mAP vs. IoU Threshold ---")
    model.eval()

    all_preds, all_gts = [], []
    for images, targets in tqdm(data_loader, desc="Gathering predictions for mAP"):
        images = [im.to(device) for im in images]
        preds = model(images)
        all_preds.extend([dict_tensors_to_cpu(p) for p in preds])
        all_gts.extend([dict_tensors_to_cpu(t) for t in targets])

    iou_thresholds = np.arange(0.30, 1.0, 0.05)
    ap_scores = []

    for iou_thresh in tqdm(iou_thresholds, desc="Calculating AP per IoU"):
        all_scores = []
        all_matches = []
        total_gt_boxes = sum(len(t['boxes']) for t in all_gts)

        for pred, gt in zip(all_preds, all_gts):
            gt_boxes = gt['boxes']
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']

            if pred_boxes.numel() == 0:
                continue

            sort_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]

            matches = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
            if gt_boxes.numel() > 0:
                gt_assigned = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
                ious = box_iou(pred_boxes, gt_boxes)
                for i in range(pred_boxes.shape[0]):
                    iou_row = ious[i, :]
                    iou_row[gt_assigned] = -1
                    max_iou, max_idx = iou_row.max(0)
                    if max_iou >= iou_thresh:
                        matches[i] = 1
                        gt_assigned[max_idx] = True

            all_scores.extend(pred_scores.tolist())
            all_matches.extend(matches.tolist())

        if total_gt_boxes == 0:
            ap_scores.append(1.0 if len(all_scores) == 0 else 0.0)
            continue

        if len(all_scores) == 0:
            ap_scores.append(0.0)
            continue

        sort_indices = np.argsort(all_scores)[::-1]
        all_matches_sorted = np.array(all_matches)[sort_indices]

        tp = np.cumsum(all_matches_sorted)
        fp = np.cumsum(1 - all_matches_sorted)

        recall = tp / total_gt_boxes
        precision = tp / (tp + fp)

        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.
        ap_scores.append(ap)

    plt.figure(figsize=(10, 6))
    plt.plot(iou_thresholds, ap_scores, marker='o', linestyle='-', label='AP per Threshold')
    plt.axvline(x=0.5, color='r', linestyle='--', label='mAP@0.5 (PASCAL VOC)')
    plt.axvline(x=0.75, color='g', linestyle='--', label='mAP@0.75 (Strict)')
    plt.xlabel("IoU Threshold")
    plt.ylabel("Average Precision (AP)")
    plt.title("Model Performance vs. IoU Threshold")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()

    plot_path = os.path.join(output_dir, "map_vs_iou_threshold.png")
    plt.savefig(plot_path)
    print(f" mAP vs. IoU plot saved to {plot_path}")
    plt.show()


def visualize_predictions(model, dataset, device, num_samples=3, threshold=0.5, output_dir='.'):
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 6))
    if num_samples == 1: axes = [axes]
    idx2label = {1: 'char'}

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, ax in enumerate(axes):
        if len(dataset) == 0:
            print("Validation dataset is empty, cannot visualize.")
            return
        image, _ = dataset[indices[i]]
        with torch.no_grad():
            orig = model.box_score_thresh
            model.box_score_thresh = threshold
            preds = model([image.to(device)])
            model.box_score_thresh = orig
        pred = preds[0]

        img_pil = transforms.ToPILImage()(image.cpu())
        draw = ImageDraw.Draw(img_pil)
        for box, label, score in zip(pred['boxes'].cpu(), pred['labels'].cpu(), pred['scores'].cpu()):
            if score.item() >= threshold:
                b = box.tolist()
                draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="lime", width=3)
                draw.text((b[0], max(0, b[1]-15)), f"{idx2label.get(label.item(),'N/A')} ({score:.2f})", fill="lime")

        ax.imshow(img_pil)
        ax.set_title(f"Sample {i+1}")
        ax.axis('off')

    plt.tight_layout()
    viz_path = os.path.join(output_dir, "fastrcnn_results.png")
    plt.savefig(viz_path)
    print(f"Predictions visualization saved to {viz_path}")
    plt.show()


def main():
    """Main function to run the training and evaluation pipeline."""
    SEED = 42
    generator = set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    OUTPUT_DIR = os.path.join('results', 'faster_rcnn')
    train_img_dir = 'dataset/train/images'
    train_label_dir = 'dataset/train/labels'
    val_img_dir = 'dataset/val/images'
    val_label_dir = 'dataset/val/labels'
    test_img_dir = 'dataset/test/images'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True); os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True);   os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    MAX_SIZE, BATCH_SIZE, NUM_WORKERS_LOADER = 800, 4, DATALOADER_WORKERS
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS = 2e-4, 1e-4, 50

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = CharacterBoxDataset(train_img_dir, train_label_dir, tfm, max_size=MAX_SIZE)
    val_ds   = CharacterBoxDataset(val_img_dir,   val_label_dir,   tfm, max_size=MAX_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS_LOADER,
        worker_init_fn=seed_worker,
        generator=generator
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS_LOADER,
        worker_init_fn=seed_worker,
        generator=generator
    )

    model = MyFasterRCNN(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler(device.type, enabled=(not DISABLE_AMP and device.type == 'cuda'))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    print(f"\n--- Starting Training for {EPOCHS} epochs ---")
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, targets in pbar_train:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss_dict = model(images, targets)
                loss = sum(loss_dict[k] for k in sorted(loss_dict.keys()))

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pbar_train.set_postfix(loss=f"{loss.item():.4f}")

        precision, recall, f1 = evaluate_f1(model, val_loader, device, iou_thresh=0.5, score_thresh=0.5)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        lr_scheduler.step()

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_f1.pth"))
            print(f" New best-by-F1 model saved with F1={best_f1:.4f}")

    print("\n--- Training Finished ---")
    print(f"Best F1 achieved: {best_f1:.4f}")

    best_model_path = os.path.join(OUTPUT_DIR, "best_model_f1.pth")
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for evaluation and inference.")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("\nWarning: No best model found. Using model from the last epoch.")

    if len(val_ds) > 0:
        calculate_and_plot_map_vs_iou(model, val_loader, device, OUTPUT_DIR)

    if len(val_ds) > 0:
        print("\n--- Visualizing on validation set ---")
        visualize_predictions(model, val_ds, device, num_samples=min(3, len(val_ds)), threshold=0.5, output_dir=OUTPUT_DIR)

    print("\n--- Starting Inference on Test Set ---")
    CONFIDENCE_THRESHOLD, MAX_TEST_SIZE, INF_BATCH = 0.95, 1024, 8

    test_dataset = TestCharacterDataset(test_img_dir, transform=tfm, max_size=MAX_TEST_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=INF_BATCH,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS_LOADER,
        worker_init_fn=seed_worker,
        generator=generator
    )

    model.eval()
    results = []
    print(f"\nRunning inference on {len(test_dataset)} test images...")

    autocast_enabled = (not DISABLE_AMP and device.type == 'cuda')
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
        for images, image_ids, scales, orig_sizes in tqdm(test_loader, desc="Generating CSV"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img_id, out, s, (H0, W0) in zip(image_ids, outputs, scales, orig_sizes):
                boxes, scores = out.get('boxes', torch.empty(0, 4)).cpu(), out.get('scores', torch.empty(0)).cpu()
                if boxes.numel() == 0:
                    continue

                if float(s) != 1.0:
                    boxes /= float(s)
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=float(W0))
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=float(H0))

                keep = scores >= CONFIDENCE_THRESHOLD
                if keep.any():
                    for b in boxes[keep]:
                        x1, y1, x2, y2 = map(float, b.tolist())
                        results.append({"image_id": img_id, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})

    output_csv_path = os.path.join(OUTPUT_DIR, 'output.csv')
    df = pd.DataFrame(results, columns=["image_id", "x", "y", "width", "height"])
    df.to_csv(output_csv_path, index=False)
    print(f"\n Predictions saved to {output_csv_path} (kept scores ≥ {CONFIDENCE_THRESHOLD}).")

if __name__ == '__main__':
    main()
