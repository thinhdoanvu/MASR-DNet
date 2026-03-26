"""
K-Fold Cross-Validation Training for YOLO — Cardiomegaly
=========================================================
Cách dùng:
    python train_cardiomegaly.py | tee yolo8_12.log            	# mặc định 5-fold
    python train_cardiomegaly.py --folds 3  | tee yolo8_12.log 	# 3-fold

Yêu cầu cấu trúc dataset (giống cardiomegaly.yaml):
    ../Cardiomegaly/
        images/
            train/  *.jpg (hoặc .png)
            val/    *.jpg
        labels/
            train/  *.txt
            val/    *.txt
        cardiomegaly.yaml

Script sẽ:
  1. Gộp train+val thành pool chung
  2. Chia ngẫu nhiên thành k folds (stratified theo label class nếu muốn)
  3. Với mỗi fold: tạo yaml tạm → train → lưu metrics
  4. In bảng tổng hợp mean ± std cuối cùng
"""

import argparse
import shutil
import yaml
import random
import json
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

# ─── Cấu hình ────────────────────────────────────────────────────────────────

DATASET_YAML   = "../Cardiomegaly/cardiomegaly.yaml"
DATASET_ROOT   = Path(DATASET_YAML).parent

MODELS = [
    #"abl/v4",
    #"v8/yolov8m",
    #"v9/yolov9m",
    #"v10/yolov10l",
    #"11/yolo11l",
    "v12/yolov12l",
]

TRAIN_ARGS = dict(
    epochs    = 200,
    batch     = 32,
    imgsz     = 960,
    patience  = 50,
    
    # danh cho v4-my method
    workers = 64,
    optimizer = "SGD",
    lr0 = 1e-4,
    weight_decay = 0.05,
    warmup_epochs = 10.0,
    cos_lr = True,
    label_smoothing = 0.1,
    close_mosaic = 15,
    box = 12.0,
    cls = 0.5,
)

SEED = 42

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_original_yaml(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)

def collect_samples(dataset_root: Path, splits=("train", "val")) -> list[Path]:
    """Gộp tất cả ảnh từ các split, trả về list Path."""
    images = []
    for split in splits:
        img_dir = dataset_root / "images" / split
        if img_dir.exists():
            images.extend(sorted(img_dir.glob("*.[jp][pn]g")))
            images.extend(sorted(img_dir.glob("*.jpeg")))
    # Dedup + sort
    seen = set()
    unique = []
    for p in images:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique

def make_folds(samples: list[Path], k: int, seed: int) -> list[tuple]:
    """
    Chia samples thành k folds.
    Trả về list of (train_paths, val_paths).
    """
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k - 1 else len(shuffled)
        folds.append(shuffled[start:end])

    splits = []
    for i in range(k):
        val_paths   = folds[i]
        train_paths = [p for j, fold in enumerate(folds) if j != i for p in fold]
        splits.append((train_paths, val_paths))
    return splits

def write_fold_data(fold_idx: int, train_paths: list[Path],
                    val_paths: list[Path], dataset_root: Path,
                    tmp_root: Path) -> Path:
    """
    Tạo thư mục tạm cho fold này với symlink hoặc copy nhẹ (chỉ .txt label).
    Tạo fold_k.yaml trỏ đến thư mục tạm.
    """
    fold_dir = tmp_root / f"fold_{fold_idx}"

    for split_name, paths in [("train", train_paths), ("val", val_paths)]:
        img_out = fold_dir / "images" / split_name
        lbl_out = fold_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in paths:
            # Symlink ảnh (tiết kiệm disk)
            dst_img = img_out / img_path.name
            if not dst_img.exists():
                dst_img.symlink_to(img_path.resolve())

            # Tìm label tương ứng
            for split_src in ("train", "val"):
                lbl_src = (dataset_root / "labels" / split_src
                           / img_path.with_suffix(".txt").name)
                if lbl_src.exists():
                    dst_lbl = lbl_out / lbl_src.name
                    if not dst_lbl.exists():
                        dst_lbl.symlink_to(lbl_src.resolve())
                    break

    # Đọc nc và names từ yaml gốc
    orig = load_original_yaml(DATASET_YAML)
    fold_yaml = {
        "path"  : str(fold_dir.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : orig.get("nc", 1),
        "names" : orig.get("names", ["cardiomegaly"]),
    }
    yaml_path = tmp_root / f"fold_{fold_idx}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(fold_yaml, f, default_flow_style=False)
    return yaml_path

def extract_metrics(results) -> dict:
    """Lấy metrics quan trọng từ kết quả train."""
    if results is None:
        return {}
    try:
        # results.results_dict chứa metrics cuối epoch
        rd = results.results_dict
        return {
            "mAP50"    : float(rd.get("metrics/mAP50(B)", 0)),
            "mAP50-95" : float(rd.get("metrics/mAP50-95(B)", 0)),
            "precision": float(rd.get("metrics/precision(B)", 0)),
            "recall"   : float(rd.get("metrics/recall(B)", 0)),
        }
    except Exception as e:
        print(f"  [warn] Không lấy được metrics: {e}")
        return {}

def summarize(all_results: dict):
    """In bảng tổng hợp mean ± std cho tất cả model."""
    import statistics

    print("\n" + "=" * 72)
    print("  TỔNG HỢP K-FOLD")
    print("=" * 72)
    print(f"  {'Model':<22} {'mAP50':>10} {'mAP50-95':>12} {'Precision':>12} {'Recall':>10}")
    print("-" * 72)

    for model_name, fold_metrics in all_results.items():
        # fold_metrics: list of dict per fold
        def agg(key):
            vals = [m[key] for m in fold_metrics if key in m]
            if not vals:
                return "  N/A"
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0
            return f"{mean:.4f}±{std:.4f}"

        print(f"  {model_name:<22} {agg('mAP50'):>10} {agg('mAP50-95'):>12} "
              f"{agg('precision'):>12} {agg('recall'):>10}")

    print("=" * 72)

# ─── Main ────────────────────────────────────────────────────────────────────

SPLIT_META_FILE = "kfold_tmp/split_meta.json"

def save_split_meta(k: int, seed: int, n_samples: int, fold_splits: list):
    """Lưu metadata của split để validate lần sau."""
    meta = {
        "k"        : k,
        "seed"     : seed,
        "n_samples": n_samples,
        # Lưu tên file (không phải path đầy đủ) để kiểm tra tính nhất quán
        "folds"    : [
            {"train": [p.name for p in tr], "val": [p.name for p in vl]}
            for tr, vl in fold_splits
        ],
    }
    with open(SPLIT_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def load_and_validate_split(k: int, seed: int, n_samples: int) -> bool:
    """
    Kiểm tra split cũ có hợp lệ không.
    Trả về True nếu có thể tái sử dụng, False nếu cần tạo lại.
    """
    meta_path = Path(SPLIT_META_FILE)
    if not meta_path.exists():
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    if meta["k"] != k:
        print(f"  [!] Split cũ có k={meta['k']}, hiện tại k={k} → tạo lại.")
        return False
    if meta["seed"] != seed:
        print(f"  [!] Split cũ có seed={meta['seed']}, hiện tại seed={seed} → tạo lại.")
        return False
    if meta["n_samples"] != n_samples:
        print(f"  [!] Dataset thay đổi ({meta['n_samples']} → {n_samples} ảnh) → tạo lại.")
        return False

    # Kiểm tra các fold_k.yaml đều tồn tại
    tmp_root = Path("kfold_tmp")
    missing = [f"fold_{i+1}.yaml" for i in range(k)
               if not (tmp_root / f"fold_{i+1}.yaml").exists()]
    if missing:
        print(f"  [!] Thiếu file: {missing} → tạo lại.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5,
                        help="Số folds (3 hoặc 5, mặc định 5)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Chỉ train một số model, VD: v8/yolov8m 11/yolo11l")
    parser.add_argument("--force-resplit", action="store_true",
                        help="Bắt buộc tạo lại split dù đã tồn tại")
    args = parser.parse_args()

    k      = args.folds
    models = args.models or MODELS

    print(f"\n{'='*60}")
    print(f"  K-Fold Cross-Validation  |  k={k}  |  seed={SEED}")
    print(f"  Dataset: {DATASET_YAML}")
    print(f"{'='*60}\n")

    # Chuẩn bị samples
    samples = collect_samples(DATASET_ROOT)
    print(f"  Tổng số ảnh tìm được: {len(samples)}")
    assert len(samples) >= k, f"Cần ít nhất {k} ảnh để chia {k} folds!"

    tmp_root = Path("kfold_tmp")

    # ── Kiểm tra tái sử dụng split ────────────────────────────────────────
    can_reuse = (not args.force_resplit
                 and load_and_validate_split(k, SEED, len(samples)))

    if can_reuse:
        print("  ✓ Tìm thấy split hợp lệ → tái sử dụng (bỏ qua bước chia fold).")
        print("    (Dùng --force-resplit để tạo lại từ đầu)\n")
        fold_yamls = [tmp_root / f"fold_{i+1}.yaml" for i in range(k)]
    else:
        reason = "flag --force-resplit" if args.force_resplit else "không tìm thấy split hợp lệ"
        print(f"  → Tạo split mới ({reason}).")

        # Xóa thư mục cũ nếu có
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(exist_ok=True)

        # Chia folds
        fold_splits = make_folds(samples, k=k, seed=SEED)
        for i, (tr, vl) in enumerate(fold_splits):
            print(f"  Fold {i+1}: train={len(tr)}, val={len(vl)}")

        # Lưu metadata để validate lần sau
        save_split_meta(k, SEED, len(samples), fold_splits)

        # Chuẩn bị dữ liệu cho mỗi fold
        fold_yamls = []
        for i, (train_paths, val_paths) in enumerate(fold_splits):
            yaml_path = write_fold_data(i + 1, train_paths, val_paths,
                                        DATASET_ROOT, tmp_root)
            fold_yamls.append(yaml_path)

    # ── Training loop ──────────────────────────────────────────────────────
    all_results = defaultdict(list)  # model_name -> [metrics_fold1, ...]

    for model_cfg in models:
        model_name = model_cfg.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_cfg}")
        print(f"{'='*60}")

        for fold_idx, fold_yaml in enumerate(fold_yamls, start=1):
            print(f"\n  ── Fold {fold_idx}/{k} ──")
            run_name = f"{model_name}_fold{fold_idx}"

            try:
                yolo_model = YOLO(f"cfg/models/{model_cfg}.yaml")
                results = yolo_model.train(
                    data    = str(fold_yaml),
                    name    = run_name,
                    **TRAIN_ARGS,
                )
                metrics = extract_metrics(results)
                print(f"  Fold {fold_idx} metrics: {metrics}")
            except Exception as e:
                print(f"  [ERROR] Fold {fold_idx} thất bại: {e}")
                metrics = {}

            all_results[model_name].append(metrics)

            # Lưu metrics tạm sau mỗi fold (phòng crash)
            checkpoint_path = tmp_root / "results_checkpoint.json"
            with open(checkpoint_path, "w") as f:
                json.dump(dict(all_results), f, indent=2)
            print(f"  Checkpoint lưu tại: {checkpoint_path}")

    # ── Tổng hợp ──────────────────────────────────────────────────────────
    summarize(dict(all_results))

    # Lưu kết quả cuối
    final_path = Path("kfold_results_final.json")
    with open(final_path, "w") as f:
        json.dump(dict(all_results), f, indent=2)
    print(f"\n  Kết quả đầy đủ lưu tại: {final_path}")

    # Dọn thư mục tạm (comment lại nếu muốn giữ symlink để debug)
    # shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()