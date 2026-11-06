"""
train / val データ分割(空間ブロック分割版)
1000枚ごとに tiles_xxx に保存
mmseg ann_file(train_fixed.txt / val_fixed.txt) 生成

- スケルトン(フィラメント)ラベルを1pxも含まないパッチは除外
- パッチのstride分割によるデータリークを防ぐため、
  雲全体を空間ブロックに区切り、ブロック単位でtrain/valに割り当てる
  (ブロック境界にはバッファを設け、リーク源になるパッチは除外)
- 雲全体のサイズ(shape)はFITSヘッダから自動取得する

出力:

split/
├── train/
│   ├── in/
│   │   ├── tiles_000/
│   │   └── ...
│   └── out/
│       ├── tiles_000/
│       └── ...
│
├── val/
│   ├── in/
│   └── out/
│
├── train_fixed.txt
└── val_fixed.txt
"""

import os
import random
import shutil
import re

import numpy as np
from PIL import Image
from astropy.io import fits

# ==================================================
# 設定
# ==================================================

root_dir = "/mnt/sp4t/fujiwara.r"

train_val_regions = [
    "HGBS_aquilaM2_hires_column_density_map",
    "HGBS_ic5146_hires_column_density_map",
]

train_ratio = 0.8
seed = 42

tiles_per_dir = 1000

# フィラメントクラスのラベル値
FILAMENT_LABEL_VALUE = 1
MIN_FILAMENT_PIXELS = 1

# 空間ブロック分割の設定
PATCH_SIZE = 256          # crop_data の size と一致させる
BLOCK_SIZE = 768         # パッチサイズより十分大きいブロック(4倍以上を推奨)
BUFFER_SIZE = 256         # ブロック境界のバッファ幅。PATCH_SIZE以上を推奨
VAL_FRACTION = 0.2

output_dir = os.path.join(
    root_dir,
    "split"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ==================================================
# util
# ==================================================

def check_make_dir(path):
    os.makedirs(
        path,
        exist_ok=True
    )


def has_filament_pixels(mask_path, label_value=FILAMENT_LABEL_VALUE, min_pixels=MIN_FILAMENT_PIXELS):
    try:
        mask = np.array(Image.open(mask_path))
    except Exception as e:
        print(f"WARNING: failed to read mask {mask_path}: {e}")
        return False

    if mask.ndim == 3:
        mask = mask[..., 0]

    n_filament_px = int((mask == label_value).sum())
    return n_filament_px >= min_pixels


def get_cloud_shape(region_dir_name):
    """
    root_dir 直下の {region_dir_name}.fits からマップの (height, width) を取得する。
    例: /mnt/sp4t/fujiwara.r/HGBS_aquilaM2_hires_column_density_map.fits
    """
    fits_path = os.path.join(root_dir, region_dir_name + ".fits")
    if not os.path.exists(fits_path):
        raise FileNotFoundError(
            f"FITSファイルが見つかりません: {fits_path}\n"
            f"雲全体のサイズが取得できないため空間ブロック分割ができません。"
        )

    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        # data は (height, width) の2次元配列を想定
        h, w = data.shape[-2], data.shape[-1]

    print(f"[{region_dir_name}] cloud shape from FITS = (height={h}, width={w})")
    return h, w


# ==================================================
# in/out収集
# ==================================================

def collect_pairs(region_dir):
    pairs = []
    in_base = os.path.join(region_dir, "in")
    out_base = os.path.join(region_dir, "out")

    if not os.path.isdir(in_base):
        return pairs

    for tiles_dir in sorted(os.listdir(in_base)):
        in_tiles = os.path.join(in_base, tiles_dir)
        out_tiles = os.path.join(out_base, tiles_dir)

        if not os.path.isdir(in_tiles):
            continue

        for fname in sorted(os.listdir(in_tiles)):
            if not fname.endswith("_in.png"):
                continue

            in_path = os.path.join(in_tiles, fname)
            out_path = os.path.join(out_tiles, fname.replace("_in.png", "_out.png"))

            if os.path.exists(out_path):
                m = re.search(r"_(\d+)-(\d+)_in\.png$", fname)
                if m:
                    x = int(m.group(1))
                    y = int(m.group(2))
                    pairs.append(
                        (
                            in_path,
                            out_path,
                            x,
                            y
                        )
                    )
            else:
                print(f"WARNING: Corresponding output file not found: {out_path}")

    return pairs


# ==================================================
# フィルタ:スケルトン(フィラメント)を含まないパッチを除外
# ==================================================

def filter_empty_patches(pairs, region_name=""):
    kept = []
    excluded = 0

    for in_path, out_path, x, y in pairs:
        if has_filament_pixels(out_path):
            kept.append((in_path, out_path, x, y))
        else:
            excluded += 1

    print(
        "[{}] filament-filter: kept={} excluded(background-only)={} "
        "(kept ratio={:.1f}%)".format(
            region_name,
            len(kept),
            excluded,
            100 * len(kept) / max(1, len(pairs))
        )
    )

    return kept, excluded


# ==================================================
# 空間ブロック分割:パッチのstride重複によるリークを防ぐ
# ==================================================

def spatial_block_split(
    pairs,
    cloud_shape,
    block_size=BLOCK_SIZE,
    val_fraction=VAL_FRACTION,
    seed=42,
    min_val_patches=100,   # ← 追加:valに最低限確保したいパッチ数
    max_retry=50,          # ← 追加:シードを変えて再試行する回数
):
    h, w = cloud_shape
    n_rows = int(np.ceil(h / block_size))
    n_cols = int(np.ceil(w / block_size))
    total_blocks = n_rows * n_cols

    def get_block_id(x, y):
        row = min(y // block_size, n_rows - 1)
        col = min(x // block_size, n_cols - 1)
        return row * n_cols + col

    for attempt in range(max_retry):
        rng = random.Random(seed + attempt)
        block_ids = list(range(total_blocks))
        rng.shuffle(block_ids)
        n_val_blocks = max(1, int(total_blocks * val_fraction))
        val_block_set = set(block_ids[:n_val_blocks])
        train_block_set = set(block_ids[n_val_blocks:])

        train_pairs, val_pairs, excluded_pairs = [], [], []

        for item in pairs:
            _, _, x, y = item
            left, right = x, x + PATCH_SIZE - 1
            top, bottom = y, y + PATCH_SIZE - 1
            left_block, right_block = left // block_size, right // block_size
            top_block, bottom_block = top // block_size, bottom // block_size

            if left_block != right_block or top_block != bottom_block:
                excluded_pairs.append(item)
                continue

            bid = get_block_id(x, y)
            if bid in train_block_set:
                train_pairs.append(item)
            else:
                val_pairs.append(item)

        if len(val_pairs) >= min_val_patches:
            print(f"  (attempt {attempt}: val={len(val_pairs)}枚 -> OK)")
            return train_pairs, val_pairs, excluded_pairs
        else:
            print(f"  (attempt {attempt}: val={len(val_pairs)}枚 < {min_val_patches}, seedを変えて再試行)")

    print(f"WARNING: {max_retry}回試してもval>={min_val_patches}を満たせませんでした。"
          f"最後の結果(val={len(val_pairs)}枚)をそのまま使います。")
    return train_pairs, val_pairs, excluded_pairs
# ==================================================
# 保存
# ==================================================

def save_split(
        base_dir,
        pairs,
        txt_name
):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    in_root = os.path.join(base_dir, "in")
    out_root = os.path.join(base_dir, "out")

    check_make_dir(in_root)
    check_make_dir(out_root)

    txt_path = os.path.join(output_dir, txt_name)

    with open(txt_path, "w") as txt:
        for idx, (in_src, out_src, _, _) in enumerate(pairs):
            tile_idx = idx // tiles_per_dir
            sub_dir = f"tiles_{tile_idx:03d}"

            in_sub = os.path.join(in_root, sub_dir)
            out_sub = os.path.join(out_root, sub_dir)

            check_make_dir(in_sub)
            check_make_dir(out_sub)

            in_name = os.path.basename(in_src)
            out_name = os.path.basename(out_src)

            in_dst = os.path.join(in_sub, in_name)
            out_dst = os.path.join(out_sub, out_name)

            if not os.path.exists(in_dst):
                os.symlink(in_src, in_dst)

            if not os.path.exists(out_dst):
                os.symlink(out_src, out_dst)

            stem = in_name.replace("_in.png", "")
            txt.write(f"{sub_dir}/{stem}\n")

    print(f"saved: {base_dir} {len(pairs)}")


# ==================================================
# split
# ==================================================

random.seed(seed)

train_pairs = []
val_pairs = []

total_excluded_empty = 0
total_excluded_buffer = 0

for region in train_val_regions:
    region_dir = os.path.join(root_dir, region)
    pairs = collect_pairs(region_dir)

    # ① フィラメントを含まないパッチを除外
    pairs, n_excluded_empty = filter_empty_patches(pairs, region_name=region)
    total_excluded_empty += n_excluded_empty

    # ② 雲全体のサイズをFITSヘッダから取得
    cloud_h, cloud_w = get_cloud_shape(region)

    # ③ 空間ブロック単位でtrain/valに分割(バッファ内は除外)
    region_train_pairs, region_val_pairs, region_excluded_buffer = spatial_block_split(
    pairs,
    cloud_shape=(cloud_h, cloud_w),
    block_size=BLOCK_SIZE,
    val_fraction=VAL_FRACTION,
    seed=seed,
    min_val_patches=100,   # IC5146・Aquilaどちらでも最低100枚を保証
    )
    total_excluded_buffer += len(region_excluded_buffer)

    train_pairs.extend(region_train_pairs)
    val_pairs.extend(region_val_pairs)

    print(
        "{} total(after filament-filter)={} train={} val={} excluded(buffer)={}".format(
            region,
            len(pairs),
            len(region_train_pairs),
            len(region_val_pairs),
            len(region_excluded_buffer)
        )
    )

random.shuffle(train_pairs)
random.shuffle(val_pairs)

print(f"\ntotal excluded (background-only patches): {total_excluded_empty}")
print(f"total excluded (block-border buffer patches): {total_excluded_buffer}")

# ==================================================
# 出力
# ==================================================

save_split(
    os.path.join(output_dir, "train"),
    train_pairs,
    "train_fixed.txt"
)

save_split(
    os.path.join(output_dir, "val"),
    val_pairs,
    "val_fixed.txt"
)

print("done")
