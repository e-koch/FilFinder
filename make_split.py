"""
train / val データ分割
1000枚ごとに tiles_xxx に保存
mmseg ann_file(train_fixed.txt / val_fixed.txt) 生成

出力:

split/
├── train/
│   ├── in/
│   │   ├── tiles_000/
│   │   ├── tiles_001/
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


# ==================================================
# in/out収集
# ==================================================

def collect_pairs(region_dir):

    pairs = []

    in_base = os.path.join(
        region_dir,
        "in"
    )

    out_base = os.path.join(
        region_dir,
        "out"
    )

    for tiles_dir in sorted(
        os.listdir(in_base)
    ):

        in_tiles = os.path.join(
            in_base,
            tiles_dir
        )

        out_tiles = os.path.join(
            out_base,
            tiles_dir
        )

        if not os.path.isdir(
            in_tiles
        ):
            continue

        for fname in sorted(
            os.listdir(
                in_tiles
            )
        ):

            if not fname.endswith(
                "_in.png"
            ):
                continue

            in_path = os.path.join(
                in_tiles,
                fname
            )

            out_path = os.path.join(
                out_tiles,
                fname.replace(
                    "_in.png",
                    "_out.png"
                )
            )

            if os.path.exists(
                out_path
            ):

                pairs.append(
                    (
                        in_path,
                        out_path
                    )
                )

            else:

                print(
                    "WARNING:",
                    out_path
                )

    return pairs


# ==================================================
# 保存
# ==================================================

def save_split(
    base_dir,
    pairs,
    txt_name
):

    if os.path.exists(
        base_dir
    ):
        shutil.rmtree(
            base_dir
        )

    in_root = os.path.join(
        base_dir,
        "in"
    )

    out_root = os.path.join(
        base_dir,
        "out"
    )

    check_make_dir(
        in_root
    )

    check_make_dir(
        out_root
    )

    txt_path = os.path.join(
        output_dir,
        txt_name
    )

    with open(
        txt_path,
        "w"
    ) as txt:

        for idx, (
            in_src,
            out_src
        ) in enumerate(
            pairs
        ):

            tile_idx = (
                idx
                // tiles_per_dir
            )

            sub_dir = (
                "tiles_{:03d}"
                .format(
                    tile_idx
                )
            )

            in_sub = os.path.join(
                in_root,
                sub_dir
            )

            out_sub = os.path.join(
                out_root,
                sub_dir
            )

            check_make_dir(
                in_sub
            )

            check_make_dir(
                out_sub
            )

            in_name = (
                os.path.basename(
                    in_src
                )
            )

            out_name = (
                os.path.basename(
                    out_src
                )
            )

            in_dst = os.path.join(
                in_sub,
                in_name
            )

            out_dst = os.path.join(
                out_sub,
                out_name
            )

            if not os.path.exists(
                in_dst
            ):
                os.symlink(
                    in_src,
                    in_dst
                )

            if not os.path.exists(
                out_dst
            ):
                os.symlink(
                    out_src,
                    out_dst
                )

            # mmseg ann_file
            stem = (
                in_name
                .replace(
                    "_in.png",
                    ""
                )
            )

            txt.write(
                "{}/{}\n".format(
                    sub_dir,
                    stem
                )
            )

    print(
        "saved:",
        base_dir,
        len(
            pairs
        )
    )


# ==================================================
# split
# ==================================================

random.seed(
    seed
)

train_pairs = []
val_pairs = []

for region in train_val_regions:

    region_dir = os.path.join(
        root_dir,
        region
    )

    pairs = collect_pairs(
        region_dir
    )

    random.shuffle(
        pairs
    )

    n_train = int(
        len(
            pairs
        )
        *
        train_ratio
    )

    train_pairs.extend(
        pairs[
            :
            n_train
        ]
    )

    val_pairs.extend(
        pairs[
            n_train:
        ]
    )

    print(
        "{} total={} train={} val={}".format(
            region,
            len(
                pairs
            ),
            n_train,
            len(
                pairs
            )
            -
            n_train
        )
    )


random.shuffle(
    train_pairs
)

random.shuffle(
    val_pairs
)


# ==================================================
# 出力
# ==================================================

save_split(
    os.path.join(
        output_dir,
        "train"
    ),
    train_pairs,
    "train_fixed.txt"
)

save_split(
    os.path.join(
        output_dir,
        "val"
    ),
    val_pairs,
    "val_fixed.txt"
)

print(
    "done"
)
