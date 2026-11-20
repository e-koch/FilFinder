
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FilFinder Loose版
- 5領域に対応
- Loose Av / 領域別 LEN_CUT_PC
- 分岐を含む全Skeletonを保存
- 元画像グレースケール + 赤線PNG
- FITSは1pixelの中心線だけ保存
- PNG表示用のみ赤線を7pixelに拡張
"""

import os
import csv
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

from scipy.ndimage import binary_dilation
from fil_finder import FilFinder2D

warnings.filterwarnings("ignore")

# ============================================================
# 設定
# ============================================================

root_dir = "/mnt/sp4t/fujiwara.r"

# 環境変数REGIONで切替可能
REGION = os.environ.get("REGION", "aquila").lower()

AV_UNIT = 0.94e21

ADAPT_PC = 0.1
SMOOTH_PC = 0.1
SIZE_THRESH_PC2 = 0.1
FILL_HOLE_SIZE_PC2 = 0.1

SKEL_THRESH_PC = 0.2
BRANCH_PC = 0.2

# PNG上だけ中心線を太く表示
VIS_LINE_WIDTH_PX = 7

# ============================================================
# Loose領域設定
# ============================================================

TARGETS = {
    "aquila": {
        "fits_name": "HGBS_aquilaM2_hires_column_density_map",
        "dist_pc": 440.0,
        "beam_arcsec": 18.0,
        "av": 6,
        "len_cut_pc": 0.15,
    },
    "ic5146": {
        "fits_name": "HGBS_ic5146_hires_column_density_map",
        "dist_pc": 600.0,
        "beam_arcsec": 18.0,
        "av": 2,
        "len_cut_pc": 0.15,
    },
    "orionb": {
        "fits_name": "HGBS_orionB_hires_column_density_map",
        "dist_pc": 400.0,
        "beam_arcsec": 18.0,
        "av": 3,
        "len_cut_pc": 0.30,
    },
    "taurus": {
        "fits_name": "HGBS_tauN3_hires_column_density_map",
        "dist_pc": 140.0,
        "beam_arcsec": 18.0,
        "av": 3,
        "len_cut_pc": 0.30,
    },
    "polaris": {
        "fits_name": "HGBS_polaris_hires_column_density_map",
        "dist_pc": 150.0,
        "beam_arcsec": 18.0,
        "av": 1,
        "len_cut_pc": 0.10,
    },
}


# ============================================================
# FITS読み込み
# ============================================================

def find_fits_path(fits_name):
    candidates = [
        os.path.join(root_dir, fits_name + ".fits"),
        os.path.join(root_dir, fits_name, fits_name + ".fits"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "入力FITSが見つかりません: " + str(candidates)
    )


def load_column_density_2d(fpath):
    with fits.open(fpath, memmap=True) as hdul:
        header = hdul[0].header.copy()
        data = np.squeeze(
            np.asarray(hdul[0].data, dtype=np.float32)
        ).copy()

    if data.ndim != 2:
        raise ValueError(
            f"2次元FITSが必要です: {data.shape}"
        )

    data[~np.isfinite(data) | (data <= 0)] = np.nan

    wcs = WCS(header).celestial
    hdr2d = wcs.to_header()
    hdr2d["NAXIS"] = 2
    hdr2d["NAXIS1"] = data.shape[1]
    hdr2d["NAXIS2"] = data.shape[0]

    return data, hdr2d


def get_pc_per_pixel(hdr2d, dist_pc):
    wcs = WCS(hdr2d).celestial

    pixel_scale_deg = float(
        np.mean(proj_plane_pixel_scales(wcs)[:2])
    )

    pc_per_pixel = (
        np.deg2rad(pixel_scale_deg) * dist_pc
    )

    if not np.isfinite(pc_per_pixel) or pc_per_pixel <= 0:
        raise ValueError(
            f"pc_per_pixelが不正です: {pc_per_pixel}"
        )

    return pc_per_pixel


# ============================================================
# FilFinder実行
# ============================================================

def run_filfinder(
    data2d,
    hdr2d,
    beam_arcsec,
    dist_pc,
    glob_thresh,
):

    pc_per_pixel = get_pc_per_pixel(hdr2d, dist_pc)

    smooth_size = (
        max(round(SMOOTH_PC / pc_per_pixel), 1.5) * u.pix
    )

    adapt_thresh = (
        max(round(ADAPT_PC / pc_per_pixel), 3.0) * u.pix
    )

    fil = FilFinder2D(
        data2d,
        header=hdr2d,
        beamwidth=beam_arcsec * u.arcsec,
        distance=dist_pc * u.pc,
    )

    fil.preprocess_image(skip_flatten=True)

    fil.create_mask(
        glob_thresh=glob_thresh,
        smooth_size=smooth_size,
        adapt_thresh=adapt_thresh,
        size_thresh=SIZE_THRESH_PC2 * u.pc**2,
        fill_hole_size=FILL_HOLE_SIZE_PC2 * u.pc**2,
        verbose=False,
    )

    fil.medskel(verbose=False)

    fil.analyze_skeletons(
        skel_thresh=SKEL_THRESH_PC * u.pc,
        branch_thresh=BRANCH_PC * u.pc,
    )

    return fil


# ============================================================
# フィラメントの長さ
# ============================================================

def fil_length_pc(component, pc_per_pixel):
    try:
        length = component.length(u.pc)
        value = float(length.to_value(u.pc))

        if np.isfinite(value) and value > 0:
            return value

    except Exception:
        pass

    try:
        rows, cols = component.longpath_pixel_coords
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)

        if len(rows) < 2:
            return 0.0

        dr = np.diff(rows)
        dc = np.diff(cols)

        return float(
            np.hypot(dr, dc).sum() * pc_per_pixel
        )

    except Exception:
        return 0.0


# ============================================================
# 分岐を含む全Skeletonの保存
# ============================================================

def make_filtered_skeleton(fil, keep):

    filtered = np.zeros_like(
        fil.skeleton,
        dtype=np.uint8,
    )

    if len(keep) != len(fil.filaments):
        raise ValueError("keepとfilamentsの本数が一致しません")

    for component, accepted in zip(
        fil.filaments, keep
    ):
        if not accepted:
            continue

        # 重要：longpathではなく全Skeleton座標
        rows, cols = component.pixel_coords

        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)

        valid = (
            (rows >= 0)
            & (cols >= 0)
            & (rows < filtered.shape[0])
            & (cols < filtered.shape[1])
        )

        filtered[rows[valid], cols[valid]] = 1

    before = int(np.count_nonzero(filtered))

    original = fil.skeleton.astype(bool)
    outside = int(
        np.count_nonzero(filtered.astype(bool) & ~original)
    )

    print("AND前:", before)
    print("元Skeleton外の画素:", outside)

    filtered &= original.astype(np.uint8)

    after = int(np.count_nonzero(filtered))
    print("AND後:", after)

    if after == 0 and sum(keep) > 0:
        raise RuntimeError(
            "採用フィラメントが存在しますがSkeletonが空です"
        )

    return filtered


# ============================================================
# グレースケール + 赤線PNG
# ============================================================

def plot_overlay(data2d, skeleton, name, save):

    valid = np.isfinite(data2d) & (data2d > 0)

    if not valid.any():
        raise ValueError("可視化できる有効画素がありません")

    gray = np.full(
        data2d.shape,
        np.nan,
        dtype=np.float32,
    )

    gray[valid] = np.log10(data2d[valid])

    vmin, vmax = np.percentile(
        gray[valid], [1.0, 99.5]
    )

    if vmax <= vmin:
        vmax = vmin + 1e-6

    # PNG表示専用の拡張
    skeleton_vis = binary_dilation(
        skeleton.astype(bool),
        structure=np.ones(
            (VIS_LINE_WIDTH_PX, VIS_LINE_WIDTH_PX),
            dtype=bool,
        ),
    )

    overlay = np.zeros(
        (*skeleton.shape, 4),
        dtype=np.float32,
    )

    overlay[skeleton_vis] = [1.0, 0.0, 0.0, 1.0]

    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("black")

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(
        gray,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    ax.imshow(
        overlay,
        origin="lower",
        interpolation="nearest",
    )

    ax.set_title(f"{name} FilFinder Loose")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")

    fig.tight_layout()
    fig.savefig(save, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# メイン処理
# ============================================================

def main():

    if REGION not in TARGETS:
        raise ValueError(f"未対応の領域: {REGION}")

    cfg = TARGETS[REGION]
    fits_name = cfg["fits_name"]
    fits_path = find_fits_path(fits_name)

    base_dir = os.path.join(root_dir, fits_name)
    full_dir = os.path.join(base_dir, "full_loose_fixed")
    vis_dir = os.path.join(base_dir, "vis_loose_fixed")

    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"処理開始: {REGION} / Loose")

    data, hdr2d = load_column_density_2d(fits_path)

    glob_thresh = cfg["av"] * AV_UNIT
    len_cut_pc = cfg["len_cut_pc"]

    print("距離:", cfg["dist_pc"], "pc")
    print("Av:", cfg["av"])
    print("glob_thresh:", glob_thresh)
    print("LEN_CUT_PC:", len_cut_pc)

    fil = run_filfinder(
        data,
        hdr2d,
        cfg["beam_arcsec"],
        cfg["dist_pc"],
        glob_thresh,
    )

    pc_per_pixel = get_pc_per_pixel(
        hdr2d, cfg["dist_pc"]
    )

    lengths = [
        fil_length_pc(component, pc_per_pixel)
        for component in fil.filaments
    ]

    keep = [
        length >= len_cut_pc
        for length in lengths
    ]

    filtered_skeleton = make_filtered_skeleton(
        fil, keep
    )

    # FITSは1pixelの中心線のみ保存
    fits_path_out = os.path.join(
        full_dir, f"skeleton_{REGION}.fits"
    )

    fits.writeto(
        fits_path_out,
        filtered_skeleton.astype(np.uint8),
        hdr2d,
        overwrite=True,
    )

    csv_path = os.path.join(
        full_dir, f"filaments_{REGION}.csv"
    )

    with open(
        csv_path, "w",
        newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "fil_id", "length_pc", "robust"
        ])

        for i, (length, accepted) in enumerate(
            zip(lengths, keep)
        ):
            writer.writerow([
                i, f"{length:.6f}", int(accepted)
            ])

    png_path = os.path.join(
        vis_dir, f"overlay_{REGION}.png"
    )

    plot_overlay(
        data,
        filtered_skeleton,
        REGION,
        png_path,
    )

    print(
        f"検出 {len(fil.filaments)} 本 / "
        f"採用 {sum(keep)} 本"
    )
    print("Skeleton FITS:", fits_path_out)
    print("赤線PNG:", png_path)


if __name__ == "__main__":
    main()

