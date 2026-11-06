import os
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt   # ←追加

# =====================================================
# 設定
# =====================================================
root_dir = "/mnt/sp4t/fujiwara.r"

fits_name = "HGBS_aquilaM2_hires_column_density_map"
# fits_name = "HGBS_ic5146_hires_column_density_map"
# fits_name = "HGBS_orionB_hires_column_density_map"

vis_dir = os.path.join(root_dir, fits_name, "vis")

background_path = os.path.join(
    vis_dir,
    fits_name + "_vis_input.png"
)

skeleton_overlay_path = os.path.join(
    vis_dir,
    fits_name + "_vis_input_skeleton_overlay.png"
)

mask_overlay_path = os.path.join(
    vis_dir,
    fits_name + "_vis_input_mask_overlay.png"
)

# =====================================================
# 背景画像読み込み
# =====================================================
background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

if background is None:
    raise FileNotFoundError(background_path)

background = background.astype(np.float32) / 255.0
# ★追加（背景を暗くする）
background *= 0.9      # 0.6～0.8がおすすめ
# infernoカラーへ変換
rgb = cm.inferno(background)[..., :3]

rgb = (rgb * 255).astype(np.uint8)

# RGB → OpenCV(BGR)
rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# =====================================================
# Skeleton Overlay
# =====================================================

overlay = cv2.imread(skeleton_overlay_path)

if overlay is not None:

    # 赤画素抽出
    red = (
        (overlay[:, :, 2] > 200) &
        (overlay[:, :, 1] < 50) &
        (overlay[:, :, 0] < 50)
    )

    out = rgb.copy()

    # フィラメントを緑にする（OpenCVはBGR）
    out[red] = (0, 255, 0)

    # OpenCV(BGR)→Matplotlib(RGB)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    h, w = out.shape[:2]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    ax.imshow(out)

     # フィラメントの座標
    ys, xs = np.where(red)

    margin = 80   # 余白(pixel)

    xmin = max(xs.min() - margin, 0)
    xmax = min(xs.max() + margin, w)

    ymin = max(ys.min() - margin, 0)
    ymax = min(ys.max() + margin, h)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)

    # ===== ここがポイント =====
    # 表示範囲を狭くしてズームイン
    zoom = 0.85      # 0.85～0.95くらいがおすすめ

    x0 = w * (1 - zoom) / 2
    x1 = w - x0

    y0 = h * (1 - zoom) / 2
    y1 = h - y0

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    # =========================

    ax.axis("off")

    plt.savefig(
        os.path.join(
            vis_dir,
            fits_name + "_vis_input_skeleton_overlay_inferno.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

    print("saved skeleton image")

# =====================================================
# Mask Overlay
# =====================================================
overlay = cv2.imread(mask_overlay_path)

if overlay is not None:

    # 赤画素抽出
    red = (
        (overlay[:, :, 2] > 200) &
        (overlay[:, :, 1] < 50) &
        (overlay[:, :, 0] < 50)
    )

    out = rgb.copy()

    # フィラメントを緑にする（OpenCVはBGR）
    out[red] = (0, 255, 0)

    # OpenCV(BGR)→Matplotlib(RGB)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    h, w = out.shape[:2]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    ax.imshow(out)
    
     # フィラメントの座標
    ys, xs = np.where(red)

    margin = 80   # 余白(pixel)

    xmin = max(xs.min() - margin, 0)
    xmax = min(xs.max() + margin, w)

    ymin = max(ys.min() - margin, 0)
    ymax = min(ys.max() + margin, h)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)

    # ===== ここがポイント =====
    # 表示範囲を狭くしてズームイン
    zoom = 0.85      # 0.85～0.95くらいがおすすめ

    x0 = w * (1 - zoom) / 2
    x1 = w - x0

    y0 = h * (1 - zoom) / 2
    y1 = h - y0

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    # =========================

    ax.axis("off")

    plt.savefig(
        os.path.join(
            vis_dir,
            fits_name + "_vis_input_mask_overlay_inferno.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

    print("saved mask image")

print("Done.")
