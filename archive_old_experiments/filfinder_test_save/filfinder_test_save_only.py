from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import aplpy
from astropy.wcs import WCS
import astropy.units as u
from fil_finder import FilFinder2D
import pandas as pd

import os
import matplotlib
matplotlib.use('Agg')
import cv2

# -----------------------------------------------------------------
# 入力設定
# -----------------------------------------------------------------
root_dir = "/mnt/sp4t/fujiwara.r"
fits_name = "HGBS_orionB_hires_column_density_map"
# fits_name = "HGBS_ic5146_hires_column_density_map"
# fits_name = "HGBS_polaris_hires_column_density_map"

# -----------------------------------------------------------------
# FilFinder パラメータ設定
# -----------------------------------------------------------------
distance = 436 * u.pc
BMAJ     = 18.0 * u.arcsec

# ic5146のみ負値をNaNに変換
REPLACE_NEG_WITH_NAN = (fits_name == "HGBS_ic5146_hires_column_density_map")

# create_mask パラメータ
CREATE_MASK_PARAMS = dict(
    verbose        = False,
    adapt_thresh   = 15 * u.pix,
    smooth_size    = 5  * u.pix,
    glob_thresh    = 1.88e21 * u.Unit('H2 cm-2'),
    size_thresh    = 5000 * u.pix**2,
    fill_hole_size = 100  * u.pix**2,
)
ANALYZE_SKELETONS_PARAMS = dict(
    branch_thresh  = 16 * u.pix,
    prune_criteria = 'length',
)
# 0.1pc以下のフィラメント全体を除去するための長さ閾値
LENGTH_THRESH_PIX = 16  # 0.1pc = 16pix

# -----------------------------------------------------------------
# 出力設定（visディレクトリのみ）
# -----------------------------------------------------------------
vis_dir = os.path.join(root_dir, fits_name, "vis")


def check_make_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


check_make_dir(vis_dir)

# ============================================================
# メイン処理
# ============================================================
hdu_tmp = fits.open(os.path.join(root_dir, fits_name + ".fits"))
hdu = hdu_tmp[0]
w = WCS(hdu)

if REPLACE_NEG_WITH_NAN:
    print("ic5146: replacing negative values with NaN...")
    hdu.data[hdu.data < 0] = np.nan

# --- FilFinder2D 実行（キャッシュなし、都度実行） ---
fil = FilFinder2D(hdu, distance=distance, beamwidth=BMAJ)
fil.preprocess_image(flatten_percent=95)

print("=== create_mask ===")
fil.create_mask(**CREATE_MASK_PARAMS)

print("=== medskel ===")
fil.medskel(verbose=False)

print("=== analyze_skeletons ===")
fil.analyze_skeletons(**ANALYZE_SKELETONS_PARAMS)

# 0.1pc以下のフィラメント全体を除去
before = len(fil.filaments)
fil.filaments = [f for f in fil.filaments if f.length().value >= LENGTH_THRESH_PIX]
print('Filament length filter: {} -> {} (removed {} filaments < 0.1pc)'.format(
    before, len(fil.filaments), before - len(fil.filaments)))

print("=== exec_rht ===")
fil.exec_rht()

print('Detected filaments:', len(fil.filaments))

fil_skeleton = fil.skeleton.astype("int")
fil_skeleton_longpath = fil.skeleton_longpath.astype("int")

# -----------------------------------------------------------------
# 可視化用に正規化画像を作成（vis出力にのみ使用、ファイル保存はしない）
# -----------------------------------------------------------------
img_raw = fil.image.value.copy().astype(np.float64)
img_raw[~np.isfinite(img_raw)] = np.nan
finite_mask = np.isfinite(img_raw)

img_log = np.where(finite_mask & (img_raw > 0), np.log10(img_raw), np.nan)
des_log = pd.Series(img_log[np.isfinite(img_log)].ravel()).describe(percentiles=[0.005, 0.995])
data_bak_ml = img_log.copy()
data_bak_ml -= des_log["0.5%"]
data_bak_ml /= ((des_log["99.5%"] - des_log["0.5%"]) / 255)
data_bak_ml = np.clip(data_bak_ml, 0, 255)
data_bak_ml[~finite_mask] = np.nan

data_bak = data_bak_ml

# -----------------------------------------------------------------
# vis 出力
# -----------------------------------------------------------------
nan_mask  = np.flipud(~np.isfinite(fil.image.value))
input_vis = np.flipud(np.where(np.isfinite(data_bak), data_bak, 0).astype("uint8"))
cv2.imwrite(os.path.join(vis_dir, fits_name + "_vis_input.png"), input_vis)

mask_vis = np.flipud(fil.mask.astype("uint8") * 255)
mask_vis[nan_mask] = 0
cv2.imwrite(os.path.join(vis_dir, fits_name + "_vis_mask.png"), mask_vis)

input_rgb = cv2.cvtColor(input_vis, cv2.COLOR_GRAY2BGR)
mask_flipud = np.flipud(fil.mask.astype("uint8"))
mask_flipud[nan_mask] = 0
input_rgb[mask_flipud == 1] = (0, 0, 255)
cv2.imwrite(os.path.join(vis_dir, fits_name + "_vis_input_mask_overlay.png"), input_rgb)

input_rgb2 = cv2.cvtColor(input_vis, cv2.COLOR_GRAY2BGR)
skel_flipud = np.flipud(fil_skeleton.astype("uint8"))
skel_flipud[nan_mask] = 0
input_rgb2[skel_flipud == 1] = (0, 0, 255)
cv2.imwrite(os.path.join(vis_dir, fits_name + "_vis_input_skeleton_overlay.png"), input_rgb2)

# skeleton (log / linear)
fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=1, vmax=100, stretch="log", cmap="Greys")
f.add_colorbar()
if fil_skeleton.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton, hdu.header), levels=[0.5],
                   colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_log.png"))
plt.close()

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=-100, vmax=100, cmap="Greys")
f.add_colorbar()
if fil_skeleton.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton, hdu.header), levels=[0.5],
                   colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_linear.png"))
plt.close()

# skeleton_longpath (log / linear)
fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=1, vmax=100, stretch="log", cmap="Greys")
f.add_colorbar()
if fil_skeleton_longpath.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton_longpath, hdu.header), levels=[0.5],
                   colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_longpath_log.png"))
plt.close()

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=-100, vmax=100, cmap="Greys")
f.add_colorbar()
if fil_skeleton_longpath.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton_longpath, hdu.header), levels=[0.5],
                   colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_longpath_linear.png"))
plt.close()

print("vis 出力完了:", vis_dir)
