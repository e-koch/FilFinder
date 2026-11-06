import mmcv
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import aplpy
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import fil_finder
from fil_finder import FilFinder2D, Filament2D
import tqdm
import copy
import random
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.colorbar import ColorbarBase
import pandas as pd

import os
import matplotlib
matplotlib.use('Agg')
import pickle
import time

import cv2
from tifffile import imwrite

# -----------------------------------------------------------------
# 入力設定
# -----------------------------------------------------------------
root_dir = "/mnt/sp4t/fujiwara.r"
fits_name = "HGBS_ic5146_hires_column_density_map"

save_ext = "png"

# -----------------------------------------------------------------
# FilFinder パラメータ設定
# -----------------------------------------------------------------
distance = 460 * u.pc
BMAJ     = 18.0 * u.arcsec

# ic5146のみ負値をNaNに変換
REPLACE_NEG_WITH_NAN = (fits_name == "HGBS_ic5146_hires_column_density_map")

# create_mask パラメータ
# glob_thresh: Av=4mag相当 = 3.76e21 cm^-2
#   指導教員コメント: 従来の100は小さすぎるため物理的に意味のある閾値に変更
#   8mag相当(~7.52e21)にしてもよいが、まず4mag相当で試す
# min_length: 0.1pc相当 = 16pix (distance=436pc, pixel=3arcsec/pix)
#   0.1pc以下の短い構造は除去する
CREATE_MASK_PARAMS = dict(
    verbose        = True,
    adapt_thresh   = 15 * u.pix,
    smooth_size    = 5  * u.pix,
    # glob_thresh: Av=4mag相当 = 3.76e21 H2/cm2
    # image.unitが 'H2 cm-2' なので同じ単位で渡す
    glob_thresh = 1.88e21 * u.Unit('H2 cm-2'),
    size_thresh    = 5000 * u.pix**2,
    fill_hole_size = 100  * u.pix**2,
)
ANALYZE_SKELETONS_PARAMS = dict(
    # 0.1pc = 16pix以下の短いブランチを除去
    branch_thresh  = 16 * u.pix,
    prune_criteria = 'length',
)
# 0.1pc以下のフィラメント全体を除去するための長さ閾値
LENGTH_THRESH_PIX = 16  # 0.1pc = 16pix

# -----------------------------------------------------------------
# タイル切り出し設定
# -----------------------------------------------------------------
size          = 256
step_rate     = 0.0625
tiles_per_dir = 1000

cache_dir   = os.path.join(root_dir, fits_name, "cache")
fullimg_dir = os.path.join(root_dir, fits_name, "full")
vis_dir     = os.path.join(root_dir, fits_name, "vis")


def check_make_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def load_pkl(fpath):
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    return None


def save_pkl(fpath, val):
    with open(fpath, 'wb') as f:
        pickle.dump(val, f)


def crop_data(nda, nda_check_nan, suf):
    base_dir = os.path.join(root_dir, fits_name, suf)
    w, h = nda.shape[1::-1]
    print('canvas w,h =', w, h)
    step = int(step_rate * size)
    tile_count = 0
    y = 0
    while (y + size <= h):
        x = 0
        while (x + size <= w):
            cropped       = nda[y:y+size, x:x+size]
            cropped_check = nda_check_nan[y:y+size, x:x+size]
            o_name = fits_name + "_" + str(x).zfill(5) + "-" + str(y).zfill(5)
            x += step
            if np.isnan(cropped_check).any():
                continue
            sub_dir = os.path.join(base_dir, "tiles_{:03d}".format(tile_count // tiles_per_dir))
            check_make_dir(sub_dir)
            save_pkl(os.path.join(sub_dir, o_name + "_" + suf + ".pkl"), cropped)
            cv2.imwrite(os.path.join(sub_dir, o_name + "_" + suf + ".png"), cropped.astype("int"))
            imwrite(os.path.join(sub_dir, o_name + "_" + suf + ".tif"), cropped.astype(np.float32))
            tile_count += 1
        y += step
    print('Total tiles saved ({}): {}'.format(suf, tile_count))


# ============================================================
# メイン処理
# ============================================================
for d in [cache_dir, fullimg_dir, vis_dir]:
    check_make_dir(d)

start = time.perf_counter()
hdu_tmp = fits.open(os.path.join(root_dir, fits_name + ".fits"))
print('hdu tmp:', hdu_tmp)
hdu = hdu_tmp[0]
print('load hdu_tmp : {:.2f}s'.format(time.perf_counter() - start))

w = WCS(hdu)

fig = plt.figure(figsize=(8, 8))
f = aplpy.FITSFigure(hdu, slices=[0], convention='wells', figure=fig)
f.show_colorscale(cmap="Greys", aspect="equal")
f.save("filfinder_" + fits_name + "_1." + save_ext)
plt.close()

print('FITS header:\n', hdu.header)

if REPLACE_NEG_WITH_NAN:
    print("ic5146: replacing negative values with NaN...")
    hdu.data[hdu.data < 0] = np.nan

# FilFinder2D
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D.pkl")
start = time.perf_counter()
fil = load_pkl(pkl_name)
if fil is None:
    fil = FilFinder2D(hdu, distance=distance, beamwidth=BMAJ)
    save_pkl(pkl_name, fil)
print('FilFinder2D : {:.2f}s'.format(time.perf_counter() - start))

# preprocess_image
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-preprocess_image.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    fil.preprocess_image(flatten_percent=95)
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
print('preprocess_image : {:.2f}s'.format(time.perf_counter() - start))

flat = fil.flat_img
finite_count = int(np.isfinite(np.array(flat.value if hasattr(flat, 'value') else flat)).sum())
print('flat_img finite count: {} / {} ({:.1f}%)'.format(finite_count, flat.size, 100*finite_count/flat.size))

cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_preprocess_image.png"), fil.image.value.astype("int"))
data_des = pd.DataFrame(pd.Series(fil.image.value.astype(np.float64).ravel()).describe())
print(data_des)

# 正規化
img_raw = fil.image.value.copy().astype(np.float64)
img_raw[~np.isfinite(img_raw)] = np.nan
finite_mask = np.isfinite(img_raw)

img_log = np.where(finite_mask & (img_raw > 0), np.log10(img_raw), np.nan)
des_log = pd.DataFrame(pd.Series(img_log[np.isfinite(img_log)].ravel()).describe(percentiles=[0.005, 0.995]))
data_bak_ml = img_log.copy()
data_bak_ml -= des_log[0]["0.5%"]
data_bak_ml /= ((des_log[0]["99.5%"] - des_log[0]["0.5%"]) / 255)
data_bak_ml = np.clip(data_bak_ml, 0, 255)
data_bak_ml[~finite_mask] = np.nan

des_lin = pd.DataFrame(pd.Series(img_raw[finite_mask].ravel()).describe(
    percentiles=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]))
data_bak_ff = img_raw.copy()
data_bak_ff -= des_lin[0]["0.5%"]
data_bak_ff /= ((des_lin[0]["99.5%"] - des_lin[0]["0.5%"]) / 255)
data_bak_ff = np.clip(data_bak_ff, 0, 255)
data_bak_ff[~finite_mask] = np.nan

data_bak = data_bak_ml

cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_input_full_ml.png"), data_bak_ml.astype("uint8"))
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_input_full_ff.png"), data_bak_ff.astype("uint8"))
mmcv.image.use_backend('tifffile')
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_input_full_ml.tif"), data_bak_ml.astype(np.float32))

# create_mask
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-create_mask.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    plt.figure(figsize=(8, 8))
    fil.create_mask(**CREATE_MASK_PARAMS)
    plt.savefig("filfinder_" + fits_name + "_2." + save_ext)
    plt.close()
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
print('create_mask : {:.2f}s'.format(time.perf_counter() - start))

# medskel
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-medskel.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    plt.figure(figsize=(8, 8))
    fil.medskel(verbose=True, save_png=True)
    plt.savefig("filfinder_" + fits_name + "_3." + save_ext)
    plt.close()
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
print('medskel : {:.2f}s'.format(time.perf_counter() - start))

# analyze_skeletons（0.1pc=16pix以下のブランチを除去）
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-analyze_skeletons.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    fil.analyze_skeletons(**ANALYZE_SKELETONS_PARAMS)
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
print('analyze_skeletons : {:.2f}s'.format(time.perf_counter() - start))

# 0.1pc=16pix以下のフィラメント全体を除去（branch_threshは枝のみなので別途フィルタリング）
# fil.lengths() でフィラメントの長さ（pix）を取得してフィルタリング
before = len(fil.filaments)
fil.filaments = [f for f in fil.filaments if f.length().value >= LENGTH_THRESH_PIX]
print('Filament length filter: {} -> {} (removed {} filaments < 0.1pc)'.format(
    before, len(fil.filaments), before - len(fil.filaments)))

# exec_rht
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-exec_rht.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    fil.exec_rht()
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
print('exec_rht : {:.2f}s'.format(time.perf_counter() - start))

print('Detected filaments:', len(fil.filaments))

# ラベル保存
start = time.perf_counter()
fil_skeleton     = fil.skeleton.astype("int")
fil_skeleton_vis = (fil_skeleton * 255).astype("uint8")
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_label_01.png"), fil_skeleton)
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_label_255.png"), fil_skeleton_vis)
print(pd.DataFrame(pd.Series(fil_skeleton.ravel()).describe()))
print('label images saved : {:.2f}s'.format(time.perf_counter() - start))

# タイル切り出し
crop_data(data_bak, data_bak, "in")
crop_data(fil_skeleton, data_bak, "out")

# 可視化
start = time.perf_counter()
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

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=1, vmax=100, stretch="log", cmap="Greys")
f.add_colorbar()
if fil_skeleton.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton, hdu.header),levels=[0.5], colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_log.png"))
plt.close()

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=-100, vmax=100, cmap="Greys")
f.add_colorbar()
if fil_skeleton.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton, hdu.header), levels=[0.5],colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_linear.png"))
plt.close()

print('skeleton visualization saved : {:.2f}s'.format(time.perf_counter() - start))

start = time.perf_counter()
fil_skeleton_longpath = fil.skeleton_longpath.astype("int")

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=1, vmax=100, stretch="log", cmap="Greys")
f.add_colorbar()
if fil_skeleton_longpath.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton_longpath, hdu.header), levels=[0.5],colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_longpath_log.png"))
plt.close()

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=-100, vmax=100, cmap="Greys")
f.add_colorbar()
if fil_skeleton_longpath.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton_longpath, hdu.header), levels=[0.5],colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_longpath_linear.png"))
plt.close()

print('skeleton_longpath visualization saved : {:.2f}s'.format(time.perf_counter() - start))
