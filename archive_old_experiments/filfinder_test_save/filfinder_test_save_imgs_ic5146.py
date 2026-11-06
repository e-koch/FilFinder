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
# データセット固有の設定
#  ic5146のみ負値→NaN（background mask処理と統合）
#  全リージョン共通: flatten_percent=95
# -----------------------------------------------------------------
REPLACE_NEG_WITH_NAN = (fits_name == "HGBS_ic5146_hires_column_density_map")
FLATTEN_PERCENT = 95

# -----------------------------------------------------------------
# FilFinder create_mask パラメータ（3リージョン共通）
# -----------------------------------------------------------------
CREATE_MASK_PARAMS = dict(
    verbose        = True,
    adapt_thresh   = 15 * u.pix,
    smooth_size    = 5  * u.pix,
    glob_thresh    = 100,
    size_thresh    = 5000 * u.pix**2,
    fill_hole_size = 100  * u.pix**2,
)

# ----------------------------
# ユーティリティ関数
# ----------------------------
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

# ----------------------------
# タイル切り出し設定
# ----------------------------
size = 256
step_rate = 0.0625  # stride = 16px
tiles_per_dir = 1000

cache_dir   = os.path.join(root_dir, fits_name, "cache")
fullimg_dir = os.path.join(root_dir, fits_name, "full")
vis_dir     = os.path.join(root_dir, fits_name, "vis")


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
            cropped = nda[y:y + size, x:x + size]
            cropped_check = nda_check_nan[y:y + size, x:x + size]
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

# aplpy 可視化
fig = plt.figure(figsize=(8, 8))
f = aplpy.FITSFigure(hdu, slices=[0], convention='wells', figure=fig)
f.show_colorscale(cmap="Greys", aspect="equal")
f.save("filfinder_" + fits_name + "_1." + save_ext)
plt.close()

print('FITS header:\n', hdu.header)

# ----------------------------
# FilFinder パラメータ
# ----------------------------
distance = 436 * u.pc
BMAJ = 18.0 * u.arcsec

# ----------------------------
# 背景除去（全リージョン共通）
# ic5146は負値→NaNも兼ねる
# ----------------------------
data = hdu.data.astype(np.float64).copy()
if REPLACE_NEG_WITH_NAN:
    print("ic5146: replacing negative values with NaN...")
    data[data < 0] = np.nan
valid = np.isfinite(data)
bg = np.nanpercentile(data[valid], 2)
background = valid & (data <= bg)
kernel = np.ones((21, 21), np.uint8)
background_expand = cv2.dilate(
    background.astype(np.uint8), kernel, iterations=2
).astype(bool)
data[background_expand] = np.nan
hdu.data = data

# 背景マスク保存
cv2.imwrite(
    os.path.join(fullimg_dir, fits_name + "_background_mask.png"),
    (background_expand * 255).astype(np.uint8)
)

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
    fil.preprocess_image(flatten_percent=FLATTEN_PERCENT)
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
print('preprocess_image : {:.2f}s'.format(time.perf_counter() - start))

# flat_img診断
flat = fil.flat_img
finite_count = int(np.isfinite(np.array(flat.value if hasattr(flat, 'value') else flat)).sum())
total_count = flat.size
print('flat_img finite count: {} / {} ({:.1f}%)'.format(finite_count, total_count, 100*finite_count/total_count))

cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_preprocess_image.png"), fil.image.value.astype("int"))
data_des = pd.DataFrame(pd.Series(fil.image.value.astype(np.float64).ravel()).describe())
print(data_des)

# ----------------------------
# 正規化の切り分け
#  - data_bak_ml: MLモデル入力用（log10スケール）
#  - data_bak_ff: FilFinder真値生成用（線形スケール）
# ----------------------------
img_raw = fil.image.value.copy().astype(np.float64)
img_raw[~np.isfinite(img_raw)] = np.nan
finite_mask = np.isfinite(img_raw)

# ML用: log10正規化
img_log = np.where(finite_mask & (img_raw > 0), np.log10(img_raw), np.nan)
des_log = pd.DataFrame(pd.Series(img_log[np.isfinite(img_log)].ravel()).describe(
    percentiles=[0.005, 0.995]))
data_bak_ml = img_log.copy()
data_bak_ml -= des_log[0]["0.5%"]
data_bak_ml /= ((des_log[0]["99.5%"] - des_log[0]["0.5%"]) / 255)
data_bak_ml = np.clip(data_bak_ml, 0, 255)
data_bak_ml[~finite_mask] = np.nan
print('ML input (log norm):')
print(pd.DataFrame(pd.Series(data_bak_ml[np.isfinite(data_bak_ml)].ravel()).describe()))

# FilFinder用: 線形正規化
des_lin = pd.DataFrame(pd.Series(img_raw[finite_mask].ravel()).describe(
    percentiles=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]))
data_bak_ff = img_raw.copy()
data_bak_ff -= des_lin[0]["0.5%"]
data_bak_ff /= ((des_lin[0]["99.5%"] - des_lin[0]["0.5%"]) / 255)
data_bak_ff = np.clip(data_bak_ff, 0, 255)
data_bak_ff[~finite_mask] = np.nan
print('FilFinder label (linear norm):')
print(pd.DataFrame(pd.Series(data_bak_ff[np.isfinite(data_bak_ff)].ravel()).describe()))

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

# analyze_skeletons
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-analyze_skeletons.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    fil.analyze_skeletons()
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
print('analyze_skeletons : {:.2f}s'.format(time.perf_counter() - start))

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

# ----------------------------
# ① ラベル（スケルトン）全面画像を保存
# ----------------------------
start = time.perf_counter()
fil_skeleton = fil.skeleton.astype("int")
fil_skeleton_vis = (fil_skeleton * 255).astype("uint8")
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_label_01.png"), fil_skeleton)
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_label_255.png"), fil_skeleton_vis)
print(pd.DataFrame(pd.Series(fil_skeleton.ravel()).describe()))
print('label images saved : {:.2f}s'.format(time.perf_counter() - start))

# ----------------------------
# ② タイル切り出し
# ----------------------------
crop_data(data_bak, data_bak, "in")
crop_data(fil_skeleton, data_bak, "out")

# ----------------------------
# ③ 可視化
# ----------------------------
start = time.perf_counter()
nan_mask = np.flipud(~np.isfinite(fil.image.value))
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
    f.show_contour(fits.PrimaryHDU(fil_skeleton, hdu.header), colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_log.png"))
plt.close()

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=-100, vmax=100, cmap="Greys")
f.add_colorbar()
if fil_skeleton.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton, hdu.header), colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_linear.png"))
plt.close()

print('skeleton visualization saved : {:.2f}s'.format(time.perf_counter() - start))

# ----------------------------
# ④ skeleton_longpath 可視化
# ----------------------------
start = time.perf_counter()
fil_skeleton_longpath = fil.skeleton_longpath.astype("int")

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=1, vmax=100, stretch="log", cmap="Greys")
f.add_colorbar()
if fil_skeleton_longpath.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton_longpath, hdu.header), colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_longpath_log.png"))
plt.close()

fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=-100, vmax=100, cmap="Greys")
f.add_colorbar()
if fil_skeleton_longpath.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton_longpath, hdu.header), colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_longpath_linear.png"))
plt.close()

print('skeleton_longpath visualization saved : {:.2f}s'.format(time.perf_counter() - start))
