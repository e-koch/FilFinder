import mmcv
# ------------------------------------------------------------
# FilFinder 実行 + FITS 切り出し 保存用スクリプト
#
# 目的:
#  - FITS を読み込み、FilFinder を順に実行して結果をキャッシュする
#  - 前処理 / マスク生成 / 骨格化 / スケルトン解析 / RHT を行う
#  - 各段階を pickle に保存して再実行を避ける（高速化のため）
#  - 前処理結果やスケルトンを 256x256 タイルに切り出して保存する
#
# 使い方（簡易）:
#  1) 必要ライブラリをインストール: fil_finder, astropy, mmcv, tifffile, cv2 など
#  2) `root_dir` と `fits_name` を実行環境用に合わせる
#  3) このスクリプトを実行すると出力ファイル（png, tif, pkl）がカレントに生成されます
# ------------------------------------------------------------


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
matplotlib.use('Agg')  # GUI を使わずにプロットを保存する
import pickle
import time

import cv2
from tifffile import imwrite

# -----------------------------------------------------------------
# 入力設定: 実行環境に合わせて変更してください
#  - root_dir: FITS が置いてあるディレクトリ (ここではサンプルのパス)
#  - fits_name: ファイル名（拡張子 .fits は後で付加）
# -----------------------------------------------------------------
root_dir="/mnt/sp4t/fujiwara.r"
fits_name="HGBS_tauN3_hires_column_density_map"

# 出力フォーマット（PDF/PNG 等）
save_ext = "png"  # 'png' に変えるとPNGで保存します


# ----------------------------
# ユーティリティ関数
# ----------------------------
def check_make_dir(target_dir):
    """ディレクトリがなければ作成する（安全対策）"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def load_pkl(fpath):
    """Pickle ファイルを読み込む。存在しなければ None を返す"""
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    return None


def save_pkl(fpath, val):
    """オブジェクトを Pickle に保存する（上書き）"""
    with open(fpath, 'wb') as f:
        pickle.dump(val, f)


# ----------------------------
# タイル切り出し設定
#  - size: タイルサイズ（ここでは 256）
#  - step_rate: タイルのステップ（オーバーラップ率に相当）
#  - tiles_per_dir: 1ディレクトリあたりの最大ファイル数
# ----------------------------
size = 256
step_rate = 0.25  # タイル移動量 = step_rate * size (= 64px)
tiles_per_dir = 1000

# 出力サブディレクトリ
cache_dir = os.path.join(root_dir, fits_name, "cache")    # pkl キャッシュ
fullimg_dir = os.path.join(root_dir, fits_name, "full")   # 全面画像
vis_dir = os.path.join(root_dir, fits_name, "vis")        # 可視化PNG


def crop_data(nda, nda_check_nan, suf):
    """
    画像をタイルに切り出して保存する関数

    nda: 切り出す配列（表示用/保存用）
    nda_check_nan: NaN 判定用に使う配列（通常は同じサイズの元データ）
    suf: 保存ファイル名のサフィックス（"in" または "out"）
    出力先: {root_dir}/{fits_name}/{suf}/tiles_000/ 以下に1000枚ごとに分割
    """
    base_dir = os.path.join(root_dir, fits_name, suf)

    # width/height を取得（OpenCV 用の順序ではない点に注意）
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

            # NaN を含むタイルはスキップ（端部やフットプリント外を除外）
            if np.isnan(cropped_check).any():
                continue

            # 1000枚ごとにサブディレクトリを切り替える
            sub_dir = os.path.join(base_dir, "tiles_{:03d}".format(tile_count // tiles_per_dir))
            check_make_dir(sub_dir)

            # Pickle, PNG, TIFF を保存
            o_path = os.path.join(sub_dir, o_name + "_" + suf + ".pkl")
            save_pkl(o_path, cropped)
            o_path = os.path.join(sub_dir, o_name + "_" + suf + ".png")
            cv2.imwrite(o_path, cropped.astype("int"))
            tiff_path = os.path.join(sub_dir, o_name + "_" + suf + ".tif")
            imwrite(tiff_path, cropped.astype(np.float32))

            tile_count += 1

        y += step

    print('Total tiles saved ({}): {}'.format(suf, tile_count))


# ----------------------------
# 簡易テスト用に小さな TIFF を一時保存（デバッグ用）
# ----------------------------
d = np.ndarray(shape=(10,20), dtype=np.float32)
d[()] = np.arange(200).reshape(10, 20)
imwrite("filfinder_" + fits_name + ".tif", d, dtype=np.float32)


# ============================================================
# メイン処理: FITS 読み込み -> FilFinder 実行 -> 結果保存/切り出し
# ============================================================
# 出力ディレクトリを作成
for d in [cache_dir, fullimg_dir, vis_dir]:
    check_make_dir(d)

start = time.perf_counter()
# FITS を開く（hdu は Primary HDU を想定）
hdu_tmp = fits.open(os.path.join(root_dir, fits_name + ".fits"))
print('hdu tmp:', hdu_tmp)
hdu = hdu_tmp[0]
end = time.perf_counter()
print('load hdu_tmp : {:.2f}s'.format(end - start))

# WCS オブジェクト作成（必要なら座標変換に使用）
start = time.perf_counter()
w = WCS(hdu)
end = time.perf_counter()
print('WCS(hdu) : {:.2f}s'.format(end - start))

# 可視化のために簡易的に FITSFigure を保存
start = time.perf_counter()
fig = plt.figure(figsize=(8, 8))
f = aplpy.FITSFigure(hdu, slices=[0], convention='wells', figure=fig)
f.show_colorscale(cmap="Greys", aspect="equal")
f.save("filfinder_" + fits_name + "_1." + save_ext)
end = time.perf_counter()
print('FITSFigure : {:.2f}s'.format(end - start))

print('FITS header:\n', hdu.header)

# 元データを PNG として保存（簡易確認用）
cv2.imwrite(fits_name + "_hdu_data.png", hdu.data.astype("int"))
print(pd.DataFrame(pd.Series(hdu.data.astype("int").ravel()).describe()))


# ----------------------------
# FilFinder の実行パラメータ
#  - distance, beamwidth はデータに応じて調整
# ----------------------------
distance = 436 * u.pc
BMAJ = 18.0 * u.arcsec

# ① 端部の異常値（極端な負値）をNaNに置換して縦縞ノイズを除去
hdu.data[hdu.data < 0] = np.nan

# FilFinder オブジェクトを pickle キャッシュから読み込む（無ければ生成して保存）
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D.pkl")
start = time.perf_counter()
fil = load_pkl(pkl_name)
if fil is None:
    fil = FilFinder2D(hdu, distance=distance, beamwidth=BMAJ)
    save_pkl(pkl_name, fil)
end = time.perf_counter()
print('FilFinder2D : {:.2f}s'.format(end - start))


# ----------------------------
# 前処理: preprocess_image をキャッシュ化して再実行を避ける
# ② flatten_percent=95 を指定してフラットニングを有効化
# ----------------------------
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-preprocess_image.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    fil.preprocess_image(flatten_percent=95)
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
end = time.perf_counter()
print('preprocess_image : {:.2f}s'.format(end - start))

# flat_img の診断
flat = fil.flat_img
finite_count = int(np.isfinite(np.array(flat.value if hasattr(flat, 'value') else flat)).sum())
total_count = flat.size
print('flat_img finite count: {} / {} ({:.1f}%)'.format(finite_count, total_count, 100*finite_count/total_count))

# 前処理結果を画像として保存（確認用）
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_preprocess_image.png"), fil.image.value.astype("int"))
data_des = pd.DataFrame(pd.Series(fil.image.value.astype(np.float64).ravel()).describe())
print(data_des)

# ----------------------------
# 正規化の切り分け
#  - data_bak_ml : MLモデル入力用（広いダイナミックレンジ、対数スケール正規化）
#  - data_bak_ff : FilFinder真値生成用（弱い構造に反応しないよう高パーセンタイル正規化）
# ----------------------------
img_raw = fil.image.value.copy().astype(np.float64)
img_raw[~np.isfinite(img_raw)] = np.nan
finite_mask = np.isfinite(img_raw)

# MLモデル入力用: log10スケールで正規化（広いダイナミックレンジを保持）
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

# FilFinder真値生成用: 線形スケールで高パーセンタイル正規化（弱構造を抑制）
des_lin = pd.DataFrame(pd.Series(img_raw[finite_mask].ravel()).describe(
    percentiles=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]))
data_bak_ff = img_raw.copy()
data_bak_ff -= des_lin[0]["0.5%"]
data_bak_ff /= ((des_lin[0]["99.5%"] - des_lin[0]["0.5%"]) / 255)
data_bak_ff = np.clip(data_bak_ff, 0, 255)
data_bak_ff[~finite_mask] = np.nan
print('FilFinder label (linear norm):')
print(pd.DataFrame(pd.Series(data_bak_ff[np.isfinite(data_bak_ff)].ravel()).describe()))

# タイル切り出し・vis画像生成にはML用を使う
data_bak = data_bak_ml

# 全面画像として保存
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_input_full_ml.png"), data_bak_ml.astype("uint8"))
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_input_full_ff.png"), data_bak_ff.astype("uint8"))
mmcv.image.use_backend('tifffile')
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_input_full_ml.tif"), data_bak_ml.astype(np.float32))


# ----------------------------
# create_mask / medskel / analyze_skeletons / exec_rht
# それぞれキャッシュを使って高速化
# ----------------------------
pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-create_mask.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    plt.figure(figsize=(8, 8))
    fil.create_mask(
        verbose=True,

        adapt_thresh=15 * u.pix,
        smooth_size=5 * u.pix,
        glob_thresh=100,
        size_thresh=5000 * u.pix ** 2,
        fill_hole_size=100 * u.pix ** 2
    )
    plt.savefig("filfinder_" + fits_name + "_2." + save_ext)
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
end = time.perf_counter()
print('create_mask : {:.2f}s'.format(end - start))

pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-medskel.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    plt.figure(figsize=(8, 8))
    fil.medskel(verbose=True, save_png=True)
    plt.savefig("filfinder_" + fits_name + "_3." + save_ext)
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
end = time.perf_counter()
print('medskel : {:.2f}s'.format(end - start))

pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-analyze_skeletons.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    fil.analyze_skeletons()
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
end = time.perf_counter()
print('analyze_skeletons : {:.2f}s'.format(end - start))

pkl_name = os.path.join(cache_dir, fits_name + "_FilFinder2D-exec_rht.pkl")
start = time.perf_counter()
fil_tmp = load_pkl(pkl_name)
if fil_tmp is None:
    fil.exec_rht()
    save_pkl(pkl_name, fil)
else:
    fil = fil_tmp
end = time.perf_counter()
print('exec_rht : {:.2f}s'.format(end - start))

print('Detected filaments:', len(fil.filaments))


# ----------------------------
# ① ラベル（スケルトン）全面画像を保存
#    - 0/1 版（学習用ラベル）
#    - 0/255 版（可視化用）
# ----------------------------
start = time.perf_counter()
fil_skeleton = fil.skeleton.astype("int")          # 値: 0 or 1
fil_skeleton_vis = (fil_skeleton * 255).astype("uint8")  # 値: 0 or 255

# 0/1 版を保存（学習用ラベルの全面画像）
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_label_01.png"), fil_skeleton)
# 0/255 版を保存（可視化用グレースケール全面画像）
cv2.imwrite(os.path.join(fullimg_dir, fits_name + "_label_255.png"), fil_skeleton_vis)
print(pd.DataFrame(pd.Series(fil_skeleton.ravel()).describe()))
end = time.perf_counter()
print('label images saved : {:.2f}s'.format(end - start))

# ----------------------------
# ② 全面画像からタイルを切り出す
#    in: 入力パッチ（data_bak、0-255正規化済み）
#    out: ラベルパッチ（0/1 で保持）
#    NaN判定は data_bak を基準にする
# ----------------------------
crop_data(data_bak, data_bak, "in")           # 入力パッチ
crop_data(fil_skeleton, data_bak, "out")       # ラベルパッチ（0/1）

# ----------------------------
# ③ FilFinder 結果の可視化（PNG で保存）
# ----------------------------
start = time.perf_counter()

# NaN領域マスク（観測領域外を黒にする）
nan_mask = np.flipud(~np.isfinite(fil.image.value))

# 入力画像（正規化済み）をグレースケール PNG として保存（上下反転で他画像と向きを統一）
input_vis = np.flipud(np.where(np.isfinite(data_bak), data_bak, 0).astype("uint8"))
cv2.imwrite(os.path.join(vis_dir, fits_name + "_vis_input.png"), input_vis)

# マスク画像（create_mask 結果）をグレースケール PNG として保存（上下反転）
mask_vis = np.flipud(fil.mask.astype("uint8") * 255)
mask_vis[nan_mask] = 0
cv2.imwrite(os.path.join(vis_dir, fits_name + "_vis_mask.png"), mask_vis)

# 入力画像にマスクを重ねた画像（マスク領域を赤で表示）
input_rgb = cv2.cvtColor(input_vis, cv2.COLOR_GRAY2BGR)
mask_flipud = np.flipud(fil.mask.astype("uint8"))
mask_flipud[nan_mask] = 0
input_rgb[mask_flipud == 1] = (0, 0, 255)  # BGR: 赤
cv2.imwrite(os.path.join(vis_dir, fits_name + "_vis_input_mask_overlay.png"), input_rgb)

# 入力画像にスケルトンを重ねた画像（スケルトンを赤で表示）
input_rgb2 = cv2.cvtColor(input_vis, cv2.COLOR_GRAY2BGR)
skel_flipud = np.flipud(fil_skeleton.astype("uint8"))
skel_flipud[nan_mask] = 0
input_rgb2[skel_flipud == 1] = (0, 0, 255)  # BGR: 赤
cv2.imwrite(os.path.join(vis_dir, fits_name + "_vis_input_skeleton_overlay.png"), input_rgb2)

# スケルトンオーバーレイ（log stretch）を PNG で保存
fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=1, vmax=100, stretch="log", cmap="Greys")
f.add_colorbar()
if fil_skeleton.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton, hdu.header), colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_log.png"))
plt.close()

# スケルトンオーバーレイ（linear stretch）を PNG で保存
fig = plt.figure(figsize=(16, 16))
f = aplpy.FITSFigure(hdu, slices=[0], figure=fig)
f.show_colorscale(vmin=-100, vmax=100, cmap="Greys")
f.add_colorbar()
if fil_skeleton.max() > 0:
    f.show_contour(fits.PrimaryHDU(fil_skeleton, hdu.header), colors="r", linewidths=2, slices=[0], alpha=0.4)
f.save(os.path.join(vis_dir, fits_name + "_vis_skeleton_linear.png"))
plt.close()

end = time.perf_counter()
print('skeleton visualization saved : {:.2f}s'.format(end - start))

# ----------------------------
# ④ skeleton_longpath の可視化（PNG で保存）
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

end = time.perf_counter()
print('skeleton_longpath visualization saved : {:.2f}s'.format(end - start))
