
"""
柱密度画像からのフィラメント同定 — FilFinderを用いた2次元解析 【loose版】

strict版(元スクリプト)からの変更点:
  - phys_thresh を strict の半分に (3.76e+21 -> 1.88e+21)
  - LEN_CUT_PC を strict の半分に (0.3 -> 0.15)
  - 出力先を cache_loose / full_loose / vis_loose に変更(strict出力は上書きしない)
  - それ以外のパラメータ(MASK_NSIG, ADAPT_PC, SMOOTH_PC, SKEL_THRESH_PC,
    BRANCH_PC)はstrictと同一(検出ロジックの形自体は変えず、閾値のみ緩める)

他の領域(ic5146, orionB, polaris, tauN3)に展開する場合は、
TARGETSのphys_thresh(各領域のstrict値)を半分にした値に置き換えてください。
Polarisはstrict=0.94e+21なので、loose=0.47e+21になります。
"""

import os
import glob
import warnings
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
from astropy.visualization import ZScaleInterval
import astropy.units as u
import concurrent.futures as cf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

# 警告の非表示設定
warnings.filterwarnings("ignore")

# --- 環境セットアップ -------------------------------------------------------
for _v in ("ASTROPY_CONFIGDIR", "XDG_CONFIG_HOME"):
    os.environ[_v] = os.path.abspath("./.astropy_cfg")
for _v in ("ASTROPY_CACHEDIR", "XDG_CACHE_HOME"):
    os.environ[_v] = os.path.abspath("./.astropy_cache")
os.makedirs("./.astropy_cfg",   exist_ok=True)
os.makedirs("./.astropy_cache", exist_ok=True)

# --- モダンな環境で fil_finder 1.8 を動作させるための互換性調整 ---------
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from fil_finder import FilFinder2D

def _fresh_pool():
    return cf.ThreadPoolExecutor(max_workers=1)

print("インポート完了 — fil_finder の準備が整いました (loose版)")


# --- 設定 (CONFIG) ---------------------------------------------------------
root_dir  = "/mnt/sp4t/fujiwara.r"
fits_name = "HGBS_polaris_hires_column_density_map"

# フォルダの設定 【変更】 strictの cache/full/vis ではなく loose用に分離
cache_dir   = os.path.join(root_dir, fits_name, "cache_loose")
fullimg_dir = os.path.join(root_dir, fits_name, "full_loose")
vis_dir     = os.path.join(root_dir, fits_name, "vis_loose")

# ディレクトリ作成
for d in [cache_dir, fullimg_dir, vis_dir]:
    os.makedirs(d, exist_ok=True)

# FilFinder パラメータ（パーセク単位）— strictと同一(検出ロジックの形は変えない)
MASK_NSIG      = 5.0    # マスクに用いるグローバル閾値のシグマ倍
ADAPT_PC       = 0.2    # アダプティブ閾値のパッチサイズ
SMOOTH_PC      = 0.2    # メディアン平滑化カーネルサイズ
SKEL_THRESH_PC = 0.2    # 保持する孤立スケルトンの最小長さ
BRANCH_PC      = 0.2    # 保持する内部枝の最小長さ

# カタログフィルタ 【変更】 strictの半分(0.3 -> 0.15)
LEN_CUT_PC     = 0.1

# 天体メタデータ 【変更】 phys_threshをstrictの半分(3.76e+21 -> 1.88e+21)に
TARGETS = {
    "polaris": dict(
        file=os.path.join(root_dir, f"{fits_name}.fits"),
        dist_pc=140.0,
        beam_arcsec=18.0,
        phys_thresh=0.94e+21
    )
}

# 全天体で 0 以下の値（データなし・バックグラウンド外）を NaN 化する設定
REPLACE_NEG_WITH_NAN = True


# --- ヘルパー関数 --------------------------------------------------------
def load_column_density_2d(fpath):
    """2次元の柱密度FITSマップを読み込みます。"""
    with fits.open(fpath, memmap=True) as hdul:
        h = hdul[0].header.copy()
        data = np.squeeze(np.asarray(hdul[0].data, dtype=np.float32))

    if REPLACE_NEG_WITH_NAN:
        print("データ領域外・無効値(<=0)を NaN に置き換えています...")
        data[data <= 0] = np.nan

    hdr2d = make_header2d(h)

    meta = dict(tel=str(h.get("TELESCOP", "unknown")).strip(),
                obj=str(h.get("OBJECT", "unknown")).strip(),
                pix_arcsec=abs(hdr2d.get("CDELT2", hdr2d.get("CDELT1", 0.0))) * 3600.0)
    return data, hdr2d, meta


def make_header2d(h):
    """2D WCSヘッダーをクリーンアップします。"""
    h2 = h.copy()
    for k in list(h2.keys()):
        if k in ("NAXIS3", "NAXIS4") or k[-1:] in ("3", "4") and k[:-1] in (
            "NAXIS","CTYPE","CRVAL","CRPIX","CDELT","CUNIT","CROTA"):
            del h2[k]
    h2["NAXIS"]   = 2
    h2["WCSAXES"] = 2
    return h2


def estimate_noise_2d(data):
    """MADを用いてロバストにノイズを推定します。"""
    finite_data = data[np.isfinite(data)]
    rms = mad_std(finite_data)
    return float(rms)


def run_filfinder(data2d, hdr2d, beam_arcsec, dist_pc, mask_thresh,
                  skel_thresh_pc=SKEL_THRESH_PC, branch_pc=BRANCH_PC):
    """FilFinder2D を実行してフィラメント骨格を検出します。"""
    pixscale_arcsec = float(np.abs(hdr2d.get("CDELT2", hdr2d.get("CDELT1")))) * 3600.0
    pc_per_pix = (pixscale_arcsec / 206265.0) * dist_pc

    smooth_px   = SMOOTH_PC / pc_per_pix
    adapt_px    = ADAPT_PC  / pc_per_pix
    smooth_size  = max(round(smooth_px), 1.5) * u.pix
    adapt_thresh = max(round(adapt_px),   3.0) * u.pix

    fil = FilFinder2D(data2d, header=hdr2d,
                      beamwidth=beam_arcsec * u.arcsec,
                      distance=dist_pc * u.pc)

    fil.preprocess_image(skip_flatten=True)
    fil.create_mask(glob_thresh=mask_thresh,
                    smooth_size=smooth_size, adapt_thresh=adapt_thresh,
                    verbose=False)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(skel_thresh=skel_thresh_pc * u.pc,
                          branch_thresh=branch_pc * u.pc)
    return fil


def fil_length_pc(f, pc_per_pix):
    """各フィラメントの長さを pc 単位で取得します。"""
    try:
        L = f.length(u.pc)
        val = L.to(u.pc).value if hasattr(L, "unit") else float(L)
        if 0 < val < 1e3:
            if val < 300:
                return float(val)
    except Exception:
        pass
    try:
        rc, cc = f.longpath_pixel_coords
        return float(len(rc)) * pc_per_pix
    except Exception:
        return 0.0


def robust_catalogue(fil, pc_per_pix, len_cut_pc=LEN_CUT_PC):
    """長さ閾値フィルタリングを適用します。"""
    lengths = [fil_length_pc(f, pc_per_pix) for f in fil.filaments]
    keep = [L >= len_cut_pc for L in lengths]
    return lengths, keep


def plot_overlay(data2d, fil, name, meta, dist_pc, beam_arcsec, save=None):
    """
    柱密度マップ + フィラメント最長経路 + 5pcスケールバーのプロット
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_facecolor("black")

    cmap = plt.cm.get_cmap("inferno").copy()
    cmap.set_bad(color="black")

    finite = np.isfinite(data2d)
    vmin, vmax = ZScaleInterval().get_limits(data2d[finite])
    im = ax.imshow(data2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r"Column Density $N(\mathrm{H}_2)$ [$\mathrm{cm}^{-2}$]")

    for f in fil.filaments:
        try:
            rc, cc = f.longpath_pixel_coords
            ax.plot(cc, rc, color="cyan", lw=1.8, alpha=0.9)
        except Exception:
            pass

    ny, nx = data2d.shape
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    hdr2d = fil.header
    pixscale_arcsec = float(np.abs(hdr2d.get("CDELT2", hdr2d.get("CDELT1")))) * 3600.0
    pc_per_pix = (pixscale_arcsec / 206265.0) * dist_pc
    pixels_for_5pc = 5.0 / pc_per_pix

    bar_x_end = nx - int(nx * 0.05)
    bar_x_start = bar_x_end - pixels_for_5pc
    bar_y = int(ny * 0.08)

    for lw, col, z in [(6, "black", 5), (3.0, "white", 6)]:
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
                color=col, lw=lw, solid_capstyle="butt", zorder=z)
    ax.text((bar_x_start + bar_x_end) / 2.0, bar_y + (ny * 0.02), "5 pc",
            color="white", fontsize=12, fontweight="bold", ha="center", va="bottom",
            zorder=6, path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    ax.set_title(f"{name} ({meta.get('tel', 'Herschel')}) [loose] "
                 f"d={dist_pc:.0f} pc, beam={beam_arcsec:.1f}\" — {len(fil.filaments)} Filaments")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor="black")
    return fig
def plot_grayscale_skeleton_mask(data2d, fil, name, save_skeleton=None, save_mask=None):
    """
    グレースケール背景の上に、skeleton(赤線)とmask(半透明領域)をそれぞれ重ねて保存する。
    - save_skeleton: skeletonオーバーレイの保存先パス
    - save_mask    : maskオーバーレイの保存先パス
    """
    finite = np.isfinite(data2d)
    vmin, vmax = ZScaleInterval().get_limits(data2d[finite])

    # --- skeletonオーバーレイ ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("black")
    ax.imshow(data2d, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    skel_rgba = np.zeros((*fil.skeleton.shape, 4), dtype=float)
    skel_bool = fil.skeleton.astype(bool)
    skel_rgba[skel_bool] = [1.0, 0.0, 0.0, 0.9]  # 赤

    ax.imshow(skel_rgba, origin="lower")
    ax.set_title(f"{name} [loose] — Skeleton overlay (n_fil={len(fil.filaments)})")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    plt.tight_layout()
    if save_skeleton:
        fig.savefig(save_skeleton, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    # --- maskオーバーレイ ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("black")
    ax.imshow(data2d, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    mask_rgba = np.zeros((*fil.mask.shape, 4), dtype=float)
    mask_bool = fil.mask.astype(bool)
    mask_rgba[mask_bool] = [1.0, 0.4, 0.0, 0.5]  # オレンジ、半透明

    ax.imshow(mask_rgba, origin="lower")
    ax.set_title(f"{name} [loose] — Mask overlay")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    plt.tight_layout()
    if save_mask:
        fig.savefig(save_mask, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

# --- メイン処理 -------------------------------------------------------------
def main():
    summary = []

    for NAME, cfg in TARGETS.items():
        if not os.path.exists(cfg["file"]):
            print(f"[SKIP] ファイルが見つかりません: {cfg['file']}")
            continue

        try:
            print(f"\n天体の処理を開始します(loose): {NAME}")
            data, hdr2d, meta = load_column_density_2d(cfg["file"])
            rms = estimate_noise_2d(data)

            if cfg.get("phys_thresh") is not None:
                mask_thresh = cfg["phys_thresh"]
                print(f"  使用する物理マスク閾値(loose) = {mask_thresh:.2e} cm^-2")
            else:
                mask_thresh = MASK_NSIG * rms
                print(f"  使用する自動マスク閾値 ({MASK_NSIG:.0f} sigma) = {mask_thresh:.2e} cm^-2")

            fil = run_filfinder(data, hdr2d, cfg["beam_arcsec"], cfg["dist_pc"],
                                mask_thresh=mask_thresh)

            with open(os.path.join(cache_dir, fits_name + "_FilFinder2D-medskel.pkl"), "wb") as f:
                pickle.dump(fil, f)

            with open(os.path.join(cache_dir, fits_name + "_FilFinder2D-create_mask.pkl"), "wb") as f:
                pickle.dump(fil, f)

            fits.writeto(os.path.join(fullimg_dir, f"column_density_{NAME}.fits"), data.astype(np.float32),
                         hdr2d, overwrite=True)
            fits.writeto(os.path.join(fullimg_dir, f"skeleton_{NAME}.fits"),
                         fil.skeleton.astype(np.int16), hdr2d, overwrite=True)

            pc_per_pix = (float(np.abs(hdr2d.get("CDELT2", hdr2d.get("CDELT1"))))*3600.0/206265.0)*cfg["dist_pc"]
            lengths, keep = robust_catalogue(fil, pc_per_pix)
            with open(os.path.join(fullimg_dir, f"filaments_{NAME}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["fil_id", "length_pc", "robust"])
                for i, (L, k) in enumerate(zip(lengths, keep)):
                    w.writerow([i, f"{L:.3f}", int(k)])

            fig = plot_overlay(data, fil, NAME, meta, cfg["dist_pc"], cfg["beam_arcsec"],
                               save=os.path.join(vis_dir, f"overlay_{NAME}.png"))
            plt.close(fig)

            # ↓ ここを追加
            plot_grayscale_skeleton_mask(
                data, fil, NAME,
                save_skeleton=os.path.join(vis_dir, f"skeleton_overlay_{NAME}.png"),
                save_mask=os.path.join(vis_dir, f"mask_overlay_{NAME}.png"),
            )

            len_robust = [L for L, k in zip(lengths, keep) if k]
            summary.append(dict(target=NAME, telescope=meta["tel"],
                                dist_pc=cfg["dist_pc"], beam_arcsec=cfg["beam_arcsec"],
                                rms_val=round(rms, 3),
                                mask_thresh=round(mask_thresh, 3),
                                n_filaments=len(fil.filaments),
                                n_robust=int(sum(keep)),
                                len_total_pc=round(sum(lengths), 2),
                                len_total_robust_pc=round(sum(len_robust), 2),
                                len_max_pc=round(max(lengths), 2) if lengths else 0.0))
            print(f"[OK] {NAME:9s} {meta['tel']:7s}  {len(fil.filaments):2d} fil "
                  f"({int(sum(keep))} robust)  Lmax={max(lengths) if lengths else 0:.2f} pc")
        except Exception as e:
            print(f"[FAIL] {NAME}: {e}")
            summary.append(dict(target=NAME, telescope="?", n_filaments=-1))

    if summary:
        keys = ["target","telescope","dist_pc","beam_arcsec","rms_val",
                "mask_thresh","n_filaments","n_robust",
                "len_total_pc","len_total_robust_pc","len_max_pc"]
        with open(os.path.join(fullimg_dir, "filament_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for row in summary:
                w.writerow({k: row.get(k, "") for k in keys})
        print(f"\nサマリー出力完了 -> {os.path.join(fullimg_dir, 'filament_summary.csv')}")


if __name__ == "__main__":
    main()
