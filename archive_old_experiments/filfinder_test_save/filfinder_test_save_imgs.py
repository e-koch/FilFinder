```python
from astropy.io import fits
from fil_finder import FilFinder2D
import astropy.units as u
import numpy as np
import cv2
import os

# ----------------
# 入力
# ----------------
root_dir="/mnt/sp4t/fujiwara.r"
fits_name="HGBS_ic5146_hires_column_density_map"

save_dir=os.path.join(
    root_dir,
    fits_name,
    "mask_test"
)

os.makedirs(
    save_dir,
    exist_ok=True
)

# ----------------
# FITS読込
# ----------------
hdu=fits.open(
    os.path.join(
        root_dir,
        fits_name+".fits"
    )
)[0]

# =====================================================
# ic5146だけ追加処理
# =====================================================
flatten_percent=90

if "ic5146" in fits_name.lower():

    data=hdu.data.astype(
        np.float64
    ).copy()

    # 負値→NaN
    data[
        data<0
    ]=np.nan

    hdu.data=data

    flatten_percent=95

# ----------------
# FilFinder
# ----------------
fil=FilFinder2D(

    hdu,

    distance=436*u.pc,

    beamwidth=18*u.arcsec
)

# ----------------
# 前処理
# ----------------
fil.preprocess_image(
    flatten_percent=flatten_percent
)

# ----------------
# マスク作成
# ----------------
fil.create_mask(

    verbose=True,

    adapt_thresh=15*u.pix,

    smooth_size=5*u.pix,

    glob_thresh=100,

    size_thresh=5000*u.pix**2,

    fill_hole_size=100*u.pix**2
)

# ----------------
# スケルトン生成
# ----------------
fil.medskel()

# ----------------
# 背景画像作成
# ----------------
img=fil.image.value.astype(
    np.float64
)

p1=np.nanpercentile(
    img,
    0.5
)

p2=np.nanpercentile(
    img,
    99.5
)

img=(
    img-p1
)/(
    p2-p1
)*255

img=np.clip(
    img,
    0,
    255
)

gray=np.where(

    np.isfinite(
        img
    ),

    img,

    0

).astype(
    np.uint8
)

gray=np.flipud(
    gray
)

# ----------------
# ① skeleton overlay
# ----------------
skel=np.flipud(

    fil.skeleton.astype(
        np.uint8
    )
)

overlay_skel=cv2.cvtColor(

    gray,

    cv2.COLOR_GRAY2BGR
)

overlay_skel[
    skel==1
]=(0,0,255)

cv2.imwrite(

    os.path.join(

        save_dir,

        "01_input_skeleton_overlay.png"

    ),

    overlay_skel
)

# ----------------
# ② mask overlay
# ----------------
mask=np.flipud(

    fil.mask.astype(
        np.uint8
    )
)

overlay_mask=cv2.cvtColor(

    gray,

    cv2.COLOR_GRAY2BGR
)

overlay_mask[
    mask==1
]=(0,0,255)

cv2.imwrite(

    os.path.join(

        save_dir,

        "02_input_mask_overlay.png"

    ),

    overlay_mask
)

# ----------------
# ③ mask only
# ----------------
mask_img=(
    mask*255
).astype(
    np.uint8
)

cv2.imwrite(

    os.path.join(

        save_dir,

        "03_mask_only.png"

    ),

    mask_img
)

print(
    "saved:",
    save_dir
)
```

