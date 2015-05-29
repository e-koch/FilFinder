# Licensed under an MIT open source license - see LICENSE

'''
Script to run fil_finder on the Herschel Gould Belt data set.
Can be run on multiple cores.
Data can be downloaded at http://www.herschel.fr/cea/gouldbelt/en/Phocea/Vie_des_labos/Ast/ast_visu.php?id_ast=66.
'''

from fil_finder import *
from astropy.io import fits
from astropy import convolution
import os
from shutil import move
from datetime import datetime
from pandas import DataFrame
from scipy.ndimage import zoom

# Attempt at multi-core implementation


def wrapper(filename, distance, beamwidth, offset, verbose=False):
    print "Running " + filename + " at " + str(datetime.now())
    # hdu = fits.open(filename)
      # img = hdu[1].data
      # hdr = hdu[1].data
    img, hdr = fits.getdata("../" + filename, header=True)
    img = img + offset
    if not os.path.exists(filename[:-5]):
        os.makedirs(filename[:-5])

    # Convolve to the distance of IC-5146 (460 pc)
    convolve_to_common = True
    regrid_to_common = True
    if convolve_to_common:
        r = 460. / float(distance)
        if r != 1.:
            conv = np.sqrt(r ** 2. - 1) * \
                (beamwidth / np.sqrt(8*np.log(2)) / (np.abs(hdr["CDELT2"]) * 3600.))
            if conv > 1.5:
                kernel = convolution.Gaussian2DKernel(conv)
                good_pixels = np.isfinite(img)
                nan_pix = np.ones(img.shape)
                nan_pix[good_pixels == 0] = np.NaN
                img = convolution.convolve(img, kernel, boundary='fill',
                                           fill_value=np.NaN)
                # Avoid edge effects from smoothing
                img = img * nan_pix

                beamwidth *= r

    if regrid_to_common:

        # Regrid to nearest distance, which for this data set is Taurus at 140 pc
        r = float(distance) / 140.

        if r != 1:

            good_pixels = np.isfinite(img)
            good_pixels = zoom(good_pixels, round(r, 3),
                               order=0)

            img[np.isnan(img)] = 0.0
            regrid_conv_img = zoom(img, round(r, 3))


            nan_pix = np.ones(regrid_conv_img.shape)
            nan_pix[good_pixels == 0] = np.NaN


            img = regrid_conv_img * nan_pix

            distance = 140.

            hdr['CDELT2'] /= r


    # Toggle saving of the exact maps used in the algorithm
    save_regrid_convolve = True
    if save_regrid_convolve:
        hdr['NAXIS1'] = img.shape[1]
        hdr['NAXIS2'] = img.shape[0]

        hdu = fits.PrimaryHDU(img.astype(">f4"), header=hdr)

        hdu.writeto(filename[:-5]+"/"+filename[:-5]+"_regrid_convolved.fits")

    print filename, distance

    filfind = fil_finder_2D(img, hdr, beamwidth,
                            distance=distance, glob_thresh=20)

    print filfind.beamwidth, filfind.imgscale

    save_name = filename[:-5]

    filfind.create_mask()
    filfind.medskel(verbose=verbose)

    filfind.analyze_skeletons()
    filfind.compute_filament_brightness()
    filfind.exec_rht(branches=True)
    # Save the branches output separately
    for i in range(len(filfind.rht_curvature["Median"])):
        vals = np.vstack([filfind.rht_curvature[key][i] for key in filfind.rht_curvature.keys()]).T
        if i == 0:
            branches_rht = vals
        else:
            branches_rht = np.vstack((branches_rht, vals))
    df = DataFrame(branches_rht, columns=filfind.rht_curvature.keys())
    df.to_csv(filename[:-5] + "_rht_branches.csv")
    move(filename[:-5] + "_rht_branches.csv", filename[:-5])
    filfind.exec_rht(branches=False)
    filfind.find_widths(verbose=verbose)
    filfind.save_table(save_name=save_name, table_type="fits")
    # filfind.save_table(save_name=save_name, table_type="csv")
    filfind.save_fits(save_name=save_name, stamps=False)

    try:
        move(filename[:-5] + "_table.fits", filename[:-5])
    except:
        pass
    # Move the stamps folder
    try:
        move("stamps_" + filename[:-5], filename[:-5])
    except:
        pass

    move(filename[:-5] + "_mask.fits", filename[:-5])
    move(filename[:-5] + "_skeletons.fits", filename[:-5])
    move(filename[:-5] + "_filament_model.fits", filename[:-5])

    del filfind, img, hdr


def single_input(a):
    return wrapper(*a)

if __name__ == "__main__":

    # from multiprocessing import Pool
    from interruptible_pool import InterruptiblePool as Pool
    from itertools import izip

    MULTICORE = bool(raw_input("Run on multiple cores? (T or blank): "))
    if MULTICORE:
        NCORES = int(raw_input("How many cores to use? "))

    # os.chdir("/srv/astro/erickoch/gould_belt/degrade_all/")

    fits250 = ["pipeCenterB59-250.fits",  "lupusI-250.fits", "aquilaM2-250.fits", "orionB-250.fits", "polaris-250.fits",
               "chamaeleonI-250_normed.fits", "perseus04-250.fits", "taurusN3-250.fits", "ic5146-250.fits",
               "orionA-C-250.fits", "orionA-S-250.fits", "california_cntr-250_normed.fits", "california_east-250_normed.fits",
               "california_west-250_normed.fits"]
    fits350 = ["pipeCenterB59-350.fits", "lupusI-350.fits", "aquilaM2-350.fits", "orionB-350.fits", "polaris-350.fits",
               "chamaeleonI-350.fits", "perseus04-350.fits", "taurusN3-350.fits", "ic5146-350.fits",
               "orionA-C-350.fits", "orionA-S-350.fits", "california_cntr-350.fits", "california_east-350.fits",
               "california_west-350.fits"]
    distances = [145., 150., 260., 400., 150., 170., 235.,
                 140., 460., 400., 400., 450., 450., 450.]  # pc

    offsets = [31.697, 14.437, 85.452, 26.216, 9.330, -879.063, 23.698,
               21.273, 20.728, 32.616, 35.219, 9.005, 10.124, 14.678]

    beamwidth_250 = [18.2] * len(fits250)
    beamwidth_350 = [24.9] * len(fits350)

    # Inputs (adjust to desired wavelength)
    beamwidths = beamwidth_350  # + beamwidth_350
    distances = distances  # + distances
    fits_files = fits350  # + fits350

    print "Started at " + str(datetime.now())

    if not MULTICORE:
        for i, filename in enumerate(fits_files):
            wrapper(filename, distances[i], beamwidths[i], offsets[i], verbose=False)

    else:
        pool = Pool(processes=NCORES)
        pool.map(single_input, izip(fits_files, distances, beamwidths, offsets))
        pool.close()
        # pool.join()
