
'''
Creates figures of the skeletons used in the analysis (convolved+regridded),
and those on the original data.
'''

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
import matplotlib.pyplot as p

p.ion()


def overlap_skeletons(image, big_skel, norm_skel, aplpy_plot=True,
                      save_figure=True, save_name="skeletons",
                      output_file_type="png", rasterized=True,
                      vmin=None, vmax=None):
    '''
    Make a nice aplpy plot of the different skeletons. The original image
    should be passed, and it will be expanded to match the dimensions of the
    regridded skeleton.
    '''

    # Load files in

    image, hdr = fits.getdata(image, header=True)
    image[np.isnan(image)] = 0.0

    norm_skel = fits.getdata(norm_skel)
    norm_skel = (norm_skel > 0).astype(int)

    big_skel = fits.getdata(big_skel)
    big_skel = (big_skel > 0).astype(int)

    big_skel_hdu = fits.PrimaryHDU(big_skel, header=hdr)

    # The original image and the normal skeleton should have the same
    # dimensions.
    assert image.shape == norm_skel.shape

    image = zoom(image,
                 [i/float(j) for j, i in zip(image.shape, big_skel.shape)])

    assert image.shape == big_skel.shape

    hdr['NAXIS1'] = image.shape[1]
    hdr['NAXIS2'] = image.shape[0]

    image_hdu = fits.PrimaryHDU(image, header=hdr)

    norm_skel_zoom = \
        zoom(norm_skel,
             [i/float(j) for j, i in zip(norm_skel.shape, big_skel.shape)],
             order=0)

    assert norm_skel_zoom.shape == big_skel.shape

    norm_skel_hdu = fits.PrimaryHDU(norm_skel_zoom, header=hdr)

    if aplpy_plot:

        try:
            import aplpy
        except ImportError:
            ImportError("Cannot import aplpy. Do not enable aplpy.")

        fig = aplpy.FITSFigure(image_hdu)

        fig.show_grayscale(invert=True, stretch="arcsinh", vmin=vmin,
                           vmax=vmax)

        fig.tick_labels.set_xformat('hh:mm')
        fig.tick_labels.set_yformat('dd:mm')

        fig.tick_labels.set_font(size='large', weight='medium',
                                 stretch='normal', family='sans-serif',
                                 style='normal', variant='normal')

        fig.axis_labels.set_font(size='large', weight='medium',
                                 stretch='normal', family='sans-serif',
                                 style='normal', variant='normal')

        # fig.add_grid()

        # NOTE! - rasterization will only work with my fork of aplpy!
        # git@github.com:e-koch/aplpy.git on branch 'rasterize_contours'
        fig.show_contour(norm_skel_hdu, colors="red", linewidths=1.5,
                         rasterize=True)

        fig.show_contour(big_skel_hdu, colors="blue", rasterize=True)

        fig.show_colorbar()
        fig.colorbar.set_label_properties(size='large', weight='medium',
                                          stretch='normal',
                                          family='sans-serif',
                                          style='normal', variant='normal')
        fig.colorbar.set_axis_label_text('Surface Brightness (MJy/sr)')
        fig.colorbar.set_axis_label_font(size='large', weight='medium',
                                         stretch='normal',
                                         family='sans-serif',
                                         style='normal', variant='normal')

        if save_figure:
            fig.save(save_name+"."+output_file_type)
            fig.close()

    else:
        # Using matplotlib
        from astropy.visualization import scale_image

        scaled_image = scale_image(image, scale='asinh', asinh_a=0.005)

        p.imshow(scaled_image, interpolation='nearest', origin='lower',
                 cmap='binary')

        p.contour(norm_skel_zoom, colors='r', linewidths=2)
        p.contour(big_skel, colors='b', linewidths=1)

        if save_figure:
            p.save(save_name+"."+output_file_type, rasterized=rasterized)
            p.close()
        else:
            p.show(block=True)

if __name__ == "__main__":

    import sys
    import os

    # Paths
    regridded_path = sys.argv[1]
    norm_path = sys.argv[2]

    regrid_skels = [f+"/"+f+"_skeletons.fits" for f in
                    os.listdir(regridded_path) if
                    os.path.isdir(os.path.join(regridded_path, f))]
    norm_skels = [f+"/"+f+"_skeletons.fits" for f in
                  os.listdir(norm_path) if
                  os.path.isdir(os.path.join(norm_path, f))]

    regrid_skels.sort()
    norm_skels.sort()

    # The folders should match in each directory

    for f_reg in regrid_skels:
        if f_reg not in norm_skels:
            print regrid_skels
            print norm_skels
            raise Warning("Folder lists must match. Check inputted paths.")
            break
        else:
            pass

    images = [f+"/"+f+"_regrid_convolved.fits" for f in os.listdir(norm_path)
              if os.path.isdir(os.path.join(norm_path, f))]

    images.sort()

    vmins = [60, 15, 15, 18, 10, 15, 10, 25, 20, 25, 15, 22, 4, 10]

    # Now create the images
    for img, big_skel, norm_skel, vmin in zip(images, regrid_skels, norm_skels, vmins):

        print "Image: {}".format(norm_path+img)
        print "Regridded Skeleton: {}".format(regridded_path+big_skel)
        print "Normal Skeleton: {}".format(norm_path+norm_skel)

        image_name = img.split("/")[-1]

        overlap_skeletons(norm_path+img, regridded_path+big_skel,
                          norm_path+norm_skel, save_figure=True,
                          save_name=image_name[:-5]+"_online_fig",
                          output_file_type="pdf", vmin=vmin)
