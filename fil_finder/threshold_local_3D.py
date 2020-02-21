

from scipy import ndimage as ndi
import numpy as np

'''
Copyright (C) 2011, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

def threshold_local(image, block_size, method='gaussian', offset=0,
                    mode='reflect', param=None):
    """
    Compute a threshold mask image based on local pixel neighborhood.
    Also known as adaptive or dynamic thresholding. The threshold value is
    the weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a
    given function, using the 'generic' method.

    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    block_size : int or tuple
        Odd size of pixel neighborhood which is used to calculate the
        threshold value (e.g. 3, 5, 7, ..., 21, ...).
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighbourhood in
        weighted mean image.
        * 'generic': use custom function (see `param` parameter)
        * 'gaussian': apply gaussian filter (see `param` parameter for custom\
                      sigma value)
        * 'mean': apply arithmetic mean filter
        * 'median': apply median rank filter
        By default the 'gaussian' method is used.
    offset : float, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'.
    param : {int, function}, optional
        Either specify sigma for 'gaussian' method or function object for
        'generic' method. This functions takes the flat array of local
        neighbourhood as a single argument and returns the calculated
        threshold for the centre pixel.
    Returns
    -------
    threshold : (N, M) ndarray
        Threshold image. All pixels in the input image higher than the
        corresponding pixel in the threshold image are considered foreground.
    References
    ----------
    .. [1] http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#adaptivethreshold
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()[:50, :50]
    >>> binary_image1 = image > threshold_local(image, 15, 'mean')
    >>> func = lambda arr: arr.mean()
    >>> binary_image2 = image > threshold_local(image, 15, 'generic',
    ...                                         param=func)
    """

    if isinstance(block_size, int):
        block_size = (block_size,) * len(image.shape)
    else:
        # Block size should match the number of dimensions
        if not len(block_size) == len(image.shape):
            raise ValueError("block_size must have the same length as the "
                             "number of dimensions in the given image when "
                             "a tuple is passed.")

    for b_size in block_size:
        if b_size % 2 == 0:
            raise ValueError("The kwarg ``block_size`` must be odd! Given "
                             "``block_size`` {0} is even.".format(b_size))

    thresh_image = np.zeros(image.shape, 'double')
    if method == 'generic':
        ndi.generic_filter(image, param, block_size,
                           output=thresh_image, mode=mode)
    elif method == 'gaussian':
        sigma = []
        for b_size in block_size:
            if param is None:
                # automatically determine sigma which covers > 99% of distribution
                sigma.append((b_size - 1) / 6.0)
            else:
                sigma.append(param)
        ndi.gaussian_filter(image, sigma, output=thresh_image, mode=mode)
    elif method == 'mean':
        for i, b_size in enumerate(block_size):
            mask = 1. / b_size * np.ones((b_size,))
            # separation of filters to speedup convolution
            ndi.convolve1d(image, mask, axis=i, output=thresh_image, mode=mode)
    elif method == 'median':
        ndi.median_filter(image, block_size, output=thresh_image, mode=mode)

    return thresh_image - offset
