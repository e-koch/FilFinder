
import numpy as np

from ..utilities import pad_image


def test_pad_image():

    pixels = (np.array([0, 0, 0]), np.array([0, 1, 2]))

    image = np.zeros((1, 3))
    image[pixels] = 1.

    pixel_extents = [(0, 0), (0, 2)]

    padded_image = pad_image(image, pixel_extents, pad_size=1)

    assert padded_image.shape == (3, 5)

    assert (padded_image[pixels[0] + 1, pixels[1] + 1] == 1.).all()

    padded_image = pad_image(image, pixel_extents, pad_size=2)

    assert padded_image.shape == (5, 7)

    assert (padded_image[pixels[0] + 2, pixels[1] + 2] == 1.).all()

    # Check changing the constant
    padded_image = pad_image(image, pixel_extents, pad_size=1, constant=np.NaN)

    assert padded_image.shape == (3, 5)

    assert (padded_image[pixels[0] + 1, pixels[1] + 1] == 1.).all()

    assert np.isnan(padded_image[:1]).all()
    assert np.isnan(padded_image[-1:]).all()
    assert np.isnan(padded_image[:, :1]).all()
    assert np.isnan(padded_image[:, -1:]).all()
