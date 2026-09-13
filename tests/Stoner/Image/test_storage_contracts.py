"""Characterise image storage boundaries independently of backend implementation."""

import numpy as np

from Stoner.Image import ImageFile, ImageStack


def test_ragged_stack_extents_order_and_item_writeback():
    """Retain unequal frame extents, names, metadata and writes through stack items."""
    small = ImageFile(np.arange(6, dtype=np.uint16).reshape(2, 3))
    small.filename = "small"
    small.metadata["Field"] = 10
    large = ImageFile(np.full((3, 4), 100, dtype=np.uint16))
    large.filename = "large"
    large.metadata["Field"] = -10
    stack = ImageStack([small, large])
    assert stack.shape == (2, 3, 4)
    assert stack[0].shape == (2, 3)
    assert stack[1].shape == (3, 4)
    assert stack["small"].metadata["Field"] == 10
    assert stack["large"].metadata["Field"] == -10
    assert stack.imarray.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(stack[0].image, np.arange(6).reshape(2, 3))
    stack[0][1, 2] = 77
    assert stack.imarray[0, 1, 2] == 77
    stack[0].mask[1, 2] = True
    assert np.ma.getmaskarray(stack.imarray)[0, 1, 2]
    stack[0].mask[1, 2] = False
    assert stack[0][1, 2] == 77


def test_image_mask_reduction_and_recovery():
    """Exclude an integer pixel without destroying its value or changing dtype."""
    image = ImageFile(np.array([[1, 2], [3, 100]], dtype=np.uint16))
    image.mask = np.zeros(image.shape, dtype=bool)
    image.mask[1, 1] = True
    assert image.mean() == 2
    assert image.image.dtype == np.dtype("uint16")
    image.mask[1, 1] = False
    assert image[1, 1] == 100
    assert image.mean() == 26.5


def test_stack_clone_state_independence():
    """Keep clone pixel, exclusion and per-frame metadata changes independent."""
    stack = ImageStack(np.arange(24).reshape(2, 3, 4))
    stack[0].mask[0, 1] = True
    stack[0].metadata["Field"] = 10
    cloned = stack.clone
    cloned[0][0, 0] = 100
    cloned[0].mask[0, 1] = False
    cloned[0].metadata["Field"] = 20
    assert stack[0][0, 0] == 0
    assert stack[0].mask[0, 1]
    assert stack[0].metadata["Field"] == 10


def test_legacy_padding_is_unmasked_zero():
    """Record existing padding; explicit exclusion is a proposed migration change."""
    stack = ImageStack([np.ones((2, 3)), np.ones((3, 4))])
    assert stack[0].shape == (2, 3)
    np.testing.assert_array_equal(stack.imarray[0, 2, :], 0)
    np.testing.assert_array_equal(stack.imarray[0, :, 3], 0)
    assert not np.ma.getmaskarray(stack.imarray).any()
