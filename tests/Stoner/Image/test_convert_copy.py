"""Check storage independence when image conversion keeps the existing dtype."""

import numpy as np
import pytest

from Stoner import ImageStack
from Stoner.Image.imagefuncs import convert


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float64])
def test_same_dtype_copy(dtype):
    """Explicit copying detaches pixels while the default retains storage."""
    source = np.arange(12, dtype=dtype).reshape(3, 4)
    assert np.shares_memory(convert(source, dtype), source)
    result = convert(source, dtype, force_copy=True)
    np.testing.assert_array_equal(result, source)
    assert result.dtype == source.dtype
    assert not np.shares_memory(result, source)
    result[0, 0] = 1
    assert source[0, 0] == 0


def test_stack_same_dtype_copy_preserves_mask():
    """The public stack operation replaces pixel storage and restores its mask."""
    stack = ImageStack(np.arange(24, dtype=np.uint8).reshape(2, 3, 4))
    stack[0].mask[0, 1] = True
    original = stack.imarray
    mask = original.mask.copy()
    result = stack.convert(np.uint8, force_copy=True)
    assert result is stack
    np.testing.assert_array_equal(stack.imarray.data, original.data)
    np.testing.assert_array_equal(stack.imarray.mask, mask)
    assert stack.imarray.dtype == original.dtype
    assert not np.shares_memory(stack.imarray.data, original.data)
    stack[0][0, 0] = 99
    assert original.data.flat[0] == 0
