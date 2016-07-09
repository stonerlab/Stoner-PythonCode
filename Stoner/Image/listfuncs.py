# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 20:46:50 2016

@author: phyrct
"""
import numpy as np
from Stoner.Image import KerrList, KerrArray

def hysteresis(images, fieldlist=None, box=None):
    """Make a hysteresis loop of the average intensity in the given images

    Parameters
    ----------
    ims: Kerrlist
        list of images for extracting hysteresis
    fieldlist: list or tuple
        list of fields used, if None it will try to get field from imgae metadata
    box: list
        [xmin,xmax,ymin,ymax] region of interest for hysteresis

    Returns
    -------
    hyst: nx2 np.ndarray
        fields, intensities, 2 column numpy array
    """
    if not isinstance(images,KerrList):
        raise TypeError('images must be a KerrList')
    hys_length=len(images)
    if fieldlist is None:
        fieldlist=images.slice_metadata(key='Field', values_only=True)
    fieldlist=np.array(fieldlist)
    assert len(fieldlist)==hys_length, 'images and field list must be of equal length'
    assert all([i.shape==images[0].shape for i in images]), 'images must all be same shape'
    if box is None:
        box=(0,images[0].shape[1],0,images[0].shape[0])

    hyst=np.column_stack((fieldlist,np.zeros(hys_length)))
    for i,im in enumerate(images):
        im=im.crop_image(box=box, copy=True)
        hyst[i,1] = np.average(im)
    return hyst

def correct_drifts(images, refindex, threshold=0.005, upsample_factor=50,box=None):
    """Align images to correct for image drift."""
    print type(images),type(refindex)
    ref=images[refindex]
    ret=images.apply_all('correct_drift', ref,
                                                threshold=threshold,upsample_factor=upsample_factor,box=box)
    return images