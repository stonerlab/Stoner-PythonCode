# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 20:46:50 2016

@author: phyrct
"""
import numpy as np
from Stoner.Image import KerrList, KerrArray

def hysteresis(images, fieldlist=None, mask=None):
    """Make a hysteresis loop of the average intensity in the given images

    Parameters
    ----------
    ims: Kerrlist
        list of images for extracting hysteresis
    fieldlist: list or tuple
        list of fields used, if None it will try to get field from image metadata,
        finally it will just use the image index as an x value
    mask: boolean array of same size as an image
        if True then don't include that area in the hysteresis

    Returns
    -------
    hyst: nx2 np.ndarray
        fields, intensities, 2 column numpy array
    """
    if not isinstance(images,KerrList):
        raise TypeError('images must be a KerrList')
    hys_length=len(images)
    if fieldlist is None:
        if 'field' in images[0].keys():
            fieldlist=images.slice_metadata(key='field', values_only=True)
        elif 'ocr_field' in images[0].keys():
            fieldlist=images.slice_metadata(key='ocr_field', values_only=True)
        else:
            fieldlist=range(len(images))
    fieldlist=np.array(fieldlist)
    assert len(fieldlist)==hys_length, 'images and field list must be of equal length'
    assert all([i.shape==images[0].shape for i in images]), 'images must all be same shape'       
    hyst=np.column_stack((fieldlist,np.zeros(hys_length)))
    for i,im in enumerate(images):
        if mask is not None:
            hyst[i,1] = np.average(im[np.invert(mask)])
        else:
            hyst[i,1] = np.average(im)
    return hyst

def correct_drifts(images, refindex, threshold=0.005, upsample_factor=50,box=None):
    """Align images to correct for image drift.
    
    Parameters
    ----------
    images: KerrList
        List of KerrArrays to use
    refindex: int
        index of the reference image to use for zero drift
    Other parameters see KerrArray.correct_drift
    """
    ref=images[refindex]
    ret=images.apply_all('correct_drift', ref,
                                                threshold=threshold,upsample_factor=upsample_factor,box=box)
    return images