# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:01:56 2016

@author: phyrct

kfuncs.py

Functions for manipulating Kerr (or any other) images

All of these functions are accessible through the KerrArray attributes
e.g. k=KerrArray('myfile'); k.level_image(). The first 'im' function argument
is automatically added in this case.

If you want to add new functions that's great. There's a few important points:

    * Please make sure they take an image as the first argument

    * Don't give them the same name as functions from the numpy library or
          skimage library if you don't want to override them.

    * The function should not change the shape of the array. Please use crop_image
          before doing the function if you want to do that.

    * After that you're free to treat im as a KerrArray
          or numpy array, it should all behave the same.

"""

import numpy as np
from skimage import exposure,feature,filters,measure,transform
from core import KerrArray
from Stoner import Data

def adjust_contrast(im, lims=(0.1,0.9), percent=True):
    """rescale the intensity of the image. Mostly a call through to
    skimage.exposure.rescale_intensity. The absolute limits of contrast are
    added to the metadata as 'adjust_contrast'

    Parameters
    ----------
    lims: 2-tuple
        limits of rescaling the intensity
    percent: bool
        if True then lims are the give the percentile of the image intensity
        histogram, otherwise lims are absolute

    Returns
    -------
    image: KerrArray
        rescaled image
    """
    if percent:
        vmin,vmax=np.percentile(im,np.array(lims)*100)
    else:
        vmin,vmax=lims[0],lims[1]
    im.metadata['adjust_contrast']=(vmin,vmax)
    im=im.rescale_intensity(in_range=(vmin,vmax)) #clip the intensity
    return im

def clip_intensity(im):
    """clip intensity that lies outside the range allowed by dtype.
    Most useful for float where pixels above 1 are reduced to 1.0 and -ve pixels
    are changed to 0. (Numpy should limit the range on arrays of int dtypes"""
    im=im.rescale_intensity(in_range='dtype')
    return im

def correct_drift(im, ref, threshold=0.005, upsample_factor=50):
    """Align images to correct for image drift.
    Detects common features on the images and tracks them moving.
    Adds 'drift_shift' to the metadata as the (x,y) vector that translated the
    image back to it's origin.

    Parameters
    ----------
    ref: KerrArray or ndarray
        reference image with zero drift
    threshold: float
        threshold for detecting imperfections in images
        (see skimage.feature.corner_fast for details)
    upsample_factor:
        the resolution for the shift 1/upsample_factor pixels registered.
        see skimage.feature.register_translation for more details

    Returns
    -------
    shift: array
        shift vector relative to ref (x drift, y drift)
    transim: KerrArray
        copy of im translated to account for drift"""

    refed=KerrArray(ref,get_metadata=False)
    refed=refed.filter_image(sigma=1)
    refed=refed.corner_fast(threshold=threshold)

    imed=im.clone
    imed=imed.filter_image(sigma=1)
    imed=imed.corner_fast(threshold=threshold)

    shift,err,phase=feature.register_translation(refed,imed,upsample_factor=upsample_factor)
    im=im.translate(translation=(-shift[1],-shift[0])) #x,y
    im.metadata['correct_drift']=(-shift[1],-shift[0])
    return im

def edge_det(filename,threshold1,threshold2):
    '''Detects an edges in an image according to the thresholds 1 and 2.
    Below threshold 1, a pixel is disregarded from the edge
    Above threshold 2, pixels contribute to the edge
    Inbetween 1&2, if the pixel is connected to similar pixels then the pixel conributes to the edge '''
    pass

def filter_image(im, sigma=2):
    """Alias for skimage.filters.gaussian
    """
    return im.gaussian(sigma=sigma)

def level_image(im, poly_vert=1, poly_horiz=1, box=None, poly=None,mode="clip"):
    """Subtract a polynomial background from image

    Keword Arguments:
        poly_vert (int): fit a polynomial in the vertical direction for the image of order
            given. If 0 do not fit or subtract in the vertical direction
        poly_horiz (int): fit a polynomial of order poly_horiz to the image. If 0 given
            do not subtract
        box (array, list or tuple of int): [xmin,xmax,ymin,ymax] define region for fitting. IF None use entire
            image
        poly (list or None): [pvert, phoriz] pvert and phoriz are arrays of polynomial coefficients
            (highest power first) to subtract in the horizontal and vertical
            directions. If None function defaults to fitting its own polynomial.
        mode (str): Either 'crop' or 'norm' - specifies how to handle intensitry values that end up being outside
            of the accepted range for the image.

    Returns:
        A new copy of the processed images.

    Fit and subtract a background to the image. Fits a polynomial of order
    given in the horizontal and vertical directions and subtracts. If box
    is defined then level the *entire* image according to the
    gradient within the box. The polynomial subtracted is added to the
    metadata as 'poly_vert_subtract' and 'poly_horiz_subtract'
    """
    if box is None:
        box=im.max_box
    cim=im.crop_image(box=box)
    (vertl,horizl)=cim.shape
    p_horiz=0
    p_vert=0
    if poly_horiz>0:
        comp_vert = np.average(cim, axis=0) #average (compress) the vertical values
        if poly is not None:
            p_horiz=poly[0]
        else:
            p_horiz=np.polyfit(np.arange(horizl),comp_vert,poly_horiz) #fit to the horizontal
            av=np.average(comp_vert) #get the average pixel height
            p_horiz[-1]=p_horiz[-1]-av #maintain the average image height
        horizcoord=np.indices(im.shape)[1] #now apply level to whole image
        for i,c in enumerate(p_horiz):
            im=im-c*horizcoord**(len(p_horiz)-i-1)
    if poly_vert>0:
        comp_horiz = np.average(cim, axis=1) #average the horizontal values
        if poly is not None:
            p_vert=poly[1]
        else:
            p_vert=np.polyfit(np.arange(vertl),comp_horiz,poly_vert)
            av=np.average(comp_horiz)
            p_vert[-1]=p_vert[-1]-av #maintain the average image height
        vertcoord=np.indices(im.shape)[0]
        for i,c in enumerate(p_vert):
            im=im-c*vertcoord**(len(p_vert)-i-1)
    im.metadata['poly_sub']=(p_horiz,p_vert)
    if mode=="crop":
        im=im.clip_intensity() #saturate any pixels outside allowed range
    elif mode=="norm":
        im=im.normalise()
    return im

def normalise(im):
    """Make data between 0 and 1"""
    im=im.astype(float)
    im=(im-np.min(im))/(np.max(im)-np.min(im))
    return im

def profile_line(img, src, dst, linewidth=1, order=1, mode='constant', cval=0.0):
    """Wrapper for sckit-image method of the same name to get a line_profile.

    Parameters:
        img(KerrArray): Image data to take line section of
        src, dst (2-tuple of int or float): start and end of line profile. If the co-ordinates
            are given as intergers then they are assumed to be pxiel co-ordinates, floats are
            assumed to be real-space co-ordinates using the embedded metadata.
        linewidth (int): the wideth of the profile to be taken.
        order (int 1-3): Order of interpolation used to find image data when not aligned to a point
        mode (str): How to handle data outside of the image.
        cval (float): The constant value to assume for data outside of the image is mode is "constant"

    Returns:
        A :py:class:`Stoner.Data` object containing the line profile data and the metadata from the image.
    """
    scale=img.get("MicronsPerPixel",1.0)
    if isinstance(src[0],float):
        src=(int(src[0]/scale),int(src[1]/scale))
    if isinstance(dst[0],float):
        dst=(int(dst[0]/scale),int(dst[1]/scale))

    result=measure.profile_line(img,src,dst,linewidth,order,mode,cval)
    points=measure.profile._line_profile_coordinates(src, dst, linewidth)[:,:,0]
    ret=Data()
    ret.data=points.T
    ret.setas="xy"
    ret&=np.sqrt(ret.x**2+ret.y**2)*scale
    ret&=result
    ret.column_headers=["X","Y","Distance","Intensity"]
    ret.setas="..xy"
    ret.metadata=img.metadata.copy()
    return ret

def rotate(im, angle):
    """Rotates the image.
    Areas lost by move are cropped, and areas gained are made black (0)

    Parameters
    ----------
    rotation: float
        clockwise rotation angle in radians (rotated about top right corner)

    Returns
    -------
    im: KerrArray
        rotated image
    """
    rot=transform.SimilarityTransform(rotation=angle)
    im.warp(rot)
    im.metadata['transform:rotation']=angle
    return im

def split_image(im):
    """split image into different domains, maybe by peak fitting the histogram?"""
    pass

def translate(im, translation, add_metadata=False):
    """Translates the image.
    Areas lost by move are cropped, and areas gained are made black (0)
    The area not lost or cropped is added as a metadata parameter
    'translation_limits'

    Parameters
    ----------
    translate: 2-tuple
        translation (x,y)

    Returns
    -------
    im: KerrArray
        translated image
    """
    trans=transform.SimilarityTransform(translation=translation)
    im=im.warp(trans)
    if add_metadata:
        im.metadata['translation']=translation
        im.metadata['translation_limits']=translate_limits(im,translation)
    return im

def translate_limits(im, translation):
    """Find the limits of an image after a translation
    After using KerrArray.translate some areas will be black,
    this finds the max area that still has original pixels in

    Parameters
    ----------
    translation: 2-tuple
        the (x,y) translation applied to the image

    Returns
    -------
    limits: 4-tuple
        (xmin,xmax,ymin,ymax) the maximum coordinates of the image with original
        information"""
    t=translation
    s=im.shape
    if t[0]<=0:
        xmin,xmax=0,s[1]-t[0]
    else:
        xmin,xmax=t[0],s[1]
    if t[1]<=0:
        ymin,ymax=0,s[0]-t[1]
    else:
        ymin,ymax=t[1],s[0]
    return (xmin,xmax,ymin,ymax)

def NPPixel_BW(np_image,thresh1,thresh2):
    '''Changes the colour if pixels in a np array according to an inputted threshold'''
    pass