#cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True

"""
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, float32, dstack, full, ones,\
    asarray, ascontiguousarray
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

# CYTHON IS REQUIRED
try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d
    from pygame.image import frombuffer

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

from libc.stdio cimport printf
from libc.stdlib cimport free
from libc.math cimport fmax, fmin

cimport numpy as np

__version__ = 1.01

DEF OPENMP = True

if OPENMP:
    DEF THREADS = 6
else:
    DEF THREADS = 1

DEF SCHEDULE = 'static'


DEF HALF = 1.0/2.0
DEF ONE_THIRD = 1.0/3.0
DEF ONE_FOURTH = 1.0/4.0
DEF ONE_FIFTH = 1.0/5.0
DEF ONE_SIXTH = 1.0/6.0
DEF ONE_SEVENTH = 1.0/7.0
DEF ONE_HEIGHT = 1.0/8.0
DEF ONE_NINTH = 1.0/9.0
DEF ONE_TENTH = 1.0/10.0
DEF ONE_ELEVENTH = 1.0/11.0
DEF ONE_TWELVE = 1.0/12.0
DEF ONE_255 = 1.0/255.0
DEF ONE_360 = 1.0/360.0
DEF TWO_THIRD = 2.0/3.0


cpdef saturation24_mask(array_, shift_, mask_):
    assert -1.0 <= shift_ <= 1.0, '\nArgument shift_ must be in range [-1.0 .. 1.0].'

    assert isinstance(array_, numpy.ndarray),\
        "\nInvalid array expecting a numpy.ndarray type (w, h, 3) got type %s " % type(array_)
    assert array_.dtype == numpy.uint8, \
        "\nInvalid array data type expecting uint8 got type %s " % array_.dtype

    cdef:
        int w, h, mw, mh

    try:
        w, h, bytesize = array_.shape
    except ValueError as e:
        raise ValueError("\nArray argument is invalid "
                         "expecting type (w, h, 3) \n %s " % e)

    if mask_ is not None:
        if not isinstance(mask_, numpy.ndarray):
            raise ValueError(
                "\nMask argument is invalid, expecting a "
                "numpy.ndarray shape (w, h) got type %s " % type(mask_))

        assert mask_.dtype == numpy.float32 or mask_.dtype == numpy.float64, \
            "\nInvalid array data type expecting float32 or float64 got type %s " % mask_.dtype

        try:
            mw, mh = mask_.shape
        except ValueError as e:
            raise ValueError("\nMask argument is invalid "
                             "expecting type (w, h) \n %s " %e)

        assert w == mw and h == mh, "\nArray and mask mismatch width or height"

    return saturation_array24_mask_c(array_, shift_, mask_, w, h)



cpdef saturation24_mask1(surface_, shift_, mask_):
    assert -1.0 <= shift_ <= 1.0, '\nArgument shift_ must be in range [-1.0 .. 1.0].'

    assert isinstance(surface_, pygame.Surface),\
        "\nInvalid surface type, expecting a pygame.Surface type got type %s " % type(surface_)

    cdef:
        int w, h, mw, mh

    w, h = surface_.get_size()

    if mask_ is not None:
        if not isinstance(mask_, numpy.ndarray):
            raise ValueError(
                "\nMask argument is invalid, expecting a "
                "numpy.ndarray shape (w, h) got type %s " % type(mask_))

        assert mask_.dtype == numpy.float32 or mask_.dtype == numpy.float64, \
            "\nInvalid array data type expecting float32 or float64 got type %s " % mask_.dtype

        try:
            mw, mh = mask_.shape
        except ValueError as e:
            raise ValueError("\nMask argument is invalid "
                             "expecting type (w, h) \n %s " %e)

        assert w == mw and h == mh, "\nArray and mask mismatch width or height"

    return saturation_array24_mask_c1(surface_, shift_, mask_, w, h)


cpdef saturation32_mask(surface_, shift_, mask_):


    assert -1.0 <= shift_ <= 1.0, \
        '\nshift_ argument must be in range [-1.0 .. 1.0].'
    assert surface_.get_bytesize() == 4, \
        "\nInvalid surface, the alpha channel is missing. \nImage byte size %s " % surface_.get_bytesize()
    cdef:
        int w, h, mw, mh

    w, h = surface_.get_size()

    if mask_ is not None:

        if not isinstance(mask_, numpy.ndarray):
            raise ValueError(
                "\nMask argument is invalid, expecting a "
                "numpy.ndarray shape (w, h) got type %s " % type(mask_))

        assert mask_.dtype == numpy.float32 or mask_.dtype == numpy.float64, \
            "\nInvalid array data type expecting float32 or float64 got type %s " % mask_.dtype

        try:
            mw, mh = mask_.shape
        except ValueError as e:
            raise ValueError("\nMask argument is invalid "
                             "expecting type (w, h) \n %s " % e)

        assert w == mw and h == mh, "\nArray and mask mismatch width or height"

    return saturation_array32_mask_c(surface_, shift_, mask_, w, h)


cpdef saturation32_mask1(rgb_array_, alpha_array_, shift_, mask_):

        assert -1.0 <= shift_ <= 1.0, \
            '\nshift_ argument must be in range [-1.0 .. 1.0].'
        assert isinstance(rgb_array_, numpy.ndarray), \
            "\nArgument rgb_array is invalid expecting a numpy.ndarray got %s " % type(rgb_array_)
        assert isinstance(alpha_array_, numpy.ndarray), \
            "\nArgument alpha_array_ is invalid expecting a numpy.ndarray got %s " % type(alpha_array_)

        cdef:
            int w, h, mw, mh

        try:
            w, h, bytesize = rgb_array_.shape
        except ValueError as e:
            raise ValueError("\nrgb_array_ argument is invalid "
                             "expecting type (w, h, 3) \n %s " % e)

        assert (w, h) == alpha_array_.shape[:2], \
            "\nrgb_array and alpha_array mismatch width or height"

        if mask_ is not None:

            if not isinstance(mask_, numpy.ndarray):
                raise ValueError(
                    "\nMask argument is invalid, expecting a "
                    "numpy.ndarray shape (w, h) got type %s " % type(mask_))

            assert mask_.dtype == numpy.float32 or mask_.dtype == numpy.float64, \
                "\nInvalid array data type expecting float32 or float64 got type %s " % mask_.dtype

            try:
                mw, mh = mask_.shape
            except ValueError as e:
                raise ValueError("\nMask argument is invalid "
                                 "expecting type (w, h) \n %s " % e)

            assert w == mw and h == mh, "\nArray and mask mismatch width or height"

        return saturation_array32_mask_c1(rgb_array_, alpha_array_, shift_, mask_, w, h)





# APPLY SATURATION TO AN RGB ARRAY
cpdef inline object saturation24(array_, shift_):

    assert -1.0 <= shift_ <= 1.0, '\nArgument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height

    try:
        width, height = array_.shape[:2]
    except (pygame.error, ValueError) as e:
        raise ValueError('\nArray type <array_> not understood \n%s ' % e)

    return saturation_array24_c(array_, shift_, width, height)


cpdef inline object saturation32(array_, alpha_, shift_):

    assert -1.0 <= shift_ <= 1.0, '\nArgument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height, alpha_width, alpha_height

    try:
        width, height = array_.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray <array_> type not understood \n%s ' % e)

    try:
        alpha_width, alpha_height = alpha_.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray <alpha_> type not understood \n%s ' % e)

    assert width == alpha_width and height == alpha_height, \
        "rgb array and alpha channel mismatch width or height "

    return saturation_array32_c(array_, alpha_, shift_, width, height)


# # APPLY SATURATION TO AN RGB BUFFER USING A MASK(COMPATIBLE SURFACE 24 BIT)
# def saturation_buffer_mask(buffer_, shift_, mask_array):
#     return saturation_buffer_mask_c(buffer_, shift_, mask_array)


cpdef saturation_buffer_mask(buffer_, shift_, mask_array, width_, height_):
    return saturation_buffer_mask_c(buffer_, shift_, mask_array, width_, height_)

cpdef saturation_buffer_mask_inplace(buffer_, shift_, mask_array, width_, height_):
    return saturation_buffer_mask_inplace_c(buffer_, shift_, mask_array, width_, height_)


cpdef inline object saturation24_inplace(array_, shift_):

    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    saturation_array24_inplace_c(array_, shift_)

cpdef inline object saturation32_inplace(array_, shift_):

    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    saturation_array32_inplace_c(array_, shift_)

# ----------------IMPLEMENTATION -----------------

cpdef inline object build_mask2d_grayscale(object surface_):
    """
    BUILD A MASK FROM A SURFACE (GRAYSCALE)
    
    Array filled with normalized value corresponding to the grayscale value of each pixel / 255
    * Compatible with surface 24 - 32 bit with or without alpha transparency (the alpha 
      channel is disregarded) 
    * This function return a mask (array) shape (w, h) with normalized value. 
      The value correspond to the gray magnitude of the original image (image converted
      to a grayscale format and normalized) 
        
    :param surface_: pygame.Surface compatible 24-32 bit  
    :return        : Return a numpy.ndarray shape (w, h) with value in range [0.0 ... 1.0]. Normalized array
    """

    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface is invalid, expecting a pygame.Surface got %s " % type(surface_)

    return build_mask2d_grayscale_c(surface_)

cpdef inline object build_mask2d_bw(object surface_):
    """
    BUILD A MASK FROM A SURFACE (BLACK AND WHITE)
    
    Array filled with 1.0 or 0.0 
    * Compatible with surface 24 - 32 bit with or without alpha transparency (the alpha 
      channel is disregarded) 
    * This function return a mask (array) shape (w, h) with normalized value. 
      The values are either 1.0 or 0.0 (1.0 when the grayscale value is >0.0 else 0.0)
    
    :param surface_: pygame.Surface compatible 24-32 bit  
    :return        : Return a numpy.ndarray shape (w, h) with value in range [0.0 ... 1.0]. Normalized array
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface is invalid, expecting a pygame.Surface got %s " % type(surface_)

    return build_mask2d_bw_c(surface_)

cpdef inline object build_mask2d_alpha(object surface_):
    """
    BUILD A MASK FROM A SURFACE (ALPHA)
    
    Array filled with normalized values corresponding to the alpha channel 
    
    * Compatible with surface 32-bit with alpha channel), this method will raised a ValueError
    if the image is not a 32 bit with alpha channel.  
    * This function return a mask (array) shape (w, h) of normalized values, alpha channel values /255
    
    :param surface_: pygame.Surface compatible 32 bit only with alpha channel  
    :return        : Return a numpy.ndarray shape (w, h) with value in range [0.0 ... 1.0] corresponding 
    to the channel alpha values / 255
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface is invalid, expecting a pygame.Surface got %s " % type(surface_)
    assert surface_.get_bytesize() == 4, \
        "\nInvalid surface, the alpha channel is missing. \nImage byte size %s " % surface_.get_bytesize()

    return build_mask2d_alpha_c(surface_)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline np.ndarray[np.float32_t, ndim=3] build_mask2d_grayscale_c(surface_):
    """
    BUILD A MASK FROM A SURFACE (GRAYSCALE)
    
    Array filled with normalized value corresponding to the grayscale value of each pixel / 255
    * Compatible with surface 24 - 32 bit with or without alpha transparency (the alpha 
      channel is disregarded) 
    * This function return a mask (array) shape (w, h) with normalized value. 
      The value correspond to the gray magnitude of the original image (image converted
      to a grayscale format and normalized) 
        
    :param surface_: pygame.Surface compatible 24-32 bit  
    :return        : Return a numpy.ndarray shape (w, h) with value in range [0.0 ... 1.0]. Normalized array
    """

    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface is invalid, expecting a pygame.Surface got %s " % type(surface_)

    cdef:
        int width, height
    width, height = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array
        float [:, :] mask = zeros((width, height), float32)
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float gray_value = 0.0
        int i, j

    try:
        rgb_array = pixels3d(surface_)
    except ValueError as e:
        raise ValueError("\nSurface cannot be referenced.\n%s " % e)

    with nogil:

        for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(height):

                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                gray_value = <float>(r[0] + g[0] + b[0]) / 3.0
                # Normalized value
                mask[i, j] = <float>(gray_value * ONE_255)
    return asarray(mask, dtype=float32)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline np.ndarray[np.float32_t, ndim=2] build_mask2d_bw_c(surface_):
    """
    BUILD A MASK FROM A SURFACE (BLACK AND WHITE)
    
    Array filled with 1.0 or 0.0 
    * Compatible with surface 24 - 32 bit with or without alpha transparency (the alpha 
      channel is disregarded) 
    * This function return a mask (array) shape (w, h) with normalized value. 
      The values are either 1.0 or 0.0 (1.0 when the grayscale value is >0.0 else 0.0)
    
    :param surface_: pygame.Surface compatible 24-32 bit  
    :return        : Return a numpy.ndarray shape (w, h) with value in range [0.0 ... 1.0]. Normalized array
    """

    cdef:
        int width, height
    width, height = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array
        float [:, :] mask = empty((width, height), float32)
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float gray_value = 0.0
        int i, j

    try:
        rgb_array = pixels3d(surface_)
    except ValueError as e:
        raise ValueError("\nSurface cannot be referenced.\n%s " % e)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(height):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                gray_value = (r[0] + g[0] + b[0]) / 3.0
                if gray_value > 0:
                    mask[i, j] = 1.0
                else:
                    mask[i, j] = 0.0
    return asarray(mask)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline np.ndarray[np.float32_t, ndim=2] build_mask2d_alpha_c(surface_):
    """
    BUILD A MASK FROM A SURFACE (ALPHA)

    Array filled with normalized values corresponding to the alpha channel 

    * Compatible with surface 32-bit with alpha channel), this method will raised a ValueError
      if the image is not a 32 bit with alpha channel.  
    * This function return a mask (array) shape (w, h) of normalized values, alpha channel values /255.0

    :param surface_: pygame.Surface compatible 32 bit only with alpha channel  
    :return        : Return a numpy.ndarray shape (w, h) with value in range [0.0 ... 1.0] corresponding 
    to the channel alpha values / 255.0
    """

    cdef:
        int width, height
    width, height = surface_.get_size()

    try:
        alpha_array = pixels_alpha(surface_)
    except ValueError:
        raise ValueError()

    cdef:
        unsigned char [:, :] alpha = alpha_array
        float [:, :] mask = zeros((width, height), float32)
        int i, j

    try:
        rgb_array = pixels3d(surface_)
    except ValueError as e:
        raise ValueError("\nSurface cannot be referenced.\n%s " % e)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(height):
                mask[i, j] = alpha[i, j] * ONE_255
    return asarray(mask)





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline object saturation_array24_mask_c(
        unsigned char [:, :, :] rgb_array_,
        float shift_,
        float [:, :] mask_array,
        int width,
        int height,
        ):
    """
    CHANGE THE SATURATION LEVEL  
    
    INPUT 
    ____
    ARRAY : numpy.ndarray shape (w, h, 3) uint8 containing rgb pixel values 
        
    OUTPUT 
    -----
    NEW SURFACE : Pygame.Surface size (w, h) 24-bit format without alpha channel (RGB format)
    
    * Change the saturation level of a pygame.Surface (compatible with 24 - 32 bit only).
      Transform an RGB model into HSL model and <shift_> saturation value.
    
    * mask_array to determine area to be modified.
      The mask must be a 2d array shape (w, h) of normalized float values (python float)

    :param rgb_array_     : 3d numpy.ndarray shapes (w, h, 3) representing a 24 - 32 bit format pygame.Surface. 
    The surface transparency will be ignored for a 32-bit surface
    :param shift_         : Value must be in range [-1.0 ... 1.0], between [-1.0 ... 0.0] decrease saturation and
    between [0.0  ... 1.0] increase saturation level.
    :param mask_array     : float numpy.ndarray shape (width, height) float values. The mask will be used as  
    a layer to cover the pixels that will not be affected by the saturation effect 
    :param width          : integer; width of the image
    :param height         : integer; height of the image
    :return               : Return a pygame.Surface 24-32 bit without per-pixel information 

    """

    cdef:
        unsigned char [:, :, :] rgb_array = empty((height, width, 3), uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float *m
        float s
        hsl hsl_
        rgb rgb_
        int i, j

    with nogil:

        if mask_array is not None:

            for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(height):

                    # load pixel RGB values
                    r = &rgb_array_[i, j, 0]
                    g = &rgb_array_[i, j, 1]
                    b = &rgb_array_[i, j, 2]
                    m = &mask_array[i, j]
                    if m[0] > 0:

                        hsl_ = struct_rgb_to_hsl(r[0] * ONE_255, g[0] * ONE_255, b[0] * ONE_255)
                        s = min((hsl_.s + shift_), 1.0)
                        s = max(s, 0.0)
                        rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                        rgb_array[j, i, 0] = <unsigned char> (rgb_.r * 255.0 * m[0])
                        rgb_array[j, i, 1] = <unsigned char> (rgb_.g * 255.0 * m[0])
                        rgb_array[j, i, 2] = <unsigned char> (rgb_.b * 255.0 * m[0])
                    else:
                        rgb_array[j, i, 0] = r[0]
                        rgb_array[j, i, 1] = g[0]
                        rgb_array[j, i, 2] = b[0]

        else:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(height):

                    # load pixel RGB values
                    r = &rgb_array_[i, j, 0]
                    g = &rgb_array_[i, j, 1]
                    b = &rgb_array_[i, j, 2]

                    hsl_ = struct_rgb_to_hsl(r[0] * ONE_255, g[0] * ONE_255, b[0] * ONE_255)
                    s = min((hsl_.s + shift_), 1.0)
                    s = max(s, 0.0)
                    rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                    rgb_array[j, i, 0] = <unsigned char> (rgb_.r * 255.0)
                    rgb_array[j, i, 1] = <unsigned char> (rgb_.g * 255.0)
                    rgb_array[j, i, 2] = <unsigned char> (rgb_.b * 255.0)

    return pygame.image.frombuffer(rgb_array, (width, height), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline object saturation_array24_mask_c1(
        object surface_,
        float shift_,
        float [:, :] mask_array,
        int width,
        int height,
        ):
    """
    CHANGE THE SATURATION LEVEL  

    INPUT 
    ____
    SURFACE : pygame.Surface compatible 24-32 bit with or without per pixel transparency

    OUTPUT 
    -----
    NEW SURFACE : Pygame.Surface size (w, h) 24-bit format without alpha channel (RGB format)

    * Change the saturation level of a pygame.Surface (compatible with 24 - 32 bit only).
      Transform an RGB model into HSL model and <shift_> saturation value.

    * mask_array to determine area to be modified.
      The mask must be a 2d array shape (w, h) of normalized float values (python float)

    :param surface_       :  pygame.Surface compatible 24-32 bit with or without per pixel transparency
    The surface transparency will be ignored for a 32-bit surface
    :param shift_         : Value must be in range [-1.0 ... 1.0], between [-1.0 ... 0.0] decrease saturation and
    between [0.0  ... 1.0] increase saturation level.
    :param mask_array     : float numpy.ndarray shape (width, height) float values. The mask will be used as  
    a layer to cover the pixels that will not be affected by the saturation effect 
    :param width          : integer; width of the image
    :param height         : integer; height of the image
    :return               : Return a pygame.Surface 24-32 bit without per-pixel information 

    """
    cdef unsigned char [:, :, :] rgb_array_
    try:
        rgb_array_ = pixels3d(surface_)
    except (ValueError, pygame.error) as e:
        raise ValueError("\nInvalid surface, surface should be 24-32-bit format \n %s " % e)

    cdef:
        unsigned char [:, :, :] rgb_array = empty((height, width, 3), uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float *m
        float s
        hsl hsl_
        rgb rgb_
        int i, j

    with nogil:

        if mask_array is not None:

            for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(height):

                    # load pixel RGB values
                    r = &rgb_array_[i, j, 0]
                    g = &rgb_array_[i, j, 1]
                    b = &rgb_array_[i, j, 2]
                    m = &mask_array[i, j]
                    if m[0] > 0:

                        hsl_ = struct_rgb_to_hsl(r[0] * ONE_255, g[0] * ONE_255, b[0] * ONE_255)
                        s = min((hsl_.s + shift_), 1.0)
                        s = max(s, 0.0)
                        rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                        rgb_array[j, i, 0] = <unsigned char> (rgb_.r * 255.0 * m[0])
                        rgb_array[j, i, 1] = <unsigned char> (rgb_.g * 255.0 * m[0])
                        rgb_array[j, i, 2] = <unsigned char> (rgb_.b * 255.0 * m[0])
                    else:
                        rgb_array[j, i, 0] = r[0]
                        rgb_array[j, i, 1] = g[0]
                        rgb_array[j, i, 2] = b[0]

        else:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(height):

                    # load pixel RGB values
                    r = &rgb_array_[i, j, 0]
                    g = &rgb_array_[i, j, 1]
                    b = &rgb_array_[i, j, 2]

                    hsl_ = struct_rgb_to_hsl(r[0] * ONE_255, g[0] * ONE_255, b[0] * ONE_255)
                    s = min((hsl_.s + shift_), 1.0)
                    s = max(s, 0.0)
                    rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                    rgb_array[j, i, 0] = <unsigned char> (rgb_.r * 255.0)
                    rgb_array[j, i, 1] = <unsigned char> (rgb_.g * 255.0)
                    rgb_array[j, i, 2] = <unsigned char> (rgb_.b * 255.0)

    return pygame.image.frombuffer(rgb_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline object saturation_array32_mask_c1(
        unsigned char[:, :, :] rgb_array_,
        unsigned char[:, :] alpha_array_,
        float shift_,
        float [:, :] mask_array,
        int width,
        int height
        ):
    """

    CHANGE THE SATURATION OF A PYGAME SURFACE (32-bit only)

    INPUT 
    ______
    ARRAY : numpy.ndarray shape (w, h, 4) uint8 representing the RGB pixels values 
    ARRAY : numpy.ndarray shape (w, h) uint8 representing the alpha values


    OUTPUT 
    ------
    NEW SURFACE : Pygame.Surface size (w, h) 32-bit format with alpha channel (RGB format)

    Change the saturation level of a pygame.Surface (compatible with 32-bit only).
    Transform RGB model into HSL model and <shift_> saturation value.
    * mask_array to determine area to be modified. The mask_array should be 
      a 2d array type (w, h) filled with float values 

    :param rgb_array_ : numpy.ndarray shape (w, h, 3) uint8 representing the RGB pixels values
    :param alpha_array_: numpy.ndarray shape (w, h) uint8 representing the alpha values
    :param shift_   : Value must be in range [-1.0 ... 1.0],
                      between [-1.0 ... 0.0] decrease saturation.
                      between [0.0  ... 1.0] increase saturation.
    :param mask_array: float numpy.ndarray shape (width, height) 
    :param width     : integer; width of the image
    :param height    : integer; height of the image
    :return: a pygame.Surface 32-bit with per-pixel information 
    """


    cdef:
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float *m
        float s
        hsl hsl_
        rgb rgb_
        int i, j

    with nogil:

        if mask_array is not None:

            for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(height):

                    # load pixel RGB values
                    r = &rgb_array_[i, j, 0]
                    g = &rgb_array_[i, j, 1]
                    b = &rgb_array_[i, j, 2]
                    m = &mask_array[i, j]

                    if m[0] > 0:

                        # # change saturation
                        hsl_ = struct_rgb_to_hsl(r[0] * ONE_255, g[0] * ONE_255, b[0] * ONE_255)
                        s = hsl_.s
                        s = min((s + shift_), 1.0)
                        s = max(s, 0.0)
                        rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)
                        new_array[j, i, 0] = <unsigned char>(rgb_.r * 255.0 * m[0])
                        new_array[j, i, 1] = <unsigned char>(rgb_.g * 255.0 * m[0])
                        new_array[j, i, 2] = <unsigned char>(rgb_.b * 255.0 * m[0])
                    else:
                        new_array[j, i, 0] = r[0]
                        new_array[j, i, 1] = g[0]
                        new_array[j, i, 2] = b[0]

                    new_array[j, i, 3] = alpha_array_[i, j]

        else:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(height):

                    # load pixel RGB values
                    r = &rgb_array_[i, j, 0]
                    g = &rgb_array_[i, j, 1]
                    b = &rgb_array_[i, j, 2]

                    # # change saturation
                    hsl_ = struct_rgb_to_hsl(r[0] * ONE_255, g[0] * ONE_255, b[0] * ONE_255)
                    s = hsl_.s
                    s = min((s + shift_), 1.0)
                    s = max(s, 0.0)
                    rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)
                    new_array[j, i, 0] = <unsigned char> (rgb_.r * 255.0)
                    new_array[j, i, 1] = <unsigned char> (rgb_.g * 255.0)
                    new_array[j, i, 2] = <unsigned char> (rgb_.b * 255.0)
                    new_array[j, i, 3] = alpha_array_[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline object saturation_array32_mask_c(
        object surface_,
        float shift_,
        float [:, :] mask_array,
        int width,
        int height
        ):
    """
    
    CHANGE THE SATURATION OF A PYGAME SURFACE (32-bit)
    
    INPUT 
    ______
    Pygame.Surface size (w, h) 32-bit format with alpha channel (RGBA format)
      
    
    OUTPUT 
    ------
    NEW SURFACE : Pygame.Surface size (w, h) 32-bit format with alpha channel (RGB format)
    
    Change the saturation level of a pygame.Surface (compatible with 32-bit only).
    Transform RGB model into HSL model and <shift_> saturation value.
    * mask_array to determine area to be modified. The mask_array should be 
      a 2d array type (w, h) filled with float values 
     
    :param surface_ : pygame.Surface compatible 32-bit with alpha channel
    :param shift_   : Value must be in range [-1.0 ... 1.0],
                      between [-1.0 ... 0.0] decrease saturation.
                      between [0.0  ... 1.0] increase saturation.
    :param mask_array: float numpy.ndarray shape (width, height) 
    :param width     : integer; width of the image
    :param height    : integer; height of the image
    :return: a pygame.Surface 32-bit with per-pixel information 
    """

    cdef:
        unsigned char[:, :, :] rgb_array_,
        unsigned char[:, :] alpha_array_,

    try:
        rgb_array_ = pixels3d(surface_)
    except (ValueError, pygame.error) as e:
        raise ValueError("\nInvalid surface, surface should be 32-bit \n %s " % e)
    try:
        alpha_array_ = pixels_alpha(surface_)
    except (ValueError, pygame.error) as e:
        raise ValueError("\Invalid surface, surface should be 32-bit"
                         " with per-pixel transparency \n %s " % e)

    cdef:
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float *m
        float s
        hsl hsl_
        rgb rgb_
        int i, j

    with nogil:

        if mask_array is not None:

            for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(width):

                    # load pixel RGB values
                    r = &rgb_array_[i, j, 0]
                    g = &rgb_array_[i, j, 1]
                    b = &rgb_array_[i, j, 2]
                    m = &mask_array[i, j]

                    if m[0] > 0:

                        # # change saturation
                        hsl_ = struct_rgb_to_hsl(r[0] * ONE_255, g[0] * ONE_255, b[0] * ONE_255)
                        s = hsl_.s
                        s = min((s + shift_), 1.0)
                        s = max(s, 0.0)
                        rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)
                        new_array[j, i, 0] = <unsigned char>(rgb_.r * 255.0 * m[0])
                        new_array[j, i, 1] = <unsigned char>(rgb_.g * 255.0 * m[0])
                        new_array[j, i, 2] = <unsigned char>(rgb_.b * 255.0 * m[0])
                    else:
                        new_array[j, i, 0] = r[0]
                        new_array[j, i, 1] = g[0]
                        new_array[j, i, 2] = b[0]

                    new_array[j, i, 3] = alpha_array_[i, j]

        else:
            for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(width):

                    # load pixel RGB values
                    r = &rgb_array_[i, j, 0]
                    g = &rgb_array_[i, j, 1]
                    b = &rgb_array_[i, j, 2]

                    # # change saturation
                    hsl_ = struct_rgb_to_hsl(r[0] * ONE_255, g[0] * ONE_255, b[0] * ONE_255)
                    s = hsl_.s
                    s = min((s + shift_), 1.0)
                    s = max(s, 0.0)
                    rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)
                    new_array[j, i, 0] = <unsigned char> (rgb_.r * 255.0)
                    new_array[j, i, 1] = <unsigned char> (rgb_.g * 255.0)
                    new_array[j, i, 2] = <unsigned char> (rgb_.b * 255.0)
                    new_array[j, i, 3] = alpha_array_[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline object saturation_array24_c(
        unsigned char [:, :, :] array_,
        float shift_,
        int width,
        int height
):

    """
    CHANGE SATURATION LEVEL 
    
    INPUT 
    ______
    numpy.ndarray shape (w, h, 3) uint8 values
    
    
    OUTPUT 
    ------
    Pygame.Surface size (w, h) 24-bit format without alpha channel (RGB format)
    
    Change the saturation level with a numpy.ndarray shape (w, h, 3) uint8 as argument
    Transform RGB model into HSL model and add <shift_> value to the saturation 
    
    :param array_: numpy.ndarray (w, h, 3) uint8 representing a 24-32 bit surface
    :param shift_: Value must be in range [-1.0 ... 1.0], negative values decrease saturation
    :param width : integer; width of the image 
    :param height: integer; height of the image
    :return: Return a pygame.Surface 24-bit without per-pixel information 
    """

    cdef:
        unsigned char [:, :, :] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float s
        hsl hsl_
        rgb rgb_


    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                r = &array_[i, j, 0]
                g = &array_[i, j, 1]
                b = &array_[i, j, 2]

                hsl_ = struct_rgb_to_hsl(<float>r[0] * ONE_255, <float>g[0] * ONE_255, <float>b[0] * ONE_255)
                s = min((hsl_.s + shift_), 0.5)
                s = max(s, 0.0)
                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                new_array[j, i, 0] = <unsigned char>(rgb_.r * 255.0)
                new_array[j, i, 1] = <unsigned char>(rgb_.g * 255.0)
                new_array[j, i, 2] = <unsigned char>(rgb_.b * 255.0)

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline object saturation_array32_c(
        unsigned char [:, :, :] array_,
        unsigned char [:, :] alpha_,
        float shift_,
        int width,
        int height
):
    """
    CHANGE THE SATURATION LEVEL 
    
    INPUT
    _____
    <array_> numpy.ndarray type (w, h, 3) uint8 filled with RGB pixels 
    <alpha_> numpy.ndarray type (w, h) uint8 representing the alpha channel values
    
    OUTPUT
    _____
    Pygame.Surface size (w, h) 32-bit format with alpha channel (RGBA format)
        
    :param array_: numpy.ndarray shapes (w, h, 4) representing a pygame Surface 32 bit format
    :param alpha_: numpy.ndarray shapes (w, h) containing all alpha values
    :param shift_: Value must be in range [-1.0 ... 1.0], negative values decrease saturation  
    :param width : integer; width of the surface 
    :param height: integer; height of the surface
    :return: a pygame.Surface 32-bit with per-pixel information 
    """

    cdef:
        unsigned char [:, :, :] new_array = empty((height, width, 4), dtype=uint8)
        int i=0, j=0
        float s
        float r, g, b
        hsl hsl_
        rgb rgb_

    with nogil:

        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                # Load RGB
                r, g, b = array_[i, j, 0], array_[i, j, 1], array_[i, j, 2]
                hsl_ = struct_rgb_to_hsl(r * ONE_255, g * ONE_255, b * ONE_255)
                s = hsl_.s
                s = min((s + shift_), 1.0)
                s = max(s, 0.0)
                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)
                new_array[j, i, 0] = <unsigned char>(rgb_.r * 255.0)
                new_array[j, i, 1] = <unsigned char>(rgb_.g * 255.0)
                new_array[j, i, 2] = <unsigned char>(rgb_.b * 255.0)
                new_array[j, i, 3] = alpha_[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_buffer_mask_c(
        unsigned char [::1] buffer_,
        float shift_,
        float [::1] mask_array,
        int width,
        int height
):
    """
    CHANGE THE SATURATION LEVEL OF ALL SELECTED PIXELS FROM A BUFFER.
    
    Transform RGB model into HSL model and <shift_> values.
    mask_array argument cannot be null. The mask should be a buffer type (1d array)
    (filled with normalized float values in range[0.0 ... 1.0]).
    

    :param buffer_: 1d Buffer representing a 24bit format pygame.Surface
    :param shift_ : Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: 1d Buffer mask_array ; must be equal to the buffer length
    :param width  : integer; width of the image
    :param height : integer; height of the image
    :return: a pygame.Surface 24-bit without per-pixel information
    """

    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int b_length, m_length

    try:
        b_length = len(<object>buffer_)
    except ValueError:
        raise ValueError("\nIncompatible buffer type got %s." % type(buffer_))

    if mask_array is not None:
        try:
            m_length = len(<object>mask_array)
        except (ValueError, pygame.error) as e:
            raise ValueError("\nIncompatible buffer type got %s." % type(buffer_))
    else:
        raise ValueError("\nIncompatible buffer type got %s ." % type(buffer_))


    if m_length != (b_length // 3):
        raise ValueError(
            "\nMask length and buffer length mismatch, %s %s" % (b_length, m_length))

    cdef:
        int ii=0
        unsigned char [::1] new_array = empty(b_length, dtype=uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b
        unsigned char *nr
        unsigned char *ng
        unsigned char *nb
        float s
        hsl hsl_
        rgb rgb_

    with nogil:

        for ii in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREADS):
            # load pixel RGB values
            r = &buffer_[ii    ]
            g = &buffer_[ii + 1]
            b = &buffer_[ii + 2]
            nr = &new_array[ii]
            ng = &new_array[ii + 1]
            nb = &new_array[ii + 2]

            if mask_array[ii // 3] > 0.0:

                hsl_ = struct_rgb_to_hsl(<float>r[0] * ONE_255, <float>g[0] * ONE_255, <float>b[0] * ONE_255)

                s = hsl_.s
                s = min((s + shift_), 1.0)
                s = max(s, 0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

            nr[0] = <unsigned char>(rgb_.r * 255.0)
            ng[0] = <unsigned char>(rgb_.g * 255.0)
            nb[0] = <unsigned char>(rgb_.b * 255.0)

    # return pygame.image.fromstring(bytes(new_array), (width, height), "RGB", False)
    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void saturation_buffer_mask_inplace_c(
        unsigned char [::1] buffer_,
        float shift_,
        float [::1] mask_array,
        int width,
        int height
):
    """
    CHANGE THE SATURATION LEVEL OF ALL SELECTED PIXELS FROM A BUFFER.

    Transform RGB model into HSL model and <shift_> values.
    mask_array argument cannot be null. The mask should be a buffer type (1d array)
    (filled with normalized float values in range[0.0 ... 1.0]).


    :param buffer_: 1d Buffer representing a 24bit format pygame.Surface
    :param shift_ : Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: 1d Buffer mask_array ; must be equal to the buffer length
    :param width  : integer; width of the image
    :param height : integer; height of the image
    :return: a pygame.Surface 24-bit without per-pixel information
    """

    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int b_length, m_length

    try:
        b_length = len(<object>buffer_)
    except ValueError:
        raise ValueError("\nIncompatible buffer type got %s." % type(buffer_))

    if mask_array is not None:
        try:
            m_length = len(<object>mask_array)
        except (ValueError, pygame.error) as e:
            raise ValueError("\nIncompatible buffer type got %s." % type(buffer_))
    else:
        raise ValueError("\nIncompatible buffer type got %s ." % type(buffer_))


    if m_length != (b_length // 3):
        raise ValueError(
            "\nMask length and buffer length mismatch, %s %s" % (b_length, m_length))

    cdef:
        int ii=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float s
        hsl hsl_
        rgb rgb_

    with nogil:

        for ii in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREADS):
            # load pixel RGB values
            r = &buffer_[ii    ]
            g = &buffer_[ii + 1]
            b = &buffer_[ii + 2]


            if mask_array[ii // 3] > 0.0:

                hsl_ = struct_rgb_to_hsl(<float>r[0] * ONE_255, <float>g[0] * ONE_255, <float>b[0] * ONE_255)

                s = hsl_.s
                s = min((s + shift_), 1.0)
                s = max(s, 0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

            r[0] = <unsigned char>(rgb_.r * 255.0)
            g[0] = <unsigned char>(rgb_.g * 255.0)
            b[0] = <unsigned char>(rgb_.b * 255.0)





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void saturation_array24_inplace_c(unsigned char [:, :, :] rgb_array_, float shift_):
    """
    CHANGE SATURATION LEVEL (INPLACE) 

    INPUT
    ____
    <rgb_array_> Referenced numpy.ndarray type (w, h, 3) uint8 representing the surface pixels (RGB)
    
    OUTPUT
    _____
    void 

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels, please refer to pygame 
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    e.g:
    shader_saturation_array24_inplace(3darray, 0.2)

    :param rgb_array_: numpy.ndarray shape (w, h, 3) containing RGB values uint8
    :param shift_    : float; value in range[-1.0...1.0], control the saturation level
    :return          : void
    """


    cdef int width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float s
        hsl hsl_
        rgb rgb_


    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):
                r, g, b = &rgb_array_[i, j, 0], &rgb_array_[i, j, 1], &rgb_array_[i, j, 2]
                hsl_ = struct_rgb_to_hsl(<float>r[0] * ONE_255, <float>g[0] * ONE_255, <float>b[0] * ONE_255)

                s = min((hsl_.s + shift_), 0.5)
                s = max(s, 0.0)
                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)
                r[0] = <unsigned char>(rgb_.r * 255.0)
                g[0] = <unsigned char>(rgb_.g * 255.0)
                b[0] = <unsigned char>(rgb_.b * 255.0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void saturation_array32_inplace_c(unsigned char [:, :, :] rgba_array_, float shift_):
    """
    CHANGE SATURATION LEVEL (INPLACE) 
    
    INPUT
    ____
    <rgba_array_> Referenced numpy.ndarray type (w, h, 4) uint8 representing the surface pixels (RGBA)
    
    OUTPUT
    _____
    void 
    
    The Array (rgb_array) must be a numpy array shape (w, h, 4) containing RGBA pixels, please refer to pygame 
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)
    
    e.g:
    shader_saturation_array32_inplace(3darray, 0.2)
    
    :param rgba_array_: numpy.ndarray shape (w, h, 4) containing RGBA values uint8
    :param shift_    : float; value in range[-1.0...1.0], control the saturation level
    :return          : void
    """

    cdef int width, height
    width, height = rgba_array_.shape[:2]

    cdef:
        int i = 0, j = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float s
        hsl hsl_
        rgb rgb_


    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):
                r = &rgba_array_[i, j, 0]
                g = &rgba_array_[i, j, 1]
                b = &rgba_array_[i, j, 2]

                hsl_ = struct_rgb_to_hsl(<float> r[0] * ONE_255, <float> g[0] * ONE_255, <float> b[0] * ONE_255)

                s = min((hsl_.s + shift_), 0.5)
                s = max(s, 0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)
                r[0] = <unsigned char> (rgb_.r * 255.0)
                g[0] = <unsigned char> (rgb_.g * 255.0)
                b[0] = <unsigned char> (rgb_.b * 255.0)