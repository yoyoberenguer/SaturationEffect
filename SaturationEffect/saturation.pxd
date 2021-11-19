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
cimport numpy as np

cdef extern from 'hsl_c.c' nogil:
    struct hsl:
        double h;
        double s;
        double l;
    struct rgb:
        double r;
        double g;
        double b;
    hsl struct_rgb_to_hsl(double r, double g, double b)nogil;
    rgb struct_hsl_to_rgb(double h, double s, double l)nogil;

ctypedef hsl hsl_
ctypedef rgb rgb_


# C-structure to store 3d array index values
cdef struct xyz:
    int x;
    int y;
    int z;

cdef inline np.ndarray[np.float32_t, ndim=3] build_mask2d_grayscale_c(surface_)

cdef inline np.ndarray[np.float32_t, ndim=2] build_mask2d_bw_c(surface_)

cdef inline np.ndarray[np.float32_t, ndim=2] build_mask2d_alpha_c(surface_)

cdef inline object saturation_array24_mask_c(
        unsigned char [:, :, :] rgb_array_,
        float shift_,
        float [:, :] mask_array,
        int width,
        int height,
        )

cdef inline object saturation_array24_mask_c1(
        object surface_,
        float shift_,
        float [:, :] mask_array,
        int width,
        int height,
        )
cdef inline object saturation_array32_mask_c1(
        unsigned char[:, :, :] rgb_array_,
        unsigned char[:, :] alpha_array_,
        float shift_,
        float [:, :] mask_array,
        int width,
        int height
        )
cdef inline object saturation_array32_mask_c(
        object surface_,
        float shift_,
        float [:, :] mask_array,
        int width,
        int height
        )
cdef inline object saturation_array24_c(
        unsigned char [:, :, :] array_,
        float shift_,
        int width,
        int height
)
cdef inline object saturation_array32_c(
        unsigned char [:, :, :] array_,
        unsigned char [:, :] alpha_,
        float shift_,
        int width,
        int height
)
cdef saturation_buffer_mask_c(
        unsigned char [::1] buffer_,
        float shift_,
        float [::1] mask_array,
        int width,
        int height
)
cdef inline void saturation_buffer_mask_inplace_c(
        unsigned char [::1] buffer_,
        float shift_,
        float [::1] mask_array,
        int width,
        int height
)
cdef inline void saturation_array24_inplace_c(unsigned char [:, :, :] rgb_array_, float shift_)
cdef inline void saturation_array32_inplace_c(unsigned char [:, :, :] rgba_array_, float shift_)
