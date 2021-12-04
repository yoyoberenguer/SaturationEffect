# Saturation Effect

![alt text](https://raw.githubusercontent.com/yoyoberenguer/SaturationEffect/main/SaturationEffect/Assets/full_range_saturation.png)

This library contains fast algorithms written in `Cython` and `python` to change 
the saturation level of an image or textures.
This code is using extensively the HSL (Hue, Saturation, Lightness) algorithm 
in order to change the saturation level. Please see also the project `HSL` and 
`HSV` at the following URLs if you need more details regarding those projects

https://github.com/yoyoberenguer/HSV 

https://github.com/yoyoberenguer/HSL

Real time processing

![alt text](SaturationEffect/Assets/SaturationEffect1.gif)

The methods can be used with a large variety of image format such as png, jpg, 
bmp etc, check pygame image format compatibility for more details. 
The image format can be either 24-32 bit with or without the transparency channel 
and works with image containing per-pixel transparency (32 bit). 
However, this library is not compatible with 8-bit format surface.

These algorithms can be used offline or real time processing for 
Indy Game such as pygame or Arcade game as long as the game resolution 
do not exceed 1280x1024. A modern CPU with at least 8 
logical processor is required to keep the game running between 30-60 fps.
   
The algorithms are written using `cython` with OPENMP capability (multi-
processing). This library is build by default with the flag OPENMP, 
providing the best performance for real time processing. 
You can also turn off the multi-processing to balance evenly the 
CPU load between your game and the real time saturation processing. 
Please refer to the section `OPENMP` for more details on how to turn
the multi-processing on/off. 

The saturation effect can be used for various projects such as image 
processing, 2D light effect, spritesheet, demos and video games, video 
image processing, Saturation effect for camera

The project is under the `MIT license`

### Saturation effect definition (from wikipedia) :
The saturation of a color is determined by a combination of light intensity
and how much it is distributed across the spectrum of different wavelengths. 
The purest (most saturated) color is achieved by using just one wavelength 
at a high intensity, such as in laser light. If the intensity drops, then as
a result the saturation drops. To desaturate a color of given intensity in a
subtractive system (such as watercolor), one can add white, black, gray, or 
the hue's complement.

HSL and HSV
Saturation is also one of three coordinates in the HSL and HSV color spaces.
However, in the HSL color space saturation exists independently of lightness. 
That is, both a very light color and a very dark color can be heavily 
saturated in HSL; whereas in the previous definitions—as well as in the HSV
color space—colors approaching white all feature low saturation.

Excitation purity is the relative distance from the white point.
Contours of constant purity can be found by shrinking the spectral locus about the white point.
The points along the line segment have the same hue, with pe increasing from 0 to 1 between the 
white point and position on the spectral locus (position of the color on the horseshoe shape in
the diagram) or (as at the saturated end of the line shown in the diagram) position on the line
of purples.

![alt text](https://raw.githubusercontent.com/yoyoberenguer/SaturationEffect/main/SaturationEffect/Assets/423px-Excitation_Purity.svg.png)


image ref : By I, User:adoniscik, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=3477910


## Installation 
check the link for a newest version https://test.pypi.org/project/SaturationEffect/
```
pip install SaturationEffect 
# or version 1.0.2  
pip install SaturationEffect==1.0.2
```

* version installed 
* Imported module is case sensitive 
```python
>>>from SaturationEffect.saturation import __version__
>>>__version__
```

## Saturation mask 
The library contains 4 methods using an optional mask to determine the pixels 
layer to be changed during the saturation process. 
The mask is build from a pygame.Surface (image) then converted to a 2d numpy.ndarray 
shape (width, height) of normalized float values.
The image format used by the mask can be a JPG, PNG, BMP, 24 -32 bit with or without 
alpha channel. 
Note that the method build_mask2d_alpha using the transparency layer will require a 
surface compatible 32-bit with per-pixel transparency otherwise an error message will 
be raised.

You can create 3 different type of masks:
*  mask build from the grayscale values of the image 
*  mask build from the grayscale values of the image and converted to black & white
*  mask build from the alpha channel of the image 


```cython
# Grayscale mask
cpdef inline object build_mask2d_grayscale(object surface_)
# Black and White mask
cpdef inline object build_mask2d_bw(object surface_)
# Alpha mask
cpdef inline object build_mask2d_alpha(object surface_)
```

## Saturation method details
This version includes various methods spread into two category 24-32 bit compatible 
image format and 32-bit with per-pixel transparency layer.
If you wish to process an image without the transparency layer use a method that 
specify the bitsize 24 (saturation24 for example). 
On the contrary, if the image contains a transparency layer, use any of method with
bitsize 32 such as (saturation32)

Input arguments can be a numpy.ndarray, pygame.Surface or a C -buffer data type.
Choose the right method accordingly
```cython
# Method using a mask (input can be a surface or a numpy.array)
# Compatible 24 -32 bit 
cpdef saturation24_mask(array_, shift_, mask_)
cpdef saturation24_mask1(surface_, shift_, mask_)
# Compatible with 32 bit containing transparency layer
cpdef saturation32_mask(surface_, shift_, mask_)
cpdef saturation32_mask1(rgb_array_, alpha_array_, shift_, mask_)

# Direct saturation, no mask compatible 24 -32 bit
cpdef inline object saturation24(array_, shift_)
cpdef inline object saturation32(array_, alpha_, shift_)

# Input argument is C-buffer data type, the mask is compulsory, omitting 
# the mask will raise an error 
cpdef saturation_buffer_mask(buffer_, shift_, mask_array, width_, height_) 
cpdef saturation_buffer_mask_inplace(buffer_, shift_, mask_array, width_, height_)

# Inplace method, the changes are applied to the surface directly
cpdef inline object saturation24_inplace(array_, shift_)
cpdef inline object saturation32_inplace(array_, shift_)
```

## Quick example

```python
>>> from SaturationEffect import example
```
## Building cython code

#### When do you need to compile the cython code ? 
```
Each time you are modifying any of the following files 
saturation.pyx, saturation.pxd, or any external C code if applicable

1) open a terminal window
2) Go in the main project directory where (saturation.pyx & 
   saturation.pxd files are located)
3) run : python setup_saturation.py build_ext --inplace --force

If you have to compile the code with a specific python 
version, make sure to reference the right python version 
in (python38 setup_saturation.py build_ext --inplace)

If the compilation fail, refers to the requirement section and 
make sure cython and a C-compiler are correctly install on your
 system.
- A compiler such visual studio, MSVC, CGYWIN setup correctly on 
  your system.
  - a C compiler for windows (Visual Studio, MinGW etc) install 
  on your system and linked to your windows environment.
  Note that some adjustment might be needed once a compiler is 
  install on your system, refer to external documentation or 
  tutorial in order to setup this process.e.g https://devblogs.
  microsoft.com/python/unable-to-find-vcvarsall-bat/
```
## OPENMP 
In the main project directory, locate the file `setup_saturation.py`.
The compilation flag /openmp is used by default.
To override the OPENMP feature and disable the multi-processing remove the flag `/openmp`

####
`setup_saturation.py`
```python

ext_modules=cythonize([
        Extension("SaturationEffect.saturation", ["SaturationEffect/saturation.pyx"],
                  extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"], language="c")]),
```
Save the change and build the cython code with the following instruction:

```python setup_saturation.py build_ext --inplace --force```

If the project build successfully, the compilation will end up with the following lines
```
Generating code
Finished generating code
```
If you have any compilation error refer to the section `Building cython code` 
and make sure your system has the following program & libraries installed. 
Check also that the code is not running in a different thread.  
- Pygame version > 3
- numpy >= 1.18
- cython >=0.29.21 (C extension for python) 
- A C compiler for windows (Visual Studio, MinGW etc)

## Credit
Yoann Berenguer 

## Dependencies :
```
numpy >= 1.18
pygame >=2.0.0
cython >=0.29.21
```

## License :

MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person 
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without 
restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following 
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.


## Testing: 
```python
>>>import SaturationEffect
>>>from SaturationEffect.tests.test_saturation import run_testsuite
>>>run_testsuite()
```

## Timing :
In the directory tests under the main project path

C:...tests\python profiling.py
```
TESTING WITH IMAGE 1280x1024 (result in ms)

Performance testing saturation24_mask with mask     per call 0.035846148 overall time 3.58461s for 100 iterations
Performance testing saturation24_mask without mask  per call 0.044081281 overall time 4.40813s for 100
Performance testing saturation32_mask with mask     per call 0.058718479 overall time 5.87185s for 100
Performance testing saturation32_mask without mask  per call 0.056563972 overall time 5.6564s  for 100
Performance testing saturation24                    per call 0.045149282 overall time 4.51493s for 100
Performance testing saturation32                    per call 0.046752571 overall time 4.67526s for 100
Performance testing saturation24_inplace            per call 0.039684722 overall time 3.96847s for 100
Performance testing saturation32_inplace            per call 0.039565034 overall time 3.9565s  for 100
Performance testing saturation_buffer_mask          per call 0.054190551 overall time 5.41906s for 100
Performance testing saturation_buffer_mask_inplace  per call 0.052289168 overall time 5.22892s for 100
```

### Links 
```
WIKIPEDIA https://en.wikipedia.org/wiki/Colorfulness
WIKIPEDIA https://en.wikipedia.org/wiki/HSL_and_HSV
```
