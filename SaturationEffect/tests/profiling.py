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
import timeit
import os

try:
    import pygame
    from pygame.surfarray import array3d, pixels3d, pixels_alpha
except ImportError:
    raise ImportError('\n<pygame> library is missing on your system.'
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

from SaturationEffect import build_mask2d_grayscale, saturation24_mask, saturation32_mask, \
    saturation24, saturation32, saturation24_inplace, saturation32_inplace, \
    saturation_buffer_mask, saturation_buffer_mask_inplace
import numpy

import SaturationEffect
PROJECT_PATH = list(SaturationEffect.__path__)
os.chdir(PROJECT_PATH[0] + "\\tests")

result = {}

pygame.init()
screen = pygame.display.set_mode((640, 480))

mask_image = pygame.image.load("../Assets/logo.png").convert()
mask_image = pygame.transform.smoothscale(mask_image, (1280, 1024))
mask = build_mask2d_grayscale(mask_image)

image = pygame.image.load('../Assets/p1.png').convert()
image = pygame.transform.smoothscale(image, (1280, 1024))


N = 100

saturation24_mask
arr = array3d(image)
t = timeit.timeit("saturation24_mask(arr, 0.5, mask)",
                  "from __main__ import saturation24_mask, arr, mask", number=N)
print("\nPerformance testing saturation24_mask with mask per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation24_mask with mask'] = round(float(t)/float(N), 10)

t = timeit.timeit("saturation24_mask(arr, 0.5, None)",
                  "from __main__ import saturation24_mask, arr", number=N)
print("\nPerformance testing saturation24_mask without mask per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation24_mask no mask'] = round(float(t)/float(N), 10)

image = pygame.image.load('../Assets/p1.png').convert_alpha()
image = pygame.transform.smoothscale(image, (1280, 1024))
saturation32_mask
t = timeit.timeit("saturation32_mask(image, 0.5, mask)",
                  "from __main__ import saturation32_mask, image, mask", number=N)
print("\nPerformance testing saturation32_mask with mask per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation32_mask with mask'] = round(float(t)/float(N), 10)

t = timeit.timeit("saturation32_mask(image, 0.5, None)",
                  "from __main__ import saturation32_mask, image", number=N)
print("\nPerformance testing saturation32_mask without mask per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation32_mask no mask'] = round(float(t)/float(N), 10)

saturation24
arr = pixels3d(image)
t = timeit.timeit("saturation24(arr, 0.5)",
                  "from __main__ import saturation24, arr", number=N)
print("\nPerformance testing saturation24 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation24'] = round(float(t)/float(N), 10)

image = pygame.image.load('../Assets/p1.png').convert_alpha()
image = pygame.transform.smoothscale(image, (1280, 1024))
saturation32
arr = pixels3d(image)
alpha = pixels_alpha(image)
t = timeit.timeit("saturation32(arr, alpha,  0.5)",
                  "from __main__ import saturation32, arr, alpha", number=N)
print("\nPerformance testing saturation32 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation32'] = round(float(t)/float(N), 10)

saturation24_inplace
arr = pixels3d(image)
t = timeit.timeit("saturation24_inplace(arr, 0.5)",
                  "from __main__ import saturation24_inplace, arr", number=N)
print("\nPerformance testing saturation24_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation24_inplace'] = round(float(t)/float(N), 10)

image = pygame.image.load('../Assets/p1.png').convert_alpha()
image = pygame.transform.smoothscale(image, (1280, 1024))
saturation32_inplace
arr = pixels3d(image)
t = timeit.timeit("saturation32_inplace(arr, 0.5)",
                  "from __main__ import saturation32_inplace, arr", number=N)
print("\nPerformance testing saturation32_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation32_inplace'] = round(float(t)/float(N), 10)

image = pygame.image.load('../Assets/p1.png').convert()
image = pygame.transform.smoothscale(image, (1280, 1024))
rgb_array = array3d(image)
buffer_ = rgb_array.flatten()
mask = numpy.full(1280 * 1024, 1.0, numpy.float32)

saturation_buffer_mask
t = timeit.timeit("saturation_buffer_mask(buffer_, 0.5, mask, 1280, 1024)",
                  "from __main__ import saturation_buffer_mask, buffer_, mask", number=N)
print("\nPerformance testing saturation_buffer_mask per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation_buffer_mask'] = round(float(t)/float(N), 10)

image = pygame.image.load('../Assets/p1.png').convert()
image = pygame.transform.smoothscale(image, (1280, 1024))
rgb_array = array3d(image)
buffer_ = rgb_array.flatten()
mask = numpy.full(1280 * 1024, 1.0, numpy.float32)

saturation_buffer_mask_inplace
t = timeit.timeit("saturation_buffer_mask_inplace(buffer_, 0.5, mask, 1280, 1024)",
                  "from __main__ import saturation_buffer_mask_inplace, buffer_, mask", number=N)
print("\nPerformance testing saturation_buffer_mask_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
result['saturation_buffer_mask_inplace'] = round(float(t)/float(N), 10)

sorted_result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
for k, v in sorted_result.items():
    print("\n ",  k, v)