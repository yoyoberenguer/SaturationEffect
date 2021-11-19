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

import unittest
import time
import os


try:
    import pygame
    from pygame.surfarray import array3d, pixels3d, pixels_alpha
except ImportError:
    raise ImportError('\n<pygame> library is missing on your system.'
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

from SaturationEffect import saturation24_mask, build_mask2d_grayscale, build_mask2d_bw, \
    build_mask2d_alpha, saturation32_mask, saturation24, saturation32, saturation24_inplace, \
    saturation32_inplace, saturation24_mask1, saturation32_mask1, saturation_buffer_mask, \
    saturation_buffer_mask_inplace

# numpy is require
try:
    import numpy
except ImportError:
    raise ImportError('\n<numpy> library is missing on your system.'
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")
import SaturationEffect
PROJECT_PATH = list(SaturationEffect.__path__)
os.chdir(PROJECT_PATH[0] + "\\tests")



def display_refresh(screen, image, sat_surface):

    timer = time.time()
    while 1:
        pygame.event.pump()
        screen.fill((0, 0, 0))
        screen.blit(image, (0, 0))
        screen.blit(sat_surface, (640, 0))

        if time.time() - timer > 2:
            break

        pygame.display.flip()


class TestBuildMask2dGrayscale(unittest.TestCase):
    """
    Test Mask build_mask2d_grayscale
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        import sys

        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        numpy.set_printoptions(threshold=sys.maxsize)
        pygame.display.set_caption("build_mask2d_grayscale")

        mask_image = pygame.image.load("../Assets/logo.png").convert()
        mask_image = pygame.transform.smoothscale(mask_image, (640, 480))

        mask = build_mask2d_grayscale(mask_image)
        self.assertIsInstance(mask, numpy.ndarray)
        self.assertEqual(mask.dtype, numpy.float32)
        assert hasattr(mask, "shape"), "\nmask is missing shape attribute."
        s = mask.shape
        self.assertEqual(len(s), 2)
        w, h = tuple(s)
        self.assertEqual(w, 640)
        self.assertEqual(h, 480)


class TestBuildMask2dBW(unittest.TestCase):
    """
    Test Mask build_mask2d_bw
    """

    def runTest(self) -> None:
        """
        :return:  void
        """

        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        pygame.display.set_caption("build_mask2d_bw")

        mask_image = pygame.image.load("../Assets/logo.png").convert()
        mask_image = pygame.transform.smoothscale(mask_image, (640, 480))
        mask = build_mask2d_bw(mask_image)

        self.assertIsInstance(mask, numpy.ndarray)
        self.assertEqual(mask.dtype, numpy.float32)
        assert hasattr(mask, "shape"), "\nmask is missing shape attribute."
        s = mask.shape
        self.assertEqual(len(s), 2)
        w, h = tuple(s)
        self.assertEqual(w, 640)
        self.assertEqual(h, 480)


class TestBuildMask2dAlpha(unittest.TestCase):
    """
    Test Mask build_mask2d_alpha
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        pygame.display.set_caption("build_mask2d_alpha")

        mask_image = pygame.image.load("../Assets/logo_alpha.png").convert_alpha()
        mask_image = pygame.transform.smoothscale(mask_image, (640, 480))
        mask = build_mask2d_alpha(mask_image)

        self.assertIsInstance(mask, numpy.ndarray)
        self.assertEqual(mask.dtype, numpy.float32)
        assert hasattr(mask, "shape"), "\nmask is missing shape attribute."
        s = mask.shape
        self.assertEqual(len(s), 2)
        w, h = tuple(s)
        self.assertEqual(w, 640)
        self.assertEqual(h, 480)


class TestSaturation24Mask(unittest.TestCase):
    """
    Test Mask saturation24_mask
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert()
        image = pygame.transform.smoothscale(image, (640, 480))
        pygame.display.set_caption("saturation24_mask")

        rgb_array = array3d(image)
        sat_surface = saturation24_mask(rgb_array, -0.5, None)
        self.assertEqual(
            sat_surface.get_bitsize(),
            24, msg="\nInvalid surface format expecting 24-bit got %s " % sat_surface.get_bitsize())
        self.assertEqual(
            sat_surface.get_bytesize(), 3,
            msg="\nInvalid surface format expecting 3 bytes depth got %s " % sat_surface.get_bytesize())

        self.assertIsInstance(sat_surface, pygame.Surface)
        self.assertRaises(AssertionError, saturation24_mask, rgb_array, -1.5, None)
        self.assertRaises(AssertionError, saturation24_mask, rgb_array, 1.5, None)
        self.assertRaises(ValueError, saturation24_mask, rgb_array, .5, True)

        # array with wrong data type
        self.assertRaises(AssertionError, saturation24_mask,
                          numpy.full((800, 600, 3), (255.0, 255.0, 255.0),  numpy.float32), 0.2, None)

        # Array and mask with different sizes
        self.assertRaises(AssertionError, saturation24_mask,
                          numpy.full((800, 600, 3), (255.0, 255.0, 255.0), numpy.uint8), 0.2,
                          numpy.full((801, 600), 255, numpy.uint8))

        display_refresh(screen, image, sat_surface)


class TestSaturation24Mask1(unittest.TestCase):
    """
    Test Mask saturation24_mask1
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert()
        image = pygame.transform.smoothscale(image, (640, 480))
        pygame.display.set_caption("saturation24_mask1")

        sat_surface = saturation24_mask1(image, -0.5, None)
        self.assertEqual(
            sat_surface.get_bitsize(),
            24, msg="\nInvalid surface format expecting 24-bit got %s " % sat_surface.get_bitsize())
        self.assertEqual(
            sat_surface.get_bytesize(), 3,
            msg="\nInvalid surface format expecting 3 bytes depth got %s " % sat_surface.get_bytesize())

        self.assertIsInstance(sat_surface, pygame.Surface)
        self.assertRaises(AssertionError, saturation24_mask1, image, -1.5, None)
        self.assertRaises(AssertionError, saturation24_mask1, image, 1.5, None)
        self.assertRaises(ValueError, saturation24_mask1, image, .5, True)

        display_refresh(screen, image, sat_surface)


class TestSaturation32Mask(unittest.TestCase):
    """
    Test saturation32_mask
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (640, 480))

        pygame.display.set_caption("saturation32_mask")

        sat_surface = saturation32_mask(image, +0.5, None)
        self.assertEqual(
            sat_surface.get_bitsize(),
            32, msg="\nInvalid surface format expecting 32-bit got %s " % sat_surface.get_bitsize())
        self.assertEqual(
            sat_surface.get_bytesize(), 4,
            msg="\nInvalid surface format expecting 4 bytes depth got %s " % sat_surface.get_bytesize())

        self.assertIsInstance(sat_surface, pygame.Surface)
        self.assertRaises(AssertionError, saturation32_mask, image, -1.5, None)
        self.assertRaises(AssertionError, saturation32_mask, image, 1.5, None)

        self.assertRaises(ValueError, saturation32_mask, image, .5, True)

        # Convert the image to 24 bit and test
        image = image.convert(24)
        self.assertTrue(image.get_bytesize() == 3)
        self.assertTrue(image.get_bitsize() == 24)
        # Testing with 24-bit without alpha channed this should raise an error
        self.assertRaises(AssertionError, saturation32_mask, image, 0.2, None)

        # Convert image back to 32-bit
        image = image.convert_alpha()
        mask_image = pygame.image.load("../Assets/logo.png").convert()
        mask = build_mask2d_bw(mask_image)

        # This should raise an AssertionError (mismatch width and height)
        self.assertRaises(AssertionError, saturation32_mask, image, 0.2, mask)

        # mask array with wrong dimension (w, h, 3) instead of (w, h
        self.assertRaises(AssertionError, saturation32_mask, image,
                          0.2, numpy.full((640, 480, 3), (1, 1, 1), numpy.uint8))
        # mask array with wrong data type uint8 instead of float
        self.assertRaises(AssertionError, saturation32_mask, image,
                          0.2, numpy.full((640, 480), 1, numpy.uint8))

        mask = numpy.full((640, 480), 1.0, numpy.float32)
        sat_surface = saturation32_mask(image, -0.1, mask)

        display_refresh(screen, image, sat_surface)


class TestSaturation32Mask1(unittest.TestCase):
    """
    Test saturation32_mask1
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (640, 480))

        pygame.display.set_caption("saturation32_mask1")

        rgb_array = pixels3d(image)
        alpha_array = pixels_alpha(image)
        sat_surface = saturation32_mask1(rgb_array, alpha_array, +0.5, None)
        self.assertEqual(
            sat_surface.get_bitsize(),
            32, msg="\nInvalid surface format expecting 32-bit got %s " % sat_surface.get_bitsize())
        self.assertEqual(
            sat_surface.get_bytesize(), 4,
            msg="\nInvalid surface format expecting 4 bytes depth got %s " % sat_surface.get_bytesize())

        self.assertIsInstance(sat_surface, pygame.Surface)
        self.assertRaises(AssertionError, saturation32_mask1, rgb_array, alpha_array, -1.5, None)
        self.assertRaises(AssertionError, saturation32_mask1, rgb_array, alpha_array, 1.5, None)

        self.assertRaises(ValueError, saturation32_mask1, rgb_array, alpha_array, .5, True)
        del rgb_array, alpha_array
        display_refresh(screen, image, sat_surface)


class TestSaturation24(unittest.TestCase):
    """
    Test saturation24
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert()
        image = pygame.transform.smoothscale(image, (640, 480))

        pygame.display.set_caption("saturation24")

        sat_surface = saturation24(array3d(image), +0.5)
        self.assertEqual(
            sat_surface.get_bitsize(),
            24, msg="\nInvalid surface format expecting 24-bit got %s " % sat_surface.get_bitsize())
        self.assertEqual(
            sat_surface.get_bytesize(), 3,
            msg="\nInvalid surface format expecting 3 bytes depth got %s " % sat_surface.get_bytesize())

        self.assertIsInstance(sat_surface, pygame.Surface)
        self.assertRaises(AssertionError, saturation24, array3d(image), -1.5)
        self.assertRaises(AssertionError, saturation24, array3d(image), 1.5)

        sat_surface = saturation24(array3d(image), -0.1)

        display_refresh(screen, image, sat_surface)


class TestSaturation32(unittest.TestCase):
    """
    Test saturation32
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (640, 480))

        pygame.display.set_caption("saturation32")

        arr = pixels3d(image)
        alpha = pixels_alpha(image)
        sat_surface = saturation32(arr, alpha, 0.5)
        self.assertEqual(
            sat_surface.get_bitsize(),
            32, msg="\nInvalid surface format expecting 32-bit got %s " % sat_surface.get_bitsize())
        self.assertEqual(
            sat_surface.get_bytesize(), 4,
            msg="\nInvalid surface format expecting 4 bytes depth got %s " % sat_surface.get_bytesize())

        self.assertIsInstance(sat_surface, pygame.Surface)
        self.assertRaises(AssertionError, saturation32, arr, alpha, -1.5)
        self.assertRaises(AssertionError, saturation32, arr, alpha, 1.5)
        # rgb_array with wrong dimensions (w, h) instead of (w, h, 3)
        self.assertRaises(ValueError, saturation32, numpy.full((640, 480), 255, numpy.uint8), alpha, 0.2)
        # rgb_array with wrong data type
        self.assertRaises(ValueError, saturation32, numpy.full((640, 480, 3), 255, numpy.float32), alpha, 0.2)
        # alpha_array with wrong dimensions (w, h, 3) instead of (w, h)
        self.assertRaises(ValueError, saturation32, arr, numpy.full((640, 480, 3), 255, numpy.uint8), 0.2)
        # alpha_array with wrong data type
        self.assertRaises(ValueError, saturation32, arr, numpy.full((640, 480), 255, numpy.float32), 0.2)

        sat_surface = saturation32(arr, alpha, -0.1)
        del arr, alpha

        display_refresh(screen, image, sat_surface)


class TestSaturation24Inplace(unittest.TestCase):
    """
    Test saturation24_inplace
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert()
        image = pygame.transform.smoothscale(image, (640, 480))

        pygame.display.set_caption("saturation24_inplace")

        arr = pixels3d(image)

        saturation24_inplace(arr, 0.8)
        self.assertRaises(AssertionError, saturation24_inplace, arr, -1.1)
        self.assertRaises(AssertionError, saturation24_inplace, arr, 1.5)
        # rgb_array with wrong dimensions (w, h) instead of (w, h, 3)
        self.assertRaises(ValueError, saturation24_inplace, numpy.full((640, 480), 255, numpy.uint8), 0.2)
        # rgb_array with wrong data type
        self.assertRaises(ValueError, saturation24_inplace, numpy.full((640, 480, 3), 255, numpy.float32), 0.2)

        saturation24_inplace(arr, -0.8)
        del arr
        display_refresh(screen, image, image)


class TestSaturation32Inplace(unittest.TestCase):
    """
    Test saturation32_inplace
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (640, 480))

        pygame.display.set_caption("saturation32_inplace")

        arr = pixels3d(image)

        saturation32_inplace(arr, 0.8)
        self.assertRaises(AssertionError, saturation32_inplace, arr, -1.1)
        self.assertRaises(AssertionError, saturation32_inplace, arr, 1.5)
        # rgb_array with wrong dimensions (w, h) instead of (w, h, 3)
        self.assertRaises(ValueError, saturation32_inplace, numpy.full((640, 480), 255, numpy.uint8), 0.2)
        # rgb_array with wrong data type
        self.assertRaises(ValueError, saturation32_inplace, numpy.full((640, 480, 3), 255, numpy.float32), 0.2)

        saturation32_inplace(arr, -0.8)
        del arr
        display_refresh(screen, image, image)


class TestSaturationBufferMask(unittest.TestCase):
    """
    Test saturation_buffer_mask
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (640, 480))

        pygame.display.set_caption("saturation_buffer_mask")

        arr = (pixels3d(image).transpose(1, 0, 2)).flatten()
        mask = numpy.full(arr.size // 3, 1.0, numpy.float32)
        image = saturation_buffer_mask(arr, 0.1, mask, 640, 480)

        display_refresh(screen, image, image)


class TestSaturationBufferMaskInplace(unittest.TestCase):
    """
    Test saturation_buffer_mask_inplace
    """

    def runTest(self) -> None:
        """
        :return:  void
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 480))

        image = pygame.image.load('../Assets/p1.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (640, 480))

        pygame.display.set_caption("saturation_buffer_mask_inplace")

        arr = (pixels3d(image).transpose(1, 0, 2)).flatten()
        mask = numpy.full(arr.size // 3, 1.0, numpy.float32)
        saturation_buffer_mask_inplace(arr, 0.1, mask, 640, 480)


        display_refresh(screen, image, image)


def run_testsuite():
    """
    test suite

    :return: void
    """

    suite = unittest.TestSuite()

    suite.addTests([
        TestBuildMask2dGrayscale(),
        TestBuildMask2dBW(),
        TestBuildMask2dAlpha(),
        TestSaturation24Mask(),
        TestSaturation24Mask1(),
        TestSaturation32Mask(),
        TestSaturation32Mask1(),
        TestSaturation24(),
        TestSaturation32(),
        TestSaturation24Inplace(),
        TestSaturation32Inplace(),
        TestSaturationBufferMask(),
        TestSaturationBufferMaskInplace()
    ])

    unittest.TextTestRunner().run(suite)
    pygame.quit()


if __name__ == '__main__':
    run_testsuite()

