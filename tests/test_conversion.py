import unittest
from colorsys import rgb_to_hsv as rgb_to_hsv_colorsys, \
                     hsv_to_rgb as hsv_to_rgb_colorsys
import numpy as np
from ..rgb2hsv import rgb_to_hsv, hsv_to_rgb

class RGB2HSVTestCase(unittest.TestCase):
    def setUp(self):
        self.test_method = rgb_to_hsv
        self.validator_method = rgb_to_hsv_colorsys
        self.shape = (256, 256, 256, 3)

    def test_nan(self):
        self.assertEqual(self.test_method(0, 0, 0),
                         self.validator_method(0, 0, 0),
                         'incorrect for all 0')

    def test_bounds(self):
        self.assertTrue(check_hsv(self.test_method(255, 255, 0),
                                  self.validator_method(255, 255, 0)),
                        'incorrect for (255,255,0)')

    def test_all_uint8(self):
        validator_array = np.empty(self.shape, dtype=np.uint8)
        test_array = np.empty(self.shape, dtype=np.uint8)
        factor = 255
        for r in range(self.shape[0]):
            for b in range(self.shape[1]):
                for g in range(self.shape[2]):
                    hsv_val = self.validator_method(r, g, b)
                    validator_array[r, g, b, 0] = hsv_val[0] * factor
                    validator_array[r, g, b, 1] = hsv_val[1] * factor
                    validator_array[r, g, b, 2] = hsv_val[2]
                    ####
                    hsv_test = self.test_method(r, g, b)
                    test_array[r, g, b, 0] = hsv_test[0] * factor
                    test_array[r, g, b, 1] = hsv_test[1] * factor
                    test_array[r, g, b, 2] = hsv_test[2]
        # accept 2 pixels deviation - less than 1% relative error
        test_value = np.sum(np.abs(test_array - \
                                   validator_array) > 2)
        self.assertEqual(test_value, 0,
                         'incorrect, got %d colors wrong' % (test_value))

    def test_random_4096(self):
        rgbs = np.random.randint(256, size=(4096, 3))
        for r, g, b in rgbs:
            self.assertTrue(check_hsv(self.test_method(r, g, b),
                                      self.validator_method(r, g, b)),
                            'incorrect for (%d,%d,%d)' % (r, g, b))


class HSV2RGBTestCase(unittest.TestCase):
    def setUp(self):
        self.test_method = hsv_to_rgb
        self.validator_method = hsv_to_rgb_colorsys
        self.shape = (256, 256, 256, 3)

    def test_nan(self):
        self.assertEqual(self.test_method(0, 0, 0),
                         self.validator_method(0, 0, 0),
                         'incorrect for all 0')

    def test_bounds(self):
        self.assertTrue(check_hsv(self.test_method(1, 1, 255),
                                  self.validator_method(1, 1, 255)),
                        'incorrect for (255,255,0)')

    def test_random_4096(self):
        hsvs = np.random.randint(256, size=(4096, 3))
        factor = 1/255.
        for h, s, v in hsvs:
            self.assertTrue(check_hsv(self.test_method(h * factor, s * factor, v),
                                      self.validator_method(h * factor, s * factor, v)),
                            'incorrect for (%d,%d,%d)' % (h, s, v))

    def test_all_uint8(self):
        validator_array = np.empty(self.shape, dtype=np.uint8)
        test_array = np.empty(self.shape, dtype=np.uint8)
        factor = 1/255.
        for h in range(self.shape[0]):
            for s in range(self.shape[1]):
                for v in range(self.shape[2]):
                    hsv_val = self.validator_method(h * factor, s * factor, v)
                    validator_array[h, s, v] = hsv_val
                    ####
                    hsv_test = self.test_method(h * factor, s * factor, v)
                    test_array[h, s, v] = hsv_test
        # accept 1 pixels deviation - less than 0.5% relative error
        test_value = np.sum(np.abs(test_array.astype(np.int16) - \
                                   validator_array.astype(np.int16)) > 1)
        self.assertEqual(test_value, 0,
                         'incorrect, got %d colors wrong' % (test_value))


def check_hsv(a, b):
    a_ = a[0] if a[0] >= 0 else 1+a[0], a[1], a[2]
    return np.isclose(a_, b).all()


def suite_cases():
    suite = unittest.TestSuite()
    ### RGB to HSV
    suite.addTest(RGB2HSVTestCase('test_nan'))
    suite.addTest(RGB2HSVTestCase('test_bounds'))
    suite.addTest(RGB2HSVTestCase('test_random_4096'))
    suite.addTest(RGB2HSVTestCase('test_all_uint8'))
    ### HSV to RGB
    suite.addTest(HSV2RGBTestCase('test_nan'))
    suite.addTest(HSV2RGBTestCase('test_bounds'))
    suite.addTest(HSV2RGBTestCase('test_random_4096'))
    suite.addTest(HSV2RGBTestCase('test_all_uint8'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_cases())
