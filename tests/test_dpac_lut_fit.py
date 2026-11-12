import unittest
import numpy as np

from benchtop.dpac_lut_fit import candidates, apply_lut


class DPACLUT(unittest.TestCase):
    def test_candidate_endpoints_monotonicity_and_quantization(self):
        tables = candidates()
        self.assertEqual(len(tables), 7)
        for table in tables.values():
            self.assertEqual(table.dtype, np.uint8)
            self.assertEqual(table.shape, (256,))
            self.assertEqual((int(table[0]), int(table[255])), (0, 255))
            self.assertTrue(np.all(np.diff(table.astype(int)) >= 0))
        self.assertEqual(int(tables['k128_+12'][128]), 140)
        self.assertEqual(int(tables['k128_-12'][128]), 116)

    def test_lut_applied_once_to_intended_rgb_channel_without_mutating_input(self):
        codes = np.full((960, 1728, 3), 128, np.uint8)
        tables = np.tile(candidates()['identity'], (3, 1))
        tables[1] = candidates()['k128_+12']
        result = apply_lut(codes, tables)
        np.testing.assert_array_equal(result[0, 0], [128, 140, 128])
        self.assertTrue(np.all(codes == 128))
        np.testing.assert_array_equal(apply_lut(codes, np.tile(candidates()['identity'], (3, 1))), codes)


if __name__ == '__main__':
    unittest.main()
