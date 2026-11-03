import unittest
from benchtop.stage import Stage, validate_target


class StageSafety(unittest.TestCase):
    def test_monitor_connection_cannot_command_motion(self):
        stage = Stage(monitor_only=True)
        stage.opened = stage.armed = True
        with self.assertRaises(RuntimeError):
            stage.move(101.0)

    def test_collision_and_nonfinite_targets_rejected(self):
        for value in [125, 125.001, 300, float('nan'), float('inf'), -1, 98.99, 103.01]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_target(value)

    def test_interval_cannot_bypass_guard(self):
        for interval in [(0, 300), (0, 125), (104, 99), (-1, 103)]:
            with self.assertRaises(ValueError):
                validate_target(101.04, interval)

    def test_backlash_path_must_stay_below_guard(self):
        with self.assertRaises(ValueError):
            validate_target(123.98, (0, 124), .05)
        self.assertEqual(validate_target(101.04), 101.04)


if __name__ == '__main__':
    unittest.main()
