# Camera capture outputs

For ordinary camera captures, save one directly viewable, full-frame 8-bit PNG per requested image. Only crop or save additional files if the user requests it.

# RGB ASM, NH3D and LUT experiment

For new experiments use a KST timestamp and a descriptive name for each top-level run directory, `results/YYYYMMDD-hhmmss_<description>/`. Use `benchtop.dpac_lut_dataset.new_run` or an equivalent exclusive creation check. Keep child folders such as `patterns` and `camera` inside their run. The active dataset uses single-pass DPAC only. Start with the current plan before using any legacy acquisition script or default path. Treat all 2026-11-03 measurements and generated drives as discarded.

Before planning, implementing, running, resuming, or analyzing this bench experiment, read the [current RGB experiment plan and run log](docs/rgb-asm-nh3d-lut-plan.md). Compare the identity LUT (A) and candidate LUT (B) on the same DPAC drive codes. Follow the plan's scene splits, per-channel calibration, and evaluation rules for slot-resolved and integrated RGB captures. Record material protocol changes there with their rationale and affected runs before using the revised protocol.

After each experiment session or interruption, update the same document with the run ID, evidence paths, actual progress, deviations, and the next action. Keep planned work separate from measured results.
