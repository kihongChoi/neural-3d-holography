"""Capture known SLM-plane amplitude masks without assuming relay magnification."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .hardware import Bench


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path('configs/benchtop_rgb80.json').read_text())
    config.update(settle_s=0.7, average_frames=3)
    h, w = config['slm_shape']
    yy, xx = np.indices((h, w))
    checker = np.where((xx+yy) % 2, np.pi/2, -np.pi/2).astype(np.float32)
    records = []
    with Bench(config) as bench:
        c = bench.camera
        geometry = {name: int(getattr(c, name).GetValue()) for name in
                    ['Width', 'Height', 'OffsetX', 'OffsetY',
                     'BinningHorizontal', 'BinningVertical',
                     'DecimationHorizontal', 'DecimationVertical']}
        if geometry['BinningHorizontal'] != 1 or geometry['BinningVertical'] != 1 or geometry['DecimationHorizontal'] != 1 or geometry['DecimationVertical'] != 1:
            raise RuntimeError('Native unbinned sensor sampling required')

        def capture(name, phase, **details):
            bench.show(phase)
            Image.fromarray(bench.last_encoded).save(out/f'{name}_slm.png')
            raw_sum = None
            metadata = []
            c.BeginAcquisition()
            bench.acquiring = True
            try:
                for i in range(4):
                    frame = c.GetNextImage(3000)
                    try:
                        if frame.IsIncomplete():
                            raise RuntimeError('Incomplete frame')
                        raw = frame.GetNDArray().copy()
                        if i:
                            raw_sum = raw.astype(np.float32) if raw_sum is None else raw_sum + raw
                            metadata.append(dict(frame_id=int(frame.GetFrameID()), timestamp_ns=int(frame.GetTimeStamp())))
                        if i == 3:
                            Image.fromarray(raw).save(out/f'{name}_raw16.png')
                    finally:
                        frame.Release()
                    bench.app.processEvents()
            finally:
                c.EndAcquisition()
                bench.acquiring = False
            bench.verify_display()
            raw = raw_sum/(3*65535.)
            green = (raw[0::2, 1::2]+raw[1::2, 0::2])/2
            np.save(out/f'{name}_green_half.npy', green)
            scale = max(float(np.quantile(green, .999)), 0.001)
            Image.fromarray(np.uint8(np.clip(green/scale, 0, 1)**(1/2.2)*255)).save(out/f'{name}_preview.png')
            row = dict(name=name, time=time.time(), exposure_us=float(c.ExposureTime.GetValue()),
                       delay_us=float(c.TriggerDelay.GetValue()), mean=float(green.mean()),
                       p999=float(np.quantile(green,.999)), saturation=float((raw>=.98).mean()),
                       frames=metadata, **details)
            records.append(row)
            (out/'acquisition.json').write_text(json.dumps(dict(config=config, geometry=geometry,
                sensor_model=c.TLDevice.DeviceModelName.GetValue(), records=records,
                half_sensor_coordinates='Each analysis cell spans 2x2 native Bayer pixels; green sample center is (2*x+0.5,2*y+0.5).',
                method='SLM-plane amplitude via +/-pi/2 checkerboard rejection; zero phase inside known masks. No propagation distance or magnification used to generate masks.'), indent=2))
            print(json.dumps(row), flush=True)
            return green

        exposure = 700.
        for attempt in range(6):
            bench.set_timing(exposure, 6022.)
            flat = capture(f'flat_{attempt}', np.zeros((h,w),np.float32))
            if np.quantile(flat,.999) < .7:
                break
            exposure /= 2
        else:
            raise RuntimeError('Cannot obtain unsaturated flat')
        capture('rejected', checker)
        mask = (abs(xx-(w-1)/2)<w/2-80)&(abs(yy-(h-1)/2)<h/2-80)
        capture('inset_rectangle', np.where(mask,0,checker).astype(np.float32),
                slm_edges_xy=[80,80,w-80,h-80], slm_size_xy=[w-160,h-160])
        if args.grid:
            for i,(cy,cx) in enumerate((cy,cx) for cy in [240,540,840] for cx in [360,960,1560]):
                mask = (abs(xx-(cx-.5))<64)&(abs(yy-(cy-.5))<64)
                capture(f'patch_{i:02d}', np.where(mask,0,checker).astype(np.float32),
                        slm_center_xy=[cx-.5,cy-.5], slm_size_xy=[128,128])
        capture('flat_end', np.zeros((h,w),np.float32))
        del c


if __name__ == '__main__':
    main()
