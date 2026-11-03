"""Frozen inverted-code LUT for the LETO green central-patch experiment."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

class GreenDriveLUT:
    def __init__(self, path, sha256):
        path = Path(path)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("Green drive LUT digest mismatch")
        import io
        with np.load(io.BytesIO(payload),allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            self.table = data["table"].copy()
        if (meta["version"] != "green-drive-v1" or meta["wavelength_nm"] != 520
                or meta["patch_xywh"] != [704,284,512,512]
                or self.table.dtype != np.uint8 or self.table.shape != (256,)):
            raise ValueError("Unsupported LUT format or optical configuration")
        self.table.flags.writeable = False

    def apply(self, green_codes):
        """Map a central 512-square nominal uint8 G patch before HDMI packing."""
        if green_codes.dtype != np.uint8 or green_codes.shape != (512,512):
            raise ValueError("Expected 512x512 nominal green uint8 drive codes")
        return cv2.LUT(green_codes,self.table)
