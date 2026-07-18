import ezc3d
import numpy

c3d = ezc3d.c3d("data/c3d/000002_003034_73_207_002_FF_809.c3d")

print(f"Total Frames: {c3d['header']['points']['size']}")
print(f"Marker Frame Rate: {c3d['header']['points']['frame_rate']}")