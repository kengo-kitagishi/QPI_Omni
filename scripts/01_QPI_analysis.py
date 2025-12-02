#位相再構成バッチ処理   いつか書く
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm

WAVELENGTH = 663e-9
NA = 0.95
PIXELSIZE = 3.45e-6/40
OFFAXIS_CENTER = (1623,1621)

BASE_DIR = "Volumes/QPI3/250910_kk/ph1"
BG_DIR = os.path.join(BASE_DIR,"Pos0")

for pos in range(1,44):
    i = f"Pos{pos}"
    POS_DIR = os.path(BASE_DIR,pos)