import os
import pickle
import numpy as np
from scipy.spatial.distance import euclidean
from pathlib import Path
from scipy.io import loadmat

from paths import *


subject=1 #[1, 8]
sequence=1 #[1, 2]

matfile=loadmat(MI3_DIR+f'/S{subject}/Seq{sequence}/annot.mat')


