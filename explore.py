import os
import pickle
import numpy as np
from scipy.spatial.distance import euclidean
from pathlib import Path

from paths import *


#infile=open(TDPW_TRAIN_DIR+'/courtyard_arguing_00.pkl', 'rb')
infile=open(TDPW_TEST_DIR+'/downtown_arguing_00.pkl', 'rb')
seq00=pickle.load(infile, encoding='latin1')
infile.close()

girl=seq00['jointPositions'][0]
guy=seq00['jointPositions'][1]


girl_diff=[max(abs(g1-g2)) for g1, g2 in zip(girl[1:], girl[0:-1])]


dirs = [ TDPW_TRAIN_DIR, TDPW_VAL_DIR, TDPW_TEST_DIR ]

for d in dirs:
    num_imgs=0
    pkls=list(Path(d).glob('*'))
    for p in pkls:
        infile=open(p,'rb')
        seq=pickle.load(infile,encoding='latin1')
        infile.close()
        num_imgs += len(seq['img_frame_ids'])
    print('num_imgs: ' + str(num_imgs))


