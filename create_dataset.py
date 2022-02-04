#!/usr/bin/env python

# Main script to create dataset
# Combines the following datasets in JSON format:
#   - MPII
#   - LSP
#   - LSPET
#   - COCO2017
#   - 3DPW
#   - MPI-INF-3DHP


import time
import argparse
import json
import numpy as np

from paths import *

from pylsp import PyLSP
from pylspet import PyLSPET
from pympii import PyMPII
from pycoco import PyCOCO
from py3dpw import Py3DPW
from pymi3 import PyMI3

from utils import GID

# Command-line arguments
ap=argparse.ArgumentParser()
ap.add_argument("-m", "--mpii", action='store_true', help='Use MPII dataset')
ap.add_argument("-l", "--lsp", action='store_true', help='Use LSP dataset')
ap.add_argument("-e", "--lspet", action='store_true', help='Use LSPET dataset')
ap.add_argument("-c", "--coco", action='store_true', help='Use COCO dataset')
ap.add_argument("-p", "--3dpw", action='store_true', help='Use 3DPW dataset')
ap.add_argument("-f", "--3dhp", action='store_true', help='Use MPI-INF-3DHP dataset')
ap.add_argument("-s", "--set", required=True, help="Set: train, val, test or toy")
args=vars(ap.parse_args())

# If no dataset is specified, generate them all
USE_MPII=args['mpii']
USE_LSP=args['lsp']
USE_LSPET=args['lspet']
USE_COCO=args['coco']
USE_3DPW=args['3dpw']
USE_3DHP=args['3dhp']
if not (USE_MPII or USE_LSP or USE_LSPET or USE_COCO or USE_3DPW or USE_3DHP):
    USE_MPII=USE_LSP=USE_LSPET=USE_COCO=USE_3DPW=USE_3DHP=True

which_set = args['set']
assert(which_set in ['train','val','test','toy'])

# Suppress scientific notation in numpy prints (really annoying for pixel locations)
np.set_printoptions(suppress=True)

# Create a global ID object
gid = GID()

# QUICK TEST
#tdpw = Py3DPW(TDPW_TRAIN_DIR, TDPW_VAL_DIR, TDPW_TEST_DIR, TDPW_IMG_DIR)
#tdpw.disp_annotations(0)
#tdpw.disp_annotations(72797)
#tdpw.disp_annotations(74619)
#exit()
# END QUICK TEST

# Create the data loader objects. If memory issues, may need to do one at a time.
# LSP Recommended split: first 1000 training, last 1000 testing
if USE_LSP:
    lsp = PyLSP(BTFM_BASE, LSP_DIR, LSP_CSV, UPI_S1H_DIR)
if USE_LSPET:
    # LSPET Recommended split: all 10000 for training only
    lspet = PyLSPET(BTFM_BASE, LSPET_DIR, LSPET_CSV, UPI_S1H_DIR)

if USE_MPII:
    # MPII dataset has a test/train bool to check for split
    mpii = PyMPII(BTFM_BASE, MPII_RELEASE_PICKLE, MPII_IMG_DIR, UPI_S1H_MPII_IMG, UPI_S1H_MPII_ANNOT)
if USE_COCO:
    coco = PyCOCO(BTFM_BASE, COCO_TRAIN_IMG, COCO_TRAIN_ANNOT, COCO_VAL_IMG, COCO_VAL_ANNOT, COCO_TEST_IMG, COCO_TEST_INFO, BTFM_PP_COCO_SILHOUETTE)
if USE_3DPW:
    tdpw = Py3DPW(BTFM_BASE, TDPW_TRAIN_DIR, TDPW_VAL_DIR, TDPW_TEST_DIR, TDPW_IMG_DIR, BTFM_PP_3DPW_SILHOUETTE, BTFM_PP_3DPW_SILHOUETTE_VALID)
if USE_3DHP:
    mi3 = PyMI3(BTFM_BASE, MI3_DIR, MI3_TEST_DIR, MI3_PP_DIR)


# Start dataset. Top level is dict.
dataset = {}

# Gather data from each enabled dataset and add to top-level dict
if USE_LSP:
    lsp_data = lsp.gather_data(which_set, gid=gid)
    dataset['lsp']=lsp_data
if USE_LSPET:
    lspet_data = lspet.gather_data(which_set, gid=gid)
    dataset['lspet']=lspet_data
if USE_MPII:
    mpii_data = mpii.gather_data(which_set, gid=gid)
    dataset['mpii']=mpii_data
if USE_COCO:
    coco_data = coco.gather_data(which_set, gid=gid)
    dataset['coco']=coco_data
if USE_3DPW:
    tdpw_data = tdpw.gather_data(which_set, gid=gid)
    dataset['3dpw']=tdpw_data
if USE_3DHP:
    tick=time.time()
    mi3_data = mi3.gather_data(which_set, gid=gid)
    tock=time.time()
    print(f'Elapsed time: {tock-tick}')
    dataset['mpi-inf-3dhp']=mi3_data

# Write to file
with open('dataset.json', 'w') as outfile:
    json.dump(dataset, outfile, indent=2)
