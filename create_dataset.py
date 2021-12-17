#!/usr/bin/env python

# Main script to create dataset
# Combines the following datasets in JSON format:
#   - MPII
#   - LSP
#   - LSPET
#   - COCO2017
#   - 3DPW
#   - MPI-INF-3DHP


import json

from paths import *

from pylsp import PyLSP
from pylspet import PyLSPET
from pympii import PyMPII
from pycoco import PyCOCO
from py3dpw import Py3DPW
#from pymi3 import PyMI3

from utils import GID

# Create a global ID object
gid = GID()

# Create the data loader objects. If memory issues, may need to do one at a time.
# LSP Recommended split: first 1000 training, last 1000 testing
lsp = PyLSP(LSP_DIR, LSP_CSV)
# LSPET Recommended split: all 10000 for training only
lspet = PyLSPET(LSPET_DIR, LSPET_CSV)
# MPII dataset has a test/train bool to check for split
mpii = PyMPII(MPII_RELEASE_PICKLE, MPII_IMG_DIR)
coco = PyCOCO(COCO_TRAIN_IMG, COCO_TRAIN_ANNOT, COCO_VAL_IMG, COCO_VAL_ANNOT, COCO_TEST_IMG, COCO_TEST_INFO)
#tdpw = Py3DPW(TDPW_TRAIN_DIR, TDPW_VAL_DIR, TDPW_TEST_DIR, TDPW_IMG_DIR)
#exit()


# Start dataset. Top level is dict.
dataset = {}

# Next level is the processed, formatted data from each dataset
#mpii_data = mpii.gather_data('toy', gid=gid)  # TODO: Remove
#coco_data = coco.gather_data('toy', gid=gid)

which_set='train'  # 'train', 'test', 'val', 'toy'
lsp_data = lsp.gather_data(which_set, gid=gid)
lspet_data = lspet.gather_data(which_set, gid=gid)
mpii_data = mpii.gather_data(which_set, gid=gid)
coco_data = coco.gather_data(which_set, gid=gid)
#tdpw_data = tdpw.gather_data('toy', gid=gid)

tdpw_data = [ "no data" ]
mi3_data = [ "no data either" ]

# Add data from each dataset to top-level dict
dataset['lsp']=lsp_data
dataset['lspet']=lspet_data
dataset['mpii']=mpii_data
dataset['coco']=coco_data
dataset['3dpw']=tdpw_data
dataset['mpi-inf-3dhp']=mi3_data

# Write to file
with open('dataset.json', 'w') as outfile:
    json.dump(dataset, outfile, indent=2)
