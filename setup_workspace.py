#!/bin/env python
import os

from paths import *

# Set up project directories and perform pre-processing here
os.system(f'mkdir -p {BTFM_BASE}')
os.system(f'mkdir -p {BTFM_BASE}{BTFM_PP}')
os.system(f'mkdir -p {BTFM_BASE}{BTFM_PP_LSP}')
os.system(f'mkdir -p {BTFM_BASE}{BTFM_PP_LSPET}')
os.system(f'mkdir -p {BTFM_BASE}{BTFM_PP_SILHOUETTE}')
os.system(f'mkdir -p {BTFM_BASE}{BTFM_PP_COCO}')
os.system(f'cp preprocessing/data/lsp/lsp.csv {BTFM_BASE}{BTFM_PP_LSP}/')
os.system(f'cp preprocessing/data/lsp/lspet.csv {BTFM_BASE}{BTFM_PP_LSPET}/')
os.system(f'cp -r preprocessing/data/mpii {BTFM_BASE}{BTFM_PP}/')

# Preprocess the MPII data into a pickle
os.system(f'python preprocessing/convertmpii.py -r {BTFM_BASE}{BTFM_PP_MPII}/RELEASE.txt > /dev/null')
os.system(f'mv mpii-RELEASE.pickle {BTFM_BASE}{BTFM_PP_MPII}/')

# Set up symlinks to all datasets
datasets=[LSP_DATASET_DIR, LSPET_DATASET_DIR, COCO_DATASET_DIR, MPII_DATASET_DIR, TDPW_DATASET_DIR, MI3_DATASET_DIR, MI3_PP_DATASET_DIR, UPI_S1H_DATASET_DIR]
links=[LSP_DIR,            LSPET_DIR,         COCO_DIR,         MPII_DIR,         TDPW_DIR,         MI3_DIR,         MI3_PP_DIR,         UPI_S1H_DIR]

for ds, link in zip(datasets, links):
    os.symlink(ds, BTFM_BASE+link)



