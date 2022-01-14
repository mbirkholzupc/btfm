# BTFM Repository
Repository to hold scripts for dataset preparation for TFM project, which focuses on recovery of human shape parameters from 2D RGB images.

The main datasets are:
 * MPII
 * LSP
 * LSPET
 * COCO
 * 3DPW
 * MPI-INF-3DHP

Preprocessing:
 * Refer to instructions in preprocessing/Readme.md for details
 * No scripts need to be run manually. Any preprocessing will be called by setup\_workspace.py
 * Download weights from https://github.com/matterport/Mask\_RCNN/releases (Specific link at this time: https://github.com/matterport/Mask\_RCNN/releases/download/v2.0/mask\_rcnn\_coco.h5) and update the MASK\_RCNN\_WEIGHTS variable in paths.py

Preparing the workspace:
 * Run setup\_workspace.sh
