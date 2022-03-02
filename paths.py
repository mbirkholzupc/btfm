
# Should only need to modify these paths in this file:
#   - BTFM_BASE
#   - LSP_DATASET_DIR
#   - LSPET_DATASET_DIR
#   - COCO_DATASET_DIR
#   - MPII_DATASET_DIR
#   - TDPW_DATASET_DIR
#   - MI3_DATASET_DIR
#   - MI3_PP_DATASET_DIR
#   - UPI_S1H_DATASET_DIR

# Note that this relies on a filesystem that supports symlinks


# Set up output path
BTFM_BASE='/media/data/btfm'

# Set up other paths in output for preprocessed data
BTFM_PP='/pp'
BTFM_PP_LSP=BTFM_PP+'/lsp'
BTFM_PP_LSPET=BTFM_PP+'/lspet'
BTFM_PP_MPII=BTFM_PP+'/mpii'
BTFM_PP_3DPW=BTFM_PP+'/3dpw'
BTFM_PP_3DPW_SILHOUETTE=BTFM_PP_3DPW+'/silhouette'
BTFM_PP_3DPW_SILHOUETTE_VALID=BTFM_PP_3DPW+'/good_3dpw_annotations.pkl'
BTFM_PP_COCO=BTFM_PP+'/coco'
BTFM_PP_COCO_SILHOUETTE=BTFM_PP_COCO+'/silhouette'

# Pre-processed files
LSP_CSV=BTFM_PP_LSP+'/lsp.csv'
LSPET_CSV=BTFM_PP_LSPET+'/lspet.csv'
MPII_RELEASE_PICKLE=BTFM_PP_MPII+'/mpii-RELEASE.pickle'

# Set up paths to datasets (absolute)
LSP_DATASET_DIR='/media/data/lsp'
LSPET_DATASET_DIR='/media/data/lspet'
COCO_DATASET_DIR='/media/data/coco2017'
MPII_DATASET_DIR='/media/data/mpii'
TDPW_DATASET_DIR='/media/data/3dpw'
MI3_DATASET_DIR='/media/data/mpi_inf_3dhp/mpi_inf_3dhp/download'
MI3_PP_DATASET_DIR='/media/data/mpi_inf_3dhp_pp'
UPI_S1H_DATASET_DIR='/media/data/up/upi-s1h'
SSP3D_DATASET_DIR='/media/data/SSP-3D'

# Other weights
MASK_RCNN_WEIGHTS='/media/data/maskrcnn_weights/mask_rcnn_coco.h5'

# Required preprocessing
# LSP: convertlsp.m to convert joints.mat to lsp.csv
# LSPET: convertlspet.m to convert joints.mat to lspet.csv
# MPII: 1) convertmpii.m to convert mpii_human_pose_v1_u12_1.mat to set of csv files
#       2) Convert output files (RELEASE.txt and *.csv) to pickle with convertmpii.py

# The following paths are all relative, starting from BTFM_BASE
LSP_DIR='/lsp'
LSP_IMG_DIR=LSP_DIR + '/images'

LSPET_DIR='/lspet'
LSPET_IMG_DIR=LSPET_DIR+'/images'

COCO_DIR='/coco2017'
COCO_TRAIN_IMG=COCO_DIR+'/train2017'
COCO_VAL_IMG=COCO_DIR+'/val2017'
COCO_TEST_IMG=COCO_DIR+'/test2017'
COCO_TRAIN_ANNOT=COCO_DIR+'/annotations/person_keypoints_train2017.json'
COCO_VAL_ANNOT=COCO_DIR+'/annotations/person_keypoints_val2017.json'
COCO_TEST_INFO=COCO_DIR+'/annotations/image_info_test2017.json'
# dev is a smaller set 
#COCO_TEST_INFO=COCO_DIR+'/annotations/image_info_test-dev2017.json'

MPII_DIR='/mpii'
MPII_IMG_DIR=MPII_DIR+'/images'

TDPW_DIR='/3dpw'
TDPW_IMG_DIR=TDPW_DIR+'/imageFiles'
TDPW_SEQ_DIR=TDPW_DIR+'/sequenceFiles'
TDPW_TRAIN_DIR=TDPW_SEQ_DIR+'/train'
TDPW_VAL_DIR=TDPW_SEQ_DIR+'/validation'
TDPW_TEST_DIR=TDPW_SEQ_DIR+'/test'

MI3_DIR='/mpi_inf_3dhp'
MI3_TEST_DIR=MI3_DIR+'/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set'
MI3_PP_DIR='/mi3_pp'


UPI_S1H_DIR='/upi-s1h'
UPI_S1H_MPII=UPI_S1H_DIR+'/data/mpii'
UPI_S1H_MPII_IMG=UPI_S1H_MPII+'/images'
UPI_S1H_MPII_ANNOT=UPI_S1H_MPII+'/correspondences.csv'

SSP3D_DIR='/ssp-3d'
