
# Set up paths to all datasets
# First line in each group is an absolute path. Following lines are relative to it.

# Required preprocessing
# LSP: convertlsp.m to convert joints.mat to lsp.csv
# LSPET: convertlspet.m to convert joints.mat to lspet.csv
# MPII: 1) convertmpii.m to convert mpii_human_pose_v1_u12_1.mat to set of csv files
#       2) Convert output files (RELEASE.txt and *.csv) to pickle with convertmpii.py

LSP_DIR='/media/mike/T7/data/lsp'
LSP_IMG_DIR=LSP_DIR + '/images'
LSP_CSV=LSP_DIR+'/lsp.csv'

LSPET_DIR='/media/mike/T7/data/lspet'
LSPET_IMG_DIR=LSPET_DIR+'/images'
LSPET_CSV=LSPET_DIR+'/lspet.csv'

COCO_DIR='/media/mike/T7/data/coco2017'
COCO_TRAIN_IMG=COCO_DIR+'/train2017'
COCO_VAL_IMG=COCO_DIR+'/val2017'
COCO_TEST_IMG=COCO_DIR+'/test2017'
COCO_TRAIN_ANNOT=COCO_DIR+'/annotations/person_keypoints_train2017.json'
COCO_VAL_ANNOT=COCO_DIR+'/annotations/person_keypoints_val2017.json'
COCO_TEST_INFO=COCO_DIR+'/annotations/image_info_test2017.json'
# dev is a smaller set 
#COCO_TEST_INFO=COCO_DIR+'/annotations/image_info_test-dev2017.json'

MPII_DIR='/media/mike/T7/data/mpii'
MPII_IMG_DIR=MPII_DIR+'/images'
MPII_RELEASE_PICKLE=MPII_DIR+'/mpii-RELEASE.pickle'

TDPW_DIR='/media/mike/T7/data/3dpw'
TDPW_IMG_DIR=TDPW_DIR+'/imageFiles'
TDPW_SEQ_DIR=TDPW_DIR+'/sequenceFiles'
TDPW_TRAIN_DIR=TDPW_SEQ_DIR+'/train'
TDPW_VAL_DIR=TDPW_SEQ_DIR+'/validation'
TDPW_TEST_DIR=TDPW_SEQ_DIR+'/test'

MI3_DIR='/media/mike/T7/data/mpi_inf_3dhp/mpi_inf_3dhp/download'
MI3_TEST_DIR=MI3_DIR+'/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set'
MI3_PP_DIR='/media/mike/T7/data/mpi_inf_3dhp/pp'
