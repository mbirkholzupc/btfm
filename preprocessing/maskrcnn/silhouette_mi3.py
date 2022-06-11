import os
import sys
import random
import math
import time
import pickle
import numpy as np
import skimage.io
from skimage.measure import find_contours
import matplotlib
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from PIL import Image
import h5py

from paths import *

# Path to Mask_RCNN
ROOT_DIR='../'

# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config (local file)
from cocodata import CocoConfig, CocoClassNames

# Directory to save logs and trained model
MODEL_DIR = './model'

# Local path to trained weights file from paths.py: MASK_RCNN_WEIGHTS

# Inference configuration, which overrides a few parameters in CocoConfig
# Batch size fixed to 1. It doesn't speed it up at all anyways to run more than 1 at a time on one GPU.
class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def save_silhouette(filename, mask):
    masked_image = np.zeros_like(mask, dtype=np.uint8)
    masked_image[:,:] = np.where((mask==1), 255, 0)

    im=Image.fromarray(masked_image)
    im.save(filename)

def valid_joints(jointsx, jointsy, dims):
    in_image=(jointsx>=0)&(jointsy>=0)&(jointsx<=dims[1])&(jointsy<=dims[0])
    nonzero=(jointsx!=0)|(jointsy!=0)
    return in_image & nonzero

def joint_bbox(jointsx, jointsy, dims):
    joint_mask=valid_joints(jointsx, jointsy, dims)
    if joint_mask.any() == True:
        c0=int(jointsx[joint_mask].min())
        c1=int(jointsx[joint_mask].max())
        r0=int(jointsy[joint_mask].min())
        r1=int(jointsy[joint_mask].max())
    else:
        c0=c1=r0=r1=0
    return (r0, c0, r1, c1)

def eval_joints_mask(mask, jointsx, jointsy):
    score=0
    joint_mask=valid_joints(jointsx, jointsy, mask.shape)
    valjx=jointsx[joint_mask]
    valjy=jointsy[joint_mask]
    if joint_mask.any()==True:
        score = np.mean([mask[int(jy),int(jx)] for jx, jy in zip(valjx, valjy)])
    return score

def bbox_iou(box1, box2):
    iou=0

    # Format of each box: (r0, c0, r1, c1)
    # row/colum_up/down/left/right
    r_u=max(box1[0], box2[0])
    c_l=max(box1[1], box2[1])
    r_d=min(box1[2], box2[2])
    c_r=min(box1[3], box2[3])

    l=max(r_d-r_u,0)
    w=max(c_r-c_l,0)
    intersection=l*w

    # This also protects us against divide by zero
    if intersection>0:
        union=((box1[2]-box1[0])*(box1[3]-box1[1])+(box2[2]-box2[0])*(box2[3]-box2[1]))-intersection
        iou=intersection/union

    return iou

def eval_bbox_mask(mask, roi, jointsx, jointsy):
    rmin=0
    rmax=mask.shape[0]
    cmin=0
    cmax=mask.shape[1]

    # ROI format: (r0, c0, r1, c1)
    r0=max(roi[0], rmin)
    r0=min(r0,     rmax)
    c0=max(roi[1], cmin)
    c0=min(c0, cmax)

    r1=max(roi[2], rmin)
    r1=min(r1,     rmax)
    c1=max(roi[3], cmin)
    c1=min(c1, cmax)

    score = np.sum(mask[r0:r1, c0:c1])/((r1-r0)*(c1-c0))

    jbb=joint_bbox(jointsx,jointsy,mask.shape)
    iou=bbox_iou((r0, c0, r1, c1), jbb)

    #print('DRAWING')
    #print(f'{r0} {r1} {c0} {c1}')
    #print(f'{jbb[0]} {jbb[2]} {jbb[1]} {jbb[3]}')
    #masked_image = np.zeros_like(mask, dtype=np.uint8)
    #masked_image[r0:r1,c0:c1] = 255
    #masked_image[jbb[0]:jbb[2],jbb[1]:jbb[3]] = 128
    #fig, ax = plt.subplots(1, figsize=(6,6))
    #ax.axis('off')
    #ax.imshow(masked_image.astype(np.uint8))
    #valj=valid_joints(jointsx, jointsy, mask.shape)
    #ax.plot(jointsx[valj],jointsy[valj],'ro')
    #plt.show()

    return iou


def choose_masks(seq_name, seq, frame, masks, rois, class_ids, scores, class_idx_person):
    # First, only consider masks tagged as "person"
    people_idxs=(class_ids==class_idx_person)
    num_actors=1

    if masks.shape[2] < num_actors:
        # If we didn't detect at least as many objects as actors, exit
        # Note: Could be improved to still return 1 if we can figure out which
        return None

    width=masks.shape[0]
    height=masks.shape[1]

    # Create arrays to hold scores
    jscore=np.zeros((num_actors, masks.shape[2]))
    bbscore=np.zeros((num_actors, masks.shape[2]))

    for a in range(num_actors):
        # Index to frame-1 because filenames are 1-based indexing
        joints=seq['annot2'][frame-1][0]
        jointsx=joints[:,0]
        jointsy=joints[:,1]
        for midx in range(masks.shape[2]):
            if not people_idxs[midx]:
                continue # Skip any that aren't human masks
            mask=masks[:,:,midx]
            roi=rois[midx,:]
            
            jscore[a,midx]=eval_joints_mask(mask, jointsx, jointsy)
            bbscore[a,midx]=eval_bbox_mask(mask,roi,jointsx,jointsy) 
            
            masked_image = np.zeros_like(mask, dtype=np.uint8)
            masked_image[:,:] = np.where((mask==1), 255, 0)
            #fig, ax = plt.subplots(1, figsize=(6,6))
            #ax.axis('off')
            #ax.imshow(masked_image.astype(np.uint8))
            #valj=valid_joints(jointsx, jointsy, mask.shape)
            #ax.plot(jointsx[valj],jointsy[valj],'ro')
            #plt.show()
    # Check the scores and pick best match for each actor
    print(f"seq: {seq_name} frame: {frame}")
    print(jscore)
    print(bbscore)
    print(people_idxs)
    best_jscores=np.argmax(jscore, axis=1)
    best_bbscores=np.argmax(bbscore, axis=1)
    if( not np.array_equal(best_jscores, best_bbscores) ):
        print(f"WARNING! J/BB scores not equal!' Seq: {seq_name} Frame: {frame}")
        print('J Scores: ' + str(best_jscores))
        print(jscore)
        print('BB Scores: ' + str(best_bbscores))
        print(bbscore)

    # Additionally, make sure best match is a person, not some other class
    for i in range(len(best_jscores)):
        if not people_idxs[best_jscores[i]]:
            print(f"J score not human! Seq: {seq_name} Frame: {frame}")
        if not people_idxs[best_jscores[i]]:
            print(f"BB score not human! Seq: {seq_name} Frame: {frame}")

    return best_bbscores

# Clean up everything before we start
os.system(f'rm -rf {BTFM_BASE}{MI3_SIL_DIR}/*')

config = InferenceConfig()
config.display()

# Create model in inference mode
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MASK_RCNN_WEIGHTS, by_name=True)

# Look up class index of person
class_idx_person=CocoClassNames.index('person')

# Get list of 3DPW dirs and get all images inside it
all_mi3_images=[]
base_img_dir=f'{BTFM_BASE}{MI3_TEST_DIR}'
dirs_mi3 = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

# Load metadata about each sequence/frame
seqs={}
for d in dirs_mi3:
    ann_path=f'{BTFM_BASE}{MI3_TEST_DIR}/{d}/annot_data.mat'
    seqs[d] = h5py.File(ann_path,'r')

# join dirs
for d in dirs_mi3:
    curdir=os.path.join(base_img_dir, d, 'imageSequence')
    img_list=sorted(next(os.walk(curdir))[2])
    for img in img_list:
        # Skip "non-valid" images
        frame=int(img[4:-4]) # 1-based because MATLAB
        # There are a few missing annotations at the end of TS3 and TS4
        if d=='TS3' and frame>5838:
            break
        if d=='TS4' and frame>6007:
            break
        if seqs[d]['valid_frame'][frame-1][0] == 1:
            all_mi3_images.append(os.path.join(MI3_TEST_DIR, d, 'imageSequence', img))

total_imgs=len(all_mi3_images)
print('Total images: ' + str(total_imgs))


# Example:
#seqs['TS1']['annot2'][frame-1][0]

#train_pkl=next(os.walk(f'{BTFM_BASE}{TDPW_TRAIN_DIR}'))[2]
#val_pkl=next(os.walk(f'{BTFM_BASE}{TDPW_VAL_DIR}'))[2]
#test_pkl=next(os.walk(f'{BTFM_BASE}{TDPW_TEST_DIR}'))[2]
#pkl_list = sorted(train_pkl+val_pkl+test_pkl)

#sequences={}
#for d, p_list in zip([f'{BTFM_BASE}{TDPW_TRAIN_DIR}', f'{BTFM_BASE}{TDPW_VAL_DIR}', f'{BTFM_BASE}{TDPW_TEST_DIR}'],
#                     [train_pkl, val_pkl, test_pkl]):
#    for p in p_list:
#        inpickle=open(os.path.join(d,p),'rb')
#        sequences[p[:-4]] = pickle.load(inpickle,encoding='latin1')
#        inpickle.close()

images_processed=0
octr=0
for d in dirs_mi3:
    ictr=0
    os.mkdir(f'{BTFM_BASE}{MI3_SIL_DIR}/{d}')

    curdir=os.path.join(base_img_dir, d, 'imageSequence')
    img_list=sorted(next(os.walk(curdir))[2])
    for img in img_list:
        frame=int(img[4:-4]) # 1-based because MATLAB
        # Skip "non-valid" images
        # There are a few missing annotations at the end of TS3 and TS4
        if d=='TS3' and frame>5838:
            break
        if d=='TS4' and frame>6007:
            break
        if seqs[d]['valid_frame'][frame-1][0] == 1:
            curimg=skimage.io.imread(os.path.join(curdir, img))
            results=model.detect([curimg], verbose=0)
            r=results[0]  # First (only) result in batch

            # Note: choose_masks() returns an array, so get the first (only) item in it
            best_mask = choose_masks(d, seqs[d], frame, r['masks'], r['rois'],
                r['class_ids'], r['scores'], class_idx_person)[0]

            if best_mask is not None:
                save_silhouette(f'{BTFM_BASE}{MI3_SIL_DIR}/{d}/{img[:-4]}.png',r['masks'][:,:,best_mask])

            images_processed+=1
            ictr+=1
            if 0 == (ictr%10):
                print(f'Images processed: {images_processed}/{total_imgs}')
    print(f'Images processed: {images_processed}/{total_imgs}')
    octr+=1

#    if octr==3:
#        #break
#        pass

#        # Use metadata to decide best silhouettes for each actor
#        seq_key=d
#        frame=int(img[-9:-4])
#        best_masks = choose_masks(sequences[seq_key], frame, r['masks'], r['rois'],
#            r['class_ids'], r['scores'], class_idx_person)
#
#        if best_masks is not None:
#            # Note: If choose_masks is updated to return only a single mask, need to update this too
#            for a, m in enumerate(best_masks):
#                save_silhouette(f'{BTFM_BASE}{BTFM_PP_3DPW_SILHOUETTE}/{d}/{img[:-4]}_subj{a}.png',r['masks'][:,:,m])
#            
#        #for m in range(r['masks'].shape[2]):
#        #    save_silhouette(f'{BTFM_BASE}{BTFM_PP_3DPW_SILHOUETTE}/{d}/{img[:-4]}_{m}.png',r['masks'][:,:,m])
#        images_processed+=1
#        ictr+=1
#        if 0 == (ictr%100):
#            print(f'Images processed: {images_processed}/{total_imgs}')
#
#    print(f'Images processed: {images_processed}/{total_imgs}')
#    octr+=1
#    if octr==3:
#        #break
#        pass
#
print('Done.')
