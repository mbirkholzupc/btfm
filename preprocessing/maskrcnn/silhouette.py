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
    c0=int(jointsx[joint_mask].min())
    c1=int(jointsx[joint_mask].max())
    r0=int(jointsy[joint_mask].min())
    r1=int(jointsy[joint_mask].max())
    print(f'joint_bbox: {r0} {c0} {r1} {c1}')
    return (r0, c0, r1, c1)

def eval_joints_mask(mask, jointsx, jointsy):
    joint_mask=valid_joints(jointsx, jointsy, mask.shape)
    valjx=jointsx[joint_mask]
    valjy=jointsy[joint_mask]
    return np.mean([mask[int(jy),int(jx)] for jx, jy in zip(valjx, valjy)])

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
        print(f'I: {intersection}')
        print(f'U: {union}')
        print(f'IoU: {iou}')
    else:
        print(f'IoU: 0')

    return iou

def eval_bbox_mask(mask, roi, jointsx, jointsy):
    print('ROI: ' + str(roi))
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

    print(f'{rmin} {rmax} {cmin} {cmax}')
    print(f'{roi[0]} {roi[1]} {roi[2]} {roi[3]}')
    print(f'{r0} {c0} {r1} {c1}')

    print('mask: ' + str(mask.shape))
    print('roi: ' + str(mask[r0:r1, c0:c1].shape))
    score = np.sum(mask[r0:r1, c0:c1])/((r1-r0)*(c1-c0))
    print('score: ' + str(score))

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


def choose_masks(seq, frame, masks, rois, class_ids, scores, class_idx_person):
    print('choose_masks')
    print('Sequence/Frame: ' + seq['sequence'] + '/' + str(frame))
    print('Total masks: ' + str(masks.shape[2]))
    # First, only consider masks tagged as "person"
    people_idxs=(class_ids==class_idx_person)
    print('people_idxs: ' + str(people_idxs))
    frame_idx=seq['img_frame_ids'][frame]
    print('frame_idx: ' + str(frame_idx))
    num_actors=len(seq['poses2d'])
    print('Num actors: ' + str(num_actors))
    print('ROIs: ' + str(rois.shape))

    # Create arrays to hold scores
    jscore=np.zeros((num_actors, masks.shape[2]))
    bbscore=np.zeros((num_actors, masks.shape[2]))

    for a in range(num_actors):
        joints=seq['poses2d'][a][frame_idx]
        jointsx=joints[0]
        jointsy=joints[1]
        jointsv=joints[2]
        for midx in range(masks.shape[2]):
            mask=masks[:,:,midx]
            roi=rois[midx,:]
            
            print(mask.shape)
            print(jointsx)
            print(jointsy)
            jscore[a,midx]=eval_joints_mask(mask, jointsx, jointsy)
            print('jpoints: ' + str(jscore[a,midx]))
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
    print('JSCORE')
    print(jscore)
    print('BBSCORE')
    print(bbscore)
    best_jscores=np.argmax(jscore, axis=1)
    best_bbscores=np.argmax(bbscore, axis=1)
    if( not np.array_equal(best_jscores, best_bbscores) ):
        print('WARNING! Not equal!')
        print('J Scores: ' + str(best_jscores))
        print('BB Scores: ' + str(best_bbscores))

    # Additionally, make sure best match is a person, not some other class
    for i in range(len(best_jscores)):
        if not people_idxs[best_jscores[i]]:
            print('J Score not human!')
        if not people_idxs[best_jscores[i]]:
            print('BB Score not human!')

    return best_bbscores

# Clean up everything before we start
os.system(f'rm -rf {BTFM_BASE}{BTFM_PP_SILHOUETTE}/*')

config = InferenceConfig()
config.display()

# Create model in inference mode
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MASK_RCNN_WEIGHTS, by_name=True)

# Look up class index of person
class_idx_person=CocoClassNames.index('person')

# Get list of 3DPW dirs and get all images inside it
all_3dpw_imgs=[]
base_img_dir=BTFM_BASE+TDPW_IMG_DIR
dirs_3dpw = sorted(next(os.walk(base_img_dir))[1])
# join dirs
for d in dirs_3dpw:
    curdir=os.path.join(base_img_dir, d)
    img_list=sorted(next(os.walk(curdir))[2])
    for img in img_list:
        all_3dpw_imgs.append(os.path.join(TDPW_IMG_DIR, d, img))

total_imgs=len(all_3dpw_imgs)
print('Total images: ' + str(total_imgs))

# Load metadata about each sequence/frame
train_pkl=next(os.walk(f'{BTFM_BASE}{TDPW_TRAIN_DIR}'))[2]
val_pkl=next(os.walk(f'{BTFM_BASE}{TDPW_VAL_DIR}'))[2]
test_pkl=next(os.walk(f'{BTFM_BASE}{TDPW_TEST_DIR}'))[2]
pkl_list = sorted(train_pkl+val_pkl+test_pkl)

sequences={}
for d, p_list in zip([f'{BTFM_BASE}{TDPW_TRAIN_DIR}', f'{BTFM_BASE}{TDPW_VAL_DIR}', f'{BTFM_BASE}{TDPW_TEST_DIR}'],
                     [train_pkl, val_pkl, test_pkl]):
    for p in p_list:
        inpickle=open(os.path.join(d,p),'rb')
        sequences[p[:-4]] = pickle.load(inpickle,encoding='latin1')
        inpickle.close()

# TODO: Filter out only the (up to 2) silhouettes we care about.
#       Based on class==human and ??? (maybe bbox/sil and joint overlap?)
octr=0
for d in dirs_3dpw:
    if d=='downtown_cafe_01':
        # Unfortunately, this pkl is missing from the data
        continue
    ictr=0
    dir_results = {}
    os.mkdir(f'{BTFM_BASE}{BTFM_PP_SILHOUETTE}/{d}')
    curdir=os.path.join(base_img_dir, d)
    img_list=sorted(next(os.walk(curdir))[2])
    for img in img_list:
        curimg=skimage.io.imread(os.path.join(curdir, img))
        results=model.detect([curimg], verbose=0)
        r=results[0]
        print('Num masks: ' + str(len(r['masks'])))
        print('Shape masks: ' + str(r['masks'].shape))

        # Use metadata to decide best silhouettes for each actor
        seq_key=d
        frame=int(img[-9:-4])
        best_masks = choose_masks(sequences[seq_key], frame, r['masks'], r['rois'],
            r['class_ids'], r['scores'], class_idx_person)

#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                            class_names, r['scores'])

        for a, m in enumerate(best_masks):
            save_silhouette(f'{BTFM_BASE}{BTFM_PP_SILHOUETTE}/{d}/{img[:-4]}_subj{a}.png',r['masks'][:,:,m])
            
        #for m in range(r['masks'].shape[2]):
        #    save_silhouette(f'{BTFM_BASE}{BTFM_PP_SILHOUETTE}/{d}/{img[:-4]}_{m}.png',r['masks'][:,:,m])
        dir_results[os.path.join(TDPW_IMG_DIR, d, img)]=results
        ictr+=1
        if ictr == 10:
            break


    outfile=open(BTFM_BASE+BTFM_PP_SILHOUETTE+'/'+d+'.pkl', 'wb')
    pickle.dump(dir_results, outfile)
    outfile.close()

    octr+=1
    if octr==3:
        #break
        pass

exit()

for i in [0]:
    batch_start=i*batch_size
    batch = [skimage.io.imread(x) for x in all_3dpw_imgs[batch_start:batch_start+batch_size]]
    start_time=time.time()
    results = model.detect(batch, verbose=0)
    elapsed=time.time()-start_time
    pct = 100*(i+1)*batch_size/len(all_3dpw_imgs)
    print('Progress: ' + str(pct) + '%')
    print('Time per image: ' + str(elapsed/batch_size))

    # TODO: Put this code in handle_result

    batch_filenames=[x for x in all_3dpw_imgs[batch_start:batch_start+batch_size]]
    for fn,r,image in zip(batch_filenames,results,batch):
        print(fn)
        num_people=np.sum(np.array(r['class_ids'])==class_idx_person)
        print('People: ' + str(num_people))
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
        
#if(0==(i%10)):
#    pct = 100*i/len(all_3dpw_imgs)
#    print('Progress: ' + str(pct) + '%')



print('Done.')

#for i, img_fn in enumerate(all_3dpw_imgs):
#    image = skimage.io.imread(img_fn)
#    results = model.detect([image], verbose=0)
#    if(0==(i%100)):
#        pct = 100*i/len(all_3dpw_imgs)
#        print('Progress: ' + str(pct) + '%')

#print(next(os.walk(IMAGE_DIR)))

## Load a random image from the images folder
#print(next(os.walk(IMAGE_DIR)))
#file_names = next(os.walk(IMAGE_DIR))[2]
#print("PRINTING STUFF!")
#print(IMAGE_DIR)
#print(file_names)
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#
## Run detection
#results = model.detect([image], verbose=1)
#
## Visualize results
#r = results[0]
#print(r)
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                            class_names, r['scores'])

