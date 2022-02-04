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
from utils import plotMultiOnImage

#PLOT_IMAGES=True
PLOT_IMAGES=False

# Parameters for "good" joints
MIN_JOINTS=10
MIN_JSCORE=0.75

# TODO: Start with this file
# First step: visualize images, joints, masks (maybe add to pymi3 or utils?)
# Then: figure out how to decide if masks/joints are good fit or not
#       - Is subject/subjects present in frame?
#       - If there are two masks, are they different? Completely/marginally different?

def valid_joints(jointsx, jointsy, dims):
    # Dims: (width, height)
    in_image=(jointsx>=0)&(jointsy>=0)&(jointsx<=dims[0])&(jointsy<=dims[1])
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

def joint_bbox_plot(jointsx, jointsy, dims):
    # Dims: (width, height)
    joint_mask=valid_joints(jointsx, jointsy, dims)
    if joint_mask.any() == True:
        x0=int(jointsx[joint_mask].min())
        x1=int(jointsx[joint_mask].max())
        y0=int(jointsy[joint_mask].min())
        y1=int(jointsy[joint_mask].max())
    else:
        x0=y0=x1=y1=0

    # Plot wants lower-left coord and width and height
    return ((x0, y0), x1-x0, y1-y0)


def eval_joints_mask(mask, jointsx, jointsy):
    score=0
    joint_mask=valid_joints(jointsx, jointsy, (mask.shape[1], mask.shape[0]))
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

    jbb=joint_bbox(jointsx,jointsy,(mask.shape[1],mask.shape[0]))
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

# Check if joints exist for each actor
def evaluate_joints(seq, frame, num_actors, dims):
    # Dims: (width, height)
    joints_exist = [False, False]

    for a in range(num_actors):
        joints=seq['poses2d'][a][frame]
        jointsx=joints[0]
        jointsy=joints[1]
        jointsv=joints[2]

        valj=valid_joints(jointsx, jointsy, dims)
        if np.sum(valj) >= MIN_JOINTS:
            joints_exist[a]=True

    return joints_exist

# Check masks against metadata. Return which ones are ok/not ok.
def evaluate_masks(seq, frame, num_actors, masks):
    masks_ok=[]
    # Create arrays to hold scores
    jscore=np.zeros((num_actors))
    bbscore=np.zeros((num_actors))

    if len(masks)==0:
        # Nothing to evaluate
        return masks_ok, jscore


    for a in range(num_actors):
        joints=seq['poses2d'][a][frame]
        jointsx=joints[0]
        jointsy=joints[1]
        jointsv=joints[2]

        # masks is list of tuples. Index 1 is bitmap
        mask=masks[a][1]
        #roi=rois[midx,:]
        
        jscore[a]=eval_joints_mask(mask, jointsx, jointsy)
        #bbscore[a]=eval_bbox_mask(mask,roi,jointsx,jointsy) 
    # Check the scores and pick best match for each actor
    print(f"seq: {seq['sequence']} frame: {frame}")
    print(jscore)
    #print(bbscore)
    #best_jscores=np.argmax(jscore, axis=1)
    #best_bbscores=np.argmax(bbscore, axis=1)
    #if( not np.array_equal(best_jscores, best_bbscores) ):
    #    print(f"WARNING! J/BB scores not equal!' Seq: {seq['sequence']} Frame: {frame}")
    #    print('J Scores: ' + str(best_jscores))
    #    print(jscore)
    #    print('BB Scores: ' + str(best_bbscores))
    #    print(bbscore)

    # TODO: Analyze scores and decide which ones are/aren't valid
    masks_ok=[False]*num_actors

    return masks_ok, jscore

# Check masks against metadata. Return which ones are ok/not ok.
def check_masks(seq, frame, joints_ok, masks_exist, masks):
    masks_ok=[]

    # Figure out which combinations of joints/mask are valid
    # j0m0   j0m1
    # j1m0   j1m1
    jmarr=np.zeros((2,2),dtype=bool)
    jmarr[0,0]=joints_ok[0] & masks_exist[0]
    jmarr[1,0]=joints_ok[1] & masks_exist[0]
    jmarr[0,1]=joints_ok[0] & masks_exist[1]
    jmarr[1,1]=joints_ok[1] & masks_exist[1]

    # Create array to hold scores
    jscore=np.zeros_like(jmarr)

    # The j, m indices are associated with actor 0/1 in each scene
    for j in range(2):
        for m in range(2):
            if jmarr[j,m]:
                joints=seq['poses2d'][j][frame]
                jointsx=joints[0]
                jointsy=joints[1]

                # masks is list of tuples. Index 1 is bitmap
                mask=masks[m][1]

                jscore[j,m]=eval_joints_mask(mask, jointsx, jointsy)

    # Check which values are above minimum "good" threshold
    above_threshold=(jscore>=MIN_JSCORE)

    # Now, make sure we don't have a silhouette that encompasses all of the joints
    # like when bodies blend together
    if above_threshold[0,0] and above_threshold[1,0]:
        above_threshold[0,0]=False
        above_threshold[1,0]=False
    if above_threshold[0,1] and above_threshold[1,1]:
        above_threshold[0,1]=False
        above_threshold[1,1]=False
        
    # Create output array
    results=np.array([above_threshold[0,0], above_threshold[1,1]])

    return results

def check_silhouette_exists(image_filename, silhouette_dir, num_actors):
    silhouettes=[]
    silhouettes_exist=[False,False]
    # Filename format: image_12345.jpg
    frame=int(image_filename[-9:-4])

    for i in range(num_actors):
        possible=f'image_{frame:05d}_subj{i}.png'
        try:
            fullsilhouette=f'{silhouette_dir}/{possible}'
            img=Image.open(fullsilhouette)
            bitmap=np.array(img, dtype=float)/255.0
            img.close()

            # If we got here without an exception, the file exists
            silhouettes.append((fullsilhouette,bitmap))
            silhouettes_exist[i]=True
        except(FileNotFoundError):
            # Add None to the list as a placeholder
            silhouettes.append(None)

    if len(silhouettes) < 2:
        silhouettes.append(None)

    return silhouettes, silhouettes_exist

def draw_silhouette(img, bitmap, alpha=0.5, color=(1,0,0)):
    # Both img, bitmap are numpy arrays

    # Convert bitmap to 3-channel
    mask=np.zeros_like(img)
    #bitmapf=np.array(bitmap,dtype=float)/255.0
    mask[:,:,0]=bitmap
    mask[:,:,1]=bitmap
    mask[:,:,2]=bitmap
    mask=mask*alpha

    # Convert color to array of img size
    rchan=np.full(bitmap.shape, color[0], dtype=float)
    gchan=np.full(bitmap.shape, color[1], dtype=float)
    bchan=np.full(bitmap.shape, color[2], dtype=float)
    solidcolor=np.dstack((rchan, gchan, bchan))

    blended=img*(1-mask) + solidcolor*(mask)

    return blended

def show_annotations(imagepath, silhouettes, sequence, frame, num_actors):
    rgb_img=Image.open(imagepath)
    width, height = rgb_img.size

    joints_to_plot=[]
    joint_bboxes=[]
    for actor in range(num_actors):
        j2d=sequence['poses2d'][actor][frame]
        j2x=j2d[0]
        j2y=j2d[1]
        #j2v=j2d[2]
        valj=valid_joints(j2x, j2y, (width, height))
        jointsxy=np.transpose(np.array([j2y[valj], j2x[valj]]))
        joints_to_plot.append(jointsxy)
        jbb=joint_bbox_plot(j2x, j2y, (width, height))
        joint_bboxes.append(jbb)
        #print('joints: ' + str(actor))
        #print(j2x)
        #print(j2y)
        #print(valj)

    # Draw silhouettes (lightly) on image)
    image_w_sil=np.array(rgb_img, dtype=float)/255.0
    sil_colors=[(1,0,0), (0,0,1)]
    sil_color_idx=0
    for sil in silhouettes:
        if sil is None:
            sil_color_idx+=1
            continue

        color=sil_colors[sil_color_idx]
        bitmap=sil[1]
        image_w_sil=draw_silhouette(image_w_sil, bitmap, alpha=0.3, color=color)
        sil_color_idx+=1

    # Create bbox rectangles
    bb_rects=[]
    bb_colors=['r','b']
    bb_coloridx=0
    for bb in joint_bboxes:
        color=bb_colors[bb_coloridx]
        bb_rects.append(patches.Rectangle(bb[0], bb[1], bb[2], ec=color, fill=False))
        bb_coloridx+=1

    # Draw joints
    image_w_plots=image_w_sil
    if num_actors==1:
        image_w_plots=plotMultiOnImage(image_w_sil, zip(joints_to_plot, ['ro']), bb_rects)
    elif num_actors==2:
        image_w_plots=plotMultiOnImage(image_w_sil, zip(joints_to_plot, ['ro', 'bo']), bb_rects)


    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.axis('off')
    ax.imshow(image_w_plots)
    plt.show()

    return


# Get list of 3DPW dirs and get all images inside it
# Also create list of all of the silhouettes associated with each image
all_3dpw_imgs=[]
base_img_dir=BTFM_BASE+TDPW_IMG_DIR
base_silhouette_dir=BTFM_BASE+BTFM_PP_3DPW_SILHOUETTE

dirs_3dpw = sorted(next(os.walk(base_img_dir))[1])
# join dirs
for d in dirs_3dpw:
    curdir=os.path.join(base_img_dir, d)
    if os.path.split(curdir)[1]=='downtown_cafe_01':
        # Unfortunately, this pkl is missing from the data
        continue
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

# Create a dictionary to signal which annotations are good.
# Key: sequence/image.jpg
# Value: Tuple (actor0 joints ok, actor1 joints ok, actor0 silhouette ok, actor1 silhouette ok)
good_3dpw_annotations={}

images_processed=0
octr=0
for d in dirs_3dpw:
    skiplist=['courtyard_arguing_00', 'courtyard_backpack_00', 'courtyard_basketball_00',
              'courtyard_basketball_01', 'courtyard_bodyScannerMotions_00', 'courtyard_box_00',
              'courtyard_capoeira_00', 'courtyard_captureSelfies_00', 'courtyard_dancing_00',
              'courtyard_dancing_01', 'courtyard_drinking_00', 'courtyard_giveDirections_00',
              'courtyard_golf_00', 'courtyard_goodNews_00', 'courtyard_giveDirections_00',
              'courtyard_hug_00', 'courtyard_jacket_00', 'courtyard_jumpBench_01',
              'courtyard_laceShoe_00', 'courtyard_jacket_00', 'courtyard_jumpBench_01',
              'courtyard_rangeOfMotions_00', 'courtyard_rangeOfMotions_01', 'courtyard_relaxOnBench_00',
              'courtyard_relaxOnBench_01', 'courtyard_shakeHands_00', 'courtyard_warmWelcome_00',
              'downtown_arguing_00', 'downtown_bar_00', 'downtown_bus_00', 'downtown_cafe_00',
              'downtown_car_00', ]
    #if d in skiplist:
    #    continue
    #if d != 'courtyard_captureSelfies_00':
    #if d != 'downtown_bar_00':
    #    continue
    if d=='downtown_cafe_01':
        # Unfortunately, this pkl is missing from the data
        continue
    print(d)
    ictr=0

    curdir=os.path.join(base_img_dir, d)
    cursildir=os.path.join(base_silhouette_dir, d)
    img_list=sorted(next(os.walk(curdir))[2])
    for img in img_list:
        imagepath=os.path.join(curdir,img)
        imgfile=Image.open(imagepath)
        width, height = imgfile.size
        imgfile.close()

        output_key=f'{d}/{img}'
        output_value=[False, False, False, False]
        seq_key=d
        num_actors=len(sequences[seq_key]['poses2d'])
        frame=int(img[-9:-4])
        #if frame<700:
        #    continue

        # Check how many silhouettes exist for given image
        silhouettes, silhouettes_exist = check_silhouette_exists(imagepath, cursildir,num_actors)
        assert(len(silhouettes)==2)

        # Check frame metadata to see if there are joint annotations for either actor in frame
        joints_ok = evaluate_joints(sequences[seq_key], frame, num_actors, (width, height))

        # Use metadata to decide best silhouettes for each actor
        #masks_ok, mask_jscores = evaluate_masks(sequences[seq_key], frame, num_actors, silhouettes)
        
        # The real function to check masks
        masks_ok = check_masks(sequences[seq_key], frame, joints_ok, silhouettes_exist, silhouettes)

        print(f'({frame}) Joints: {joints_ok} Masks: {masks_ok}')

        # Check silhouette quality
        if PLOT_IMAGES:
            show_images=False
            #if (95<ictr<100):
            #if (200<ictr<250):
            if 0==(ictr%2):
                show_images=True
            #for s in mask_jscores:
            #    if 0<s<0.75:
            #        show_images=True
            #        break
            if show_images:
                show_annotations(imagepath, silhouettes, sequences[d], frame, num_actors)

        # Save output in dictionary
        output_value[0:2]=joints_ok
        output_value[2:4]=masks_ok
        good_3dpw_annotations[output_key]=output_value
        images_processed+=1
        ictr+=1
        #if 0 == (ictr%100):
        #    print(f'Images processed: {images_processed}/{total_imgs}')

    print(f'Images processed: {images_processed}/{total_imgs}')
    octr+=1
    if octr==3:
        #break
        pass

outfile=open('good_3dpw_annotations.pkl', 'wb')
pickle.dump(good_3dpw_annotations, outfile)
outfile.close()

print('Done.')
