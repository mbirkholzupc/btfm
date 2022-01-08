import os

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.io import loadmat

from paths import *
from pymi3_utils import MpiiSeqInfo, mpii_get_sequence_info

# Threshold to consider poses "different" in mm
# Current: 40 mm
JOINT_DIFF_THRESHOLD=40

SUBJS = [1, 2, 3, 4, 5, 6, 7, 8]
SEQS = [1, 2]
CAMERAS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# SMPL format
pymi3_joints = [ 'spine3', 'spine4', 'spine2', 'spine', 'pelvis', # 5
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', # 11
       'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', # 17
       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', # 23
       'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe' ]

def _filter_same_pose(annot, camera, num_frames):
    """
    Checks a list of poses and returns indices of those that have at least one joint that
    moves at least 40 mm from its previous position
    """
    # Get reference to list of frames
    frames = annot['annot3'][camera][0]
    assert(num_frames <= frames.shape[0])  # There are some annotations for frames w/o images, so might not match

    # Always add first frame
    unique_pose_idxs = [0]

    # Build list of all 3D joints
    j=[]
    for i in range(num_frames):
        jx=frames[i][0::3]
        jy=frames[i][1::3]
        jz=frames[i][2::3]
        j.append(np.transpose([jx, jy, jz]))

    # Prime loop by loading the first joint, then add images only when they differ sufficiently
    # from last included image
    last_i=0
    last_j=j[last_i]
    for i in range(1,num_frames):
        cur_j = j[i]
        d=np.array([euclidean(a, b) for a, b in zip(last_j, cur_j)]).max()
        if d > JOINT_DIFF_THRESHOLD:
            unique_pose_idxs.append(i)
            last_idx=i
            last_j=j[last_idx]

    return unique_pose_idxs

for subj in SUBJS:
    for seq in SEQS:
        annot_path=f'{MI3_DATASET_DIR}/S{subj}/Seq{seq}/annot.mat'
        print(annot_path)
        annot = loadmat(annot_path)
        seq_info = mpii_get_sequence_info(subj, seq)
        print(seq_info)

        # Create list of non-identical poses. We will just use the first camera's annotations.
        # There are small discrepancies between cameras, but nothing major.
        cam=0
        keep_idxs=_filter_same_pose(annot, cam, seq_info.num_frames)
        # Print number before/after
        keep = len(keep_idxs)
        pct = 100*keep/seq_info.num_frames
        print(f'Keeping: {keep}/{seq_info.num_frames} ({pct:0.2f}%)')

        for cam in CAMERAS:
            print(f'Camera {cam}')
            # Extract jpgs and only keep the ones we care about
            base_dir=f'{MI3_PP_DATASET_DIR}/S{subj}/Seq{seq}'
            os.system(f'mkdir -p {base_dir}/img')
            os.system(f'mkdir -p {base_dir}/chair')
            os.system(f'mkdir -p {base_dir}/fg')
            os.system(f'ffmpeg -i {MI3_DATASET_DIR}/S{subj}/Seq{seq}/imageSequence/video_{cam}.avi -qscale:v 1 -start_number 0 "{MI3_PP_DATASET_DIR}/S{subj}/Seq{seq}/img_{cam}_%06d.jpg"')
            os.system(f'ffmpeg -i {MI3_DATASET_DIR}/S{subj}/Seq{seq}/ChairMasks/video_{cam}.avi -qscale:v 1 -start_number 0 "{MI3_PP_DATASET_DIR}/S{subj}/Seq{seq}/chair_{cam}_%06d.jpg"')
            os.system(f'ffmpeg -i {MI3_DATASET_DIR}/S{subj}/Seq{seq}/FGmasks/video_{cam}.avi -qscale:v 1 -start_number 0 "{MI3_PP_DATASET_DIR}/S{subj}/Seq{seq}/fg_{cam}_%06d.jpg"')
            
            # Move all the files we want to keep to their respective directory and delete the rest
            outfile=open(f'{base_dir}/cleanup.sh', 'w')
            for idx in keep_idxs:
                print(f'mv {base_dir}/img_{cam}_{idx:06d}.jpg {base_dir}/img/', file=outfile)
                print(f'mv {base_dir}/chair_{cam}_{idx:06d}.jpg {base_dir}/chair/', file=outfile)
                print(f'mv {base_dir}/fg_{cam}_{idx:06d}.jpg {base_dir}/fg/', file=outfile)
            outfile.close()
            os.system(f'bash {base_dir}/cleanup.sh')
            os.system(f'rm {base_dir}/cleanup.sh')
            os.system(f'rm {base_dir}/*.jpg')

    # Zip up subject folder and delete uncompressed 
    os.system(f'tar -C {MI3_PP_DATASET_DIR} -c -z -f {MI3_PP_DATASET_DIR}/S{subj}.tgz S{subj}')
    os.system(f'rm -rf {MI3_PP_DATASET_DIR}/S{subj}')
