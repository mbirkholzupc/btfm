import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
from scipy.spatial.distance import euclidean
from scipy.io import loadmat
import h5py
import pickle
from PIL import Image

from utils import filterJoints, plotMultiOnImage, clip_detect, GID
from pymi3_utils import mpii_get_sequence_info

# Training info
SUBJS = [1, 2, 3, 4, 5, 6, 7, 8]
SEQS = [1, 2]
# Default camera set for download (minus extra wall/ceiling cameras)
#DEFAULT_CAMERAS = [0, 1, 2, 4, 5, 6, 7, 8]
# Different sets of cameras for reference
#ALL_CAMERAS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#EXTRA_WALL_CAMERAS=[9, 3, 10]
#EXTRA_CEILING_CAMERAS=[11,12,13]
# Chosen set:
#  - extra wall: just 9 since no other gives a top-down view
#  - 1 and 4 almost identical, so keep only 1 since 4 is a narrower FOV
#  - Extra ceiling cameras show actor upside-down. Not sure if we want this. Omit.
CAMERAS = [0, 1, 2, 5, 6, 7, 8, 9]
# Genders per subject
GENDERS={1:'f', 2:'m', 3:'m', 4:'f', 5:'f', 6:'f', 7:'m', 8:'m'}

# Test info: 6 sequences with 5 unique subjects (3, 4 are same subject but different sequences)
# Only one camera
TSEQS = [1, 2, 3, 4, 5, 6]
# Genders per sequence
TGENDERS={1:'m', 2:'m', 3:'m', 4:'m', 5:'m', 6:'f'}

# TODO: Verify both through visualization

pymi3_joints = [ 'spine3', 'spine4', 'spine2', 'spine', 'pelvis', # 5
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', # 11
       'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', # 17
       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', # 23
       'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe' ]

pymi3_test_joints = [ 'head_top', 'neck',
                      'right_shoulder', 'right_elbow', 'right_wrist',
                      'left_shoulder', 'left_elbow', 'left_wrist',
                      'right_hip', 'right_knee', 'right_ankle',
                      'left_hip', 'left_knee', 'left_ankle',
                      'pelvis', 'spine', 'head' ]

def mi3_idx(subj, seq):
    return f'/S{subj}/Seq{seq}'

def mi3_tst_idx(seq):
    return f'/TS{seq}'

def mi3_bbox(jointsx, jointsy, dims):
    # dims: (width, height)
    # Note: All joints are considered "valid," but we'll clip and make sure that
    #       the bbox actually has a positive area before deciding whether to
    #       include it
    jx=np.array(jointsx).round(0)
    jy=np.array(jointsy).round(0)

    # Check if we need to clip
    outside_image=(jx<0)|(jx>=dims[0])|(jy<0)|(jy>=dims[1])
    if outside_image.any():
        jx=jx.clip(0, dims[0]-1)
        jy=jy.clip(0, dims[1]-1)

    x0=int(jx.min())
    x1=int(jx.max())
    y0=int(jy.min())
    y1=int(jy.max())

    # Check area. If positive, we're good. Otherwise return 0 bbox
    area=(x1-x0)*(y1-y0)
    if area<=0:
        x0=x1=y0=y1=0

    return (x0, y0, x1, y1)

class PyMI3:
    def __init__(self, base_path, trn_path, test_path, pp_path):
        self._base_path = base_path
        self._trn_path = trn_path
        self._test_path = test_path
        self._pp_path = pp_path

        # Load lists of valid frames
        infile=open(self._base_path+self._pp_path+'/frames/mi3_pp_frames.pkl', 'rb')
        self._trn_frames = pickle.load(infile)
        infile.close()
        infile=open(self._base_path+self._pp_path+'/frames/mi3_pp_test_frames.pkl', 'rb')
        self._tst_frames = pickle.load(infile)
        infile.close()

        # Create dict of all combinations of subj/seq to look up annotations for each
        # Note: Training annotations are in "normal" MATLAB format and can be read by loadmat
        self._trn_annotations={}
        for subj in SUBJS:
            for seq in SEQS:
                ann_path=self._base_path+self._trn_path+mi3_idx(subj,seq)+'/annot.mat'
                matfile=loadmat(ann_path)
                self._trn_annotations[mi3_idx(subj,seq)]=matfile
                print(f"{mi3_idx(subj,seq)}: {self._trn_annotations[mi3_idx(subj,seq)]['annot2'][0][0].shape}")

        # Create dict of all test annotations too
        # Note: Training annotations are in MATLAB v7.3 format and must be read as HDF5
        self._tst_annotations={}
        for seq in TSEQS:
            ann_path=self._base_path+self._test_path+mi3_tst_idx(seq)+'/annot_data.mat'
            self._tst_annotations[mi3_tst_idx(seq)]=h5py.File(ann_path,'r')
            print(f"{mi3_tst_idx(seq)}: {self._tst_annotations[mi3_tst_idx(seq)]['annot2'].shape}")

        # Now create indices for all train/test images
        self._trn_index=[]
        for subj in SUBJS:
            for seq in SEQS:
                for cam in CAMERAS:
                    frames=self._trn_frames[mi3_idx(subj,seq)]

                    for i, f in enumerate(frames):
                        tmp_entry=self._create_index_entry(subj, seq, cam, i, f)
                        if tmp_entry:
                            self._trn_index.append(tmp_entry)
        print(f'Num train frames: {len(self._trn_index)}')
        print(f'First ones: {self._trn_index[0:10]}')
        print(self._train_image_path(self._trn_index[0]))

        self._tst_index=[]
        for seq in TSEQS:
            frames=self._tst_frames[mi3_tst_idx(seq)]

            for i, f in enumerate(frames):
                tmp_entry=self._create_tst_index_entry(seq, i, f)
                if tmp_entry:
                    self._tst_index.append(tmp_entry)
        print(f'Num test frames: {len(self._tst_index)}')
        print(f'First ones: {self._tst_index[0:10]}')
        print(self._base_path+self._test_image_path(self._tst_index[0]))

        # Store min/max limits (0-indexed)
        self._min_image=0
        self._num_trn_image=len(self._trn_index)
        self._num_tst_image=len(self._tst_index)
        self._max_image=len(self._trn_index)+len(self._tst_index)-1

        # Path tests
        #print(self._image_path(0))
        #print(self._image_path(1000))
        #print(self._image_path(779950))
        #print(self._image_path(779951))
        #print(self._image_path(779952))
        #print(self._image_path(800000))
        #print(self._image_path(1000000))

    def _train_image_path(self,entry):
        path=f"{self._pp_path}/S{entry[0]}/Seq{entry[1]}/img/img_{entry[2]}_{entry[4]:06d}.jpg"
        return path

    def _test_image_path(self,entry):
        path=f"{self._test_path}/TS{entry[0]}/imageSequence/img_{entry[2]:06d}.jpg"
        return path

    def _image_path(self, index):
        if index < self._min_image or index > self._max_image:
            raise Exception(f'Invalid image index: {index}. Must be in range [{self._min_image}, {self._max_image}]')

        # Figure out train/test, then generate path
        path=""
        if index<self._num_trn_image:
            path=self._train_image_path(self._trn_index[index])
        else:
            path=self._test_image_path(self._tst_index[index-self._num_trn_image])

        return path

    def _silhouette_path(self, index):
        if index < self._min_image or index > self._max_image:
            raise Exception(f'Invalid image index: {index}. Must be in range [{self._min_image}, {self._max_image}]')

        # Silhouettes only exist for train set. Return empty string if not in train set.
        path=""
        if index<self._num_trn_image:
            entry=self._trn_index[index]
            path=f"{self._pp_path}/S{entry[0]}/Seq{entry[1]}/fg/fg_{entry[2]}_{entry[4]:06d}.jpg"

        return path


    """
    def disp_image(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._image_path(index))
        img=img/255
        plt.imshow(img)
        plt.show()

    def disp_annotations(self, index):
        print(self._image_path(index))
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._image_path(index))
        img=img/255
        height=img.shape[0]
        width=img.shape[1]

        # Look up the 2D joint positions
        iidx = self._convert_index(index)
        pkl = self._pkls[iidx[0]][iidx[1]]
        joints = pkl['poses2d'][iidx[2]][iidx[3]]
        print(joints.shape)
        print(joints)
        
        print(f'height: {height} // width: {width}')
        jointsx=joints[0]
        jointsy=joints[1]
        jointsv=joints[2]  # Not sure how to interpret this - confidence maybe?

        #  Note: Very few images actually need this. Mainly cosmetic, to get rid of
        #        some whitespace around the image and keep a 1:1 pixel mapping
        if clip_detect(jointsx, 0, width-1):
            np.clip(jointsx, 0, width-1, out=jointsx)
        if clip_detect(jointsy, 0, height-1):
            np.clip(jointsy, 0, height-1, out=jointsy)

        jointsxy=np.transpose(np.array([jointsy, jointsx]))
        jointsxy=filterJoints(jointsxy, jointsv)
        img=plotMultiOnImage(img, zip([jointsxy], ['ro']))
        #thejoint=1
        #img=plotMultiOnImage(img, zip([jointsxy[0:thejoint+1], jointsxy[thejoint+1:]], ['ro', 'bo']))
        plt.imshow(img)
        plt.show()
    """

    def gather_data(self, which_set, gid=None, filter_same=True):
        if gid==None:
            # Start numbering from 0 if no GID given
            gid = GID()
        result = []

        if which_set == 'train':
            for i in range(self._num_trn_image):
                self._format_train_annotation(result, i, gid)
        elif which_set == 'val':
            # No val set
            pass
        elif which_set == 'test':
            for i in range(self._num_tst_image):
                self._format_test_annotation(result, i, gid)
        elif which_set == 'toy':
            for i in range(100):
                self._format_train_annotation(result, i, gid)
        return result

    def _create_index_entry(self, subj, seq, cam, index, frame):
        # Start by checking for some missing data
        if (subj==3) and (seq==2) and (cam==13) and (frame>7210):
            # Missing RGB images above 7210 (exclusive)
            # Nothing else to do, so get out
            return None

        entry=(subj, seq, cam, index, frame)

        return entry

    def _create_tst_index_entry(self, seq, index, frame):
        entry=(seq, index, frame)

        return entry

    def _format_train_annotation(self, result, index, gid):
        # Look up subj/seq/etc.
        entry=self._trn_index[index]
        subj=entry[0]
        seq=entry[1]
        cam=entry[2]
        frame=entry[4]

        # Look up the annotations provided with the dataset
        ds_anns=self._trn_annotations[mi3_idx(subj,seq)]
        gender = GENDERS[subj]

        # Start by checking for some missing data. Some RGB frames missing from one sequence
        # and silhouette data from another one.
        missing_silhouette=False
        if (subj==3) and (seq==2) and (cam==13) and (frame>7210):
            # Missing RGB images above 7210 (exclusive)
            # Nothing else to do, so get out
            return
        elif (subj==7) and (seq==2) and (cam==9) and (frame>4978):
            # Missing FG silhouettes above 4978 (exclusive)
            # Set a limit and check this when we prepare to write silhouette info
            missing_silhouette=True

        # Read width, height of image
        img=Image.open(self._base_path+self._image_path(index))
        width, height = img.size


        # Look up the 2D joint positions
        j2d=ds_anns['annot2'][cam][0][frame]
        j2x=j2d[0::2]
        j2y=j2d[1::2]

        # Look up 3D joints
        j3d=ds_anns['annot3'][cam][0][frame]
        j3x=j3d[0::3]
        j3y=j3d[1::3]
        j3z=j3d[2::3]

        # Generate the path to the silhouette file
        if not missing_silhouette:
            silhouette_filename=self._silhouette_path(index)
        else:
            silhouette_filename=""

        # Now write the info to the DB
        annotation = {}
        annotation['ID'] = gid.next()
        annotation['set'] = 'MI3'
        annotation['path'] = self._image_path(index)
        annotation['gender'] = gender

        # Minimal 2D BBOX
        bbox=mi3_bbox(j2x, j2y, (width, height))
        if bbox != (0,0,0,0):
            annotation['bbox'] = bbox


        # Add 2D joints
        for j in range(len(pymi3_joints)):
            annotation[f'x{j}'] = j2x[j]
            annotation[f'y{j}'] = j2y[j]

        # Add 3D joints
        for j in range(len(pymi3_joints)):
            annotation[f'3d_x{j}'] = j3x[j]
            annotation[f'3d_y{j}'] = j3y[j]
            annotation[f'3d_z{j}'] = j3z[j]

        # Add silhouette, if available
        if silhouette_filename != "":
            annotation['silhouette'] = silhouette_filename

        result.append(annotation)
                
        return

    def _format_test_annotation(self, result, index, gid):
        # Look up seq/etc.
        entry=self._tst_index[index]
        seq=entry[0]
        frame=entry[2]

        # Look up the annotations provided with the dataset
        ds_anns=self._tst_annotations[mi3_tst_idx(seq)]
        gender = TGENDERS[seq]

        # Read width, height of image
        # Add num train images since they come before test images in global index
        img=Image.open(self._base_path+self._image_path(index+self._num_trn_image))
        width, height = img.size

        # Look up the 2D joint positions (frame-1 since MATLAB is 1-based)
        j2d=ds_anns['annot2'][frame-1][0]
        j2x=j2d[:,0]
        j2y=j2d[:,1]

        # Look up 3D joints (frame-1 since MATLAB is 1-based)
        j3d=ds_anns['annot3'][frame-1][0]
        j3x=j3d[:,0]
        j3y=j3d[:,1]
        j3z=j3d[:,2]

        # Now write the info to the DB
        annotation = {}
        annotation['ID'] = gid.next()
        annotation['set'] = 'MI3'
        annotation['path'] = self._image_path(index+self._num_trn_image)
        annotation['gender'] = gender

        # Minimal 2D BBOX
        bbox=mi3_bbox(j2x, j2y, (width, height))
        if bbox != (0,0,0,0):
            annotation['bbox'] = bbox

        # Add 2D joints
        for j in range(len(pymi3_test_joints)):
            annotation[f'x{j}'] = j2x[j]
            annotation[f'y{j}'] = j2y[j]

        # Add 3D joints
        for j in range(len(pymi3_test_joints)):
            annotation[f'3d_x{j}'] = j3x[j]
            annotation[f'3d_y{j}'] = j3y[j]
            annotation[f'3d_z{j}'] = j3z[j]

        result.append(annotation)
                
        return

        """
        img=matplotlib.image.imread(self._image_path(index))
        height=img.shape[0]
        width=img.shape[1]

        # Get the internal index and pickle for the sequence
        iidx = self._convert_index(index)
        pkl = self._pkls[iidx[0]][iidx[1]]

        # Look up the 2D joint positions
        j2d = pkl['poses2d'][iidx[2]][iidx[3]]
        j2x=j2d[0]
        j2y=j2d[1]
        j2v=j2d[2]  # Not sure how to interpret this - confidence maybe?

        # Look up 3D joints
        j3d = pkl['jointPositions'][iidx[2]][iidx[3]]
        j3x=j3d[0::3]
        j3y=j3d[1::3]
        j3z=j3d[2::3]

        annotation = {}
        annotation['ID'] = number
        annotation['path'] = self._image_path(index)
        annotation['bbox_x'] = 0
        annotation['bbox_y'] = 0
        annotation['bbox_h'] = height
        annotation['bbox_w'] = width


        # Add 2D joints
        for j in range(len(py3dpw_joints2d)):
            annotation[f'x{j}'] = j2x[j]
            annotation[f'y{j}'] = j2y[j]
            annotation[f'v{j}'] = j2v[j]

        # Add 3D joints
        for j in range(len(py3dpw_joints3d)):
            annotation[f'3d_x{j}'] = j3x[j]
            annotation[f'3d_y{j}'] = j3y[j]
            annotation[f'3d_z{j}'] = j3z[j]
                
        return annotation
        """

#    def _filter_same_pose(self, idxs):
#        """
#        Checks a list of images and returns only those that have at least one joint that
#        moves at least 40 mm from its previous position
#        """
#
#        # Always add first frame
#        unique_pose_idxs = [idxs[0]]
#
#        # Build list of all 3D joints
#        j=[]
#        for idx in idxs:
#            iidx = self._convert_index(idx)
#            pkl = self._pkls[iidx[0]][iidx[1]]
#            joints = pkl['jointPositions'][iidx[2]][iidx[3]]
#            jx=joints[0::3]
#            jy=joints[1::3]
#            jz=joints[2::3]
#            j.append(np.transpose([jx, jy, jz]))
#
#        # Prime loop by loading the first joint, then add images only when they differ sufficiently
#        # from last included image
#        last_idx=0
#        last_j=j[last_idx]
#        for idx in idxs[1:]:
#            cur_j = j[idx]
#            d=np.array([euclidean(a, b) for a, b in zip(last_j, cur_j)]).max()
#            if d > JOINT_DIFF_THRESHOLD:
#                unique_pose_idxs.append(idx)
#                last_idx=idx
#                last_j=j[last_idx]
#
#        return unique_pose_idxs
