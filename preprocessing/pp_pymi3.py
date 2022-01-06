import os

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.io import loadmat

from paths import *
from pymi3_utils import MpiiSeqInfo, mpii_get_sequence_info

# Threshold to consider poses "different" in mm
# Current: 40 mm
JOINT_DIFF_THRESHOLD=40

#SUBJS = [1, 2, 3, 4, 5, 6, 7, 8]
# 1, 2 done
SUBJS = [ 3, 4, 5, 6, 7, 8]
SEQS = [1, 2]
CAMERAS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#SUBJS = [1, 4]
#SUBJS = [4, 5]
#SEQS = [1, 2]
#CAMERAS = [11, 12]

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
        annot_path=f'{MI3_DIR}/S{subj}/Seq{seq}/annot.mat'
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
            base_dir=f'{MI3_PP_DIR}/S{subj}/Seq{seq}'
            os.system(f'mkdir -p {base_dir}/img')
            os.system(f'mkdir -p {base_dir}/chair')
            os.system(f'mkdir -p {base_dir}/fg')
            os.system(f'ffmpeg -i {MI3_DIR}/S{subj}/Seq{seq}/imageSequence/video_{cam}.avi -qscale:v 1 -start_number 0 "{MI3_PP_DIR}/S{subj}/Seq{seq}/img_{cam}_%06d.jpg"')
            os.system(f'ffmpeg -i {MI3_DIR}/S{subj}/Seq{seq}/ChairMasks/video_{cam}.avi -qscale:v 1 -start_number 0 "{MI3_PP_DIR}/S{subj}/Seq{seq}/chair_{cam}_%06d.jpg"')
            os.system(f'ffmpeg -i {MI3_DIR}/S{subj}/Seq{seq}/FGmasks/video_{cam}.avi -qscale:v 1 -start_number 0 "{MI3_PP_DIR}/S{subj}/Seq{seq}/fg_{cam}_%06d.jpg"')
            
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
    os.system(f'tar -C {MI3_PP_DIR} -c -z -f {MI3_PP_DIR}/S{subj}.tgz S{subj}')
    os.system(f'rm -rf {MI3_PP_DIR}/S{subj}')


#class PyMI3:
#        # Filter out images whose poses are considered identical
#        # Start this off as a list because _filter_same_pose needs to traverse items in order
#        self._filtered_idxs = list(range(MAX_IMAGE+1))
#        print('Before:')
#        print(len(self._filtered_idxs))
#        if(filter_same):
#            self._filtered_idxs=self._filter_same_pose(self._filtered_idxs)
#        # Convert to set to make checking if it contains an element quicker
#        self._filtered_idxs = set(self._filtered_idxs)
#        print('After:')
#        print(len(self._filtered_idxs))
#        print('Length 2D joints: ' + str(len(py3dpw_joints2d)))
#
#    def _convert_index(self, index):
#        assert(index >= 0)
#
#        if index < self._num_imgs['train']:
#            the_set='train'
#        elif index < (self._num_imgs['train']+self._num_imgs['val']):
#            the_set='val'
#        elif index < (self._num_imgs['train']+self._num_imgs['val']+self._num_imgs['test']):
#            the_set='test'
#        else:
#            # We know this will trigger if we get here, but want to display warning to user
#            assert(index < self._num_imgs['train']+self._num_imgs['val']+self._num_imgs['test'])
#
#        # Now, find the associated pickle. Already did bounds checking above.
#        for x in self._pkl_limits:
#            # Index 2, 3 are min, max limits. Need to return index 0, which is index of pickle.
#            if index >= x[2] and index <= x[3]:
#                break
#
#        # Now, check which actor it is if there are two actors
#        actor_idx = 0
#        internal_index=index-x[2]
#
#        num_images=(x[3]-x[2]+1)
#        halfway=x[2]+(num_images/2)
#        if (x[4] == 2) and (index>=halfway):
#            actor_idx=1
#            internal_index -= (num_images/2)
#
#        internal_index=int(internal_index)
#
#        return (the_set, x[0], actor_idx, internal_index)
#
#    def _image_path(self, index):
#        if index < MIN_IMAGE or index > MAX_IMAGE:
#            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {MAX_IMAGE}]')
#
#        # Convert to internal index and look up associated annotation data
#        iidx = self._convert_index(index)
#        annotation = self._pkls[iidx[0]][iidx[1]]
#
#        return self._img_path + '/' + annotation['sequence'] + f'/image_{iidx[3]:05}.jpg'
#
#    def disp_image(self, index):
#        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
#        # get values in range [0, 1]
#        img=matplotlib.image.imread(self._image_path(index))
#        img=img/255
#        plt.imshow(img)
#        plt.show()
#
#    def disp_annotations(self, index):
#        print(self._image_path(index))
#        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
#        # get values in range [0, 1]
#        img=matplotlib.image.imread(self._image_path(index))
#        img=img/255
#        height=img.shape[0]
#        width=img.shape[1]
#
#        # Look up the 2D joint positions
#        iidx = self._convert_index(index)
#        pkl = self._pkls[iidx[0]][iidx[1]]
#        joints = pkl['poses2d'][iidx[2]][iidx[3]]
#        print(joints.shape)
#        print(joints)
#        
#        print(f'height: {height} // width: {width}')
#        jointsx=joints[0]
#        jointsy=joints[1]
#        jointsv=joints[2]  # Not sure how to interpret this - confidence maybe?
#
#        #  Note: Very few images actually need this. Mainly cosmetic, to get rid of
#        #        some whitespace around the image and keep a 1:1 pixel mapping
#        if clip_detect(jointsx, 0, width-1):
#            np.clip(jointsx, 0, width-1, out=jointsx)
#        if clip_detect(jointsy, 0, height-1):
#            np.clip(jointsy, 0, height-1, out=jointsy)
#
#        jointsxy=np.transpose(np.array([jointsy, jointsx]))
#        jointsxy=filterJoints(jointsxy, jointsv)
#        img=plotMultiOnImage(img, zip([jointsxy], ['ro']))
#        #thejoint=1
#        #img=plotMultiOnImage(img, zip([jointsxy[0:thejoint+1], jointsxy[thejoint+1:]], ['ro', 'bo']))
#        plt.imshow(img)
#        plt.show()
#
#    def gather_data(self, which_set, gid=None, filter_same=True):
#        if gid==None:
#            # Start numbering from 0 if no GID given
#            gid = GID()
#        result = []
#
#        if which_set == 'train':
#            for i in range(self._num_imgs['train']):
#                if (not filter_same) or (filter_same and (i in self._filtered_idxs)):
#                    result.append(self._format_annotation(i,gid.next()))
#        elif which_set == 'val':
#            for i in range(self._num_imgs['train'], self._num_imgs['train']+self._num_imgs['val']):
#                if (not filter_same) or (filter_same and (i in self._filtered_idxs)):
#                    result.append(self._format_annotation(i,gid.next()))
#        elif which_set == 'test':
#            for i in range(self._num_imgs['train']+self._num_imgs['val'], self._num_imgs['train']+self._num_imgs['val']+self._num_imgs['test']):
#                if (not filter_same) or (filter_same and (i in self._filtered_idxs)):
#                    result.append(self._format_annotation(i,gid.next()))
#        elif which_set == 'toy':
#            for i in range(10):
#                if (not filter_same) or (filter_same and (i in self._filtered_idxs)):
#                    result.append(self._format_annotation(i,gid.next()))
#        return result
#
#    def _format_annotation(self, index, number):
#        img=matplotlib.image.imread(self._image_path(index))
#        height=img.shape[0]
#        width=img.shape[1]
#
#        # Get the internal index and pickle for the sequence
#        iidx = self._convert_index(index)
#        pkl = self._pkls[iidx[0]][iidx[1]]
#
#        # Look up the 2D joint positions
#        j2d = pkl['poses2d'][iidx[2]][iidx[3]]
#        j2x=j2d[0]
#        j2y=j2d[1]
#        j2v=j2d[2]  # Not sure how to interpret this - confidence maybe?
#
#        # Look up 3D joints
#        j3d = pkl['jointPositions'][iidx[2]][iidx[3]]
#        j3x=j3d[0::3]
#        j3y=j3d[1::3]
#        j3z=j3d[2::3]
#
#        annotation = {}
#        annotation['ID'] = number
#        annotation['path'] = self._image_path(index)
#        annotation['bbox_x'] = 0
#        annotation['bbox_y'] = 0
#        annotation['bbox_h'] = height
#        annotation['bbox_w'] = width
#
#
#        # Add 2D joints
#        for j in range(len(py3dpw_joints2d)):
#            annotation[f'x{j}'] = j2x[j]
#            annotation[f'y{j}'] = j2y[j]
#            annotation[f'v{j}'] = j2v[j]
#
#        # Add 3D joints
#        for j in range(len(py3dpw_joints3d)):
#            annotation[f'3d_x{j}'] = j3x[j]
#            annotation[f'3d_y{j}'] = j3y[j]
#            annotation[f'3d_z{j}'] = j3z[j]
#                
#        return annotation
#
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
