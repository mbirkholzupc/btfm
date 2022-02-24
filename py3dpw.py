import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
from skimage import io
from pathlib import Path
from scipy.spatial.distance import euclidean
from PIL import Image

from utils import filterJoints, plotMultiOnImage, clip_detect, GID

# Zero-indexed limits
MIN_IMAGE=0
MAX_IMAGE=74619

# Threshold to consider poses "different" in m
# Current: 40 mm
JOINT_DIFF_THRESHOLD=0.040

# Says it's COCO format (according to README), but joints don't match up
py3dpw_joints2d = [ 'nose', 'neck',
                    'right_shoulder', 'right_elbow', 'right_wrist',
                    'left_shoulder', 'left_elbow', 'left_wrist',
                    'right_hip', 'right_knee', 'right_ankle',
                    'left_hip', 'left_knee', 'left_ankle',
                    'right_eye', 'left_eye',
                    'right_ear', 'left_ear' ]

# SMPL format (according to README)
py3dpw_joints3d = [ 'pelvis', 'left_hip', 'right_hip',
                    'spine1', 'left_knee', 'right_knee',
                    'spine2', 'left_ankle', 'right_ankle',
                    'spine3', 'left_foot', 'right_foot',
                    'neck', 'left_collar', 'right_collar',
                    'head', 'left_shoulder', 'right_shoulder',
                    'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist',
                    'left_hand', 'right_hand' ]

def tdpw_valid_joints(index, jointsx, jointsy, jointsv, dims):
    # dims: (width, height)
    assert(jointsx.shape==jointsy.shape)
    assert(jointsx.shape==jointsv.shape)

    val_joints=(jointsv>0)

    # The only one of these rules that is "broken" is too-large X
    # We will clip it in the caller, so no need to signal invalid
    # Can keep to check in future if new data added
    #inval_joints=(jointsv==0)
    #xinvalidgt0=jointsx[inval_joints]>0
    #yinvalidgt0=jointsy[inval_joints]>0
    #if (xinvalidgt0).any():
    #    print('INVALID BUT NONZERO X')
    #    print(jointsx)
    #if (yinvalidgt0).any():
    #    print('INVALID BUT NONZERO Y')
    #    print(jointsy)
    ##if (jointsx>=dims[0]).any():
    ##    print(f'BAD X BIG {index} {dims[0]}')
    ##    print(dims)
    ##    print(jointsx)
    ##    print(jointsy)
    #if (jointsx<0).any():
    #    print(f'BAD X SMALL {index} {dims[0]}')
    #    print(dims)
    #    print(jointsx)
    #    print(jointsy)
    #if (jointsy>=dims[1]).any():
    #    print(f'BAD Y BIG {index} {dims[1]}')
    #    print(dims)
    #    print(jointsx)
    #    print(jointsy)
    #if (jointsy<0).any():
    #    print(f'BAD Y SMALL {index} {dims[1]}')
    #    print(dims)
    #    print(jointsx)
    #    print(jointsy)

    return val_joints

def tdpw_bbox(index, jointsx, jointsy, jointsv, dims):
    # dims: (width, height)
    jx=np.array(jointsx).round(0)
    jy=np.array(jointsy).round(0)
    jv=np.array(jointsv)
    joint_mask=tdpw_valid_joints(index, jx, jy, jv, dims)

    if joint_mask.any() == True:
        # Check if we need to clip
        outside_image=(jx<0)|(jx>=dims[0])|(jy<0)|(jy>=dims[1])
        if outside_image.any():
            jx=jx.clip(0, dims[0]-1)
            jy=jy.clip(0, dims[1]-1)

        x0=int(jx[joint_mask].min())
        x1=int(jx[joint_mask].max())
        y0=int(jy[joint_mask].min())
        y1=int(jy[joint_mask].max())
    else:
        x0=x1=y0=y1=0
    return (x0, y0, x1, y1)

class Py3DPW:
    def __init__(self, base_path, path_to_trn_annot, path_to_val_annot, path_to_tst_annot, path_to_img, path_to_silhouette, path_to_silhouette_valid, filter_same=True):
        self._base_path = base_path
        self._trn_path = path_to_trn_annot
        self._val_path = path_to_val_annot
        self._tst_path = path_to_tst_annot
        self._img_path = path_to_img
        self._path_to_silhouette = path_to_silhouette
        self._path_to_silhouette_valid = path_to_silhouette_valid

        # Need to read in all annotations from train, val and test folder pickles and
        # string them all together. Somehow need to create a common index.
        self._pkls = { 'train': [], 'val': [], 'test': [] }
        self._num_imgs = { 'train': 0, 'val': 0, 'test': 0}
        self._pkl_limits = []

        img_count=0
        for d,p in zip([self._base_path+self._trn_path, self._base_path+self._val_path, self._base_path+self._tst_path], self._pkls):
            pkl_files = sorted(list(Path(d).glob('*')))
            pkl_idx=0
            for f in pkl_files:
                # Read in contents of pickle file
                infile=open(f,'rb')
                one_pickle=pickle.load(infile,encoding='latin1')
                infile.close()

                # Deal with accounting and build index
                low = img_count
                actors = len(one_pickle['genders'])
                img_count += len(one_pickle['img_frame_ids'])*actors
                high = img_count-1
                self._pkl_limits.append((pkl_idx, one_pickle['sequence'], low, high, actors))
                pkl_idx += 1
                self._num_imgs[p] = self._num_imgs[p] + (len(one_pickle['img_frame_ids']))*actors

                # Save the pickle
                self._pkls[p].append(one_pickle)

        print(self._num_imgs)

        # Tests of some specific values
        # Keeping code around for verification if we make any changes
        #print(self._pkl_limits)
        #print('which 0: ' + str(self._convert_index(0)))
        #print('which 764: ' + str(self._convert_index(764)))
        #print('which 765: ' + str(self._convert_index(765)))
        #print('which 1000: ' + str(self._convert_index(1000)))
        #print('which 17903: ' + str(self._convert_index(17903)))
        #print('which 17904: ' + str(self._convert_index(17904)))
        #print('which 23542: ' + str(self._convert_index(23542)))
        #print('which 23543: ' + str(self._convert_index(23543)))
        #print('which 26412: ' + str(self._convert_index(26412)))
        #print('which 26413: ' + str(self._convert_index(26413)))
        #print('which 28644: ' + str(self._convert_index(28644)))
        #print('which 29230: ' + str(self._convert_index(29230)))
        #print('which 29231: ' + str(self._convert_index(29231)))
        #print('which 29817: ' + str(self._convert_index(29817)))
        #print('which 29818: ' + str(self._convert_index(29818)))
        #print('which 52652: ' + str(self._convert_index(52652)))
        #print('which 72797: ' + str(self._convert_index(72797)))
        #print('which 72798: ' + str(self._convert_index(72798)))
        #print('which 74619: ' + str(self._convert_index(74619)))

        #print(self._image_path(0))
        #print(self._image_path(1000))
        #print(self._image_path(17903))
        #print(self._image_path(17904))
        #print(self._image_path(23542))
        #print(self._image_path(23543))
        #print(self._image_path(26412))
        #print(self._image_path(26413))
        #print(self._image_path(28644))
        #print(self._image_path(29230))
        #print(self._image_path(29231))
        #print(self._image_path(29817))
        #print(self._image_path(29818))
        #print(self._image_path(52652))
        #print(self._image_path(72797))
        #print(self._image_path(72798))
        #print(self._image_path(74619))

        # Filter out images whose poses are considered identical
        # Start this off as a list because _filter_same_pose needs to traverse items in order
        self._filtered_idxs = list(range(MAX_IMAGE+1))
        print('Filtering out same poses...')
        print('Before:')
        print(len(self._filtered_idxs))
        if(filter_same):
            self._filtered_idxs=self._filter_same_pose(self._filtered_idxs)

        # Create a lookup table to check each index against preprocessed data
        print('Creating jm lookup...')
        self._jm_lookup=self._create_jm_lookup()
        print('Done creating jm lookup...')

        # Filter even further, discarding images that do not have joint annotations
        self._filtered_idxs=self._filter_no_joint_annotations(self._filtered_idxs)

        # Convert to set to make checking if it contains an element quicker
        self._filtered_idxs = set(self._filtered_idxs)
        print('After:')
        print(len(self._filtered_idxs))

    def _convert_index(self, index):
        assert(index >= 0)

        if index < self._num_imgs['train']:
            the_set='train'
        elif index < (self._num_imgs['train']+self._num_imgs['val']):
            the_set='val'
        elif index < (self._num_imgs['train']+self._num_imgs['val']+self._num_imgs['test']):
            the_set='test'
        else:
            # We know this will trigger if we get here, but want to display warning to user
            assert(index < self._num_imgs['train']+self._num_imgs['val']+self._num_imgs['test'])

        # Now, find the associated pickle. Already did bounds checking above.
        for x in self._pkl_limits:
            # Index 2, 3 are min, max limits. Need to return index 0, which is index of pickle.
            if index >= x[2] and index <= x[3]:
                break

        # Now, check which actor it is if there are two actors
        actor_idx = 0
        internal_index=index-x[2]

        num_images=(x[3]-x[2]+1)
        halfway=x[2]+(num_images/2)
        if (x[4] == 2) and (index>=halfway):
            actor_idx=1
            internal_index -= (num_images/2)

        internal_index=int(internal_index)

        return (the_set, x[0], actor_idx, internal_index)

    def _image_path(self, index):
        if index < MIN_IMAGE or index > MAX_IMAGE:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {MAX_IMAGE}]')

        # Convert to internal index and look up associated annotation data
        iidx = self._convert_index(index)
        annotation = self._pkls[iidx[0]][iidx[1]]

        return self._img_path + '/' + annotation['sequence'] + f'/image_{iidx[3]:05}.jpg'

    def disp_image(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._base_path+self._image_path(index))
        img=img/255
        plt.imshow(img)
        plt.show()

    def disp_annotations(self, index):
        print(self._image_path(index))
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._base_path+self._image_path(index))
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

    def gather_data(self, which_set, gid=None, filter_same=True):
        if gid==None:
            # Start numbering from 0 if no GID given
            gid = GID()
        result = []

        if which_set == 'train':
            for i in range(self._num_imgs['train']):
                if (not filter_same) or (filter_same and (i in self._filtered_idxs)):
                    result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'val':
            for i in range(self._num_imgs['train'], self._num_imgs['train']+self._num_imgs['val']):
                if (not filter_same) or (filter_same and (i in self._filtered_idxs)):
                    result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'test':
            for i in range(self._num_imgs['train']+self._num_imgs['val'], self._num_imgs['train']+self._num_imgs['val']+self._num_imgs['test']):
                if (not filter_same) or (filter_same and (i in self._filtered_idxs)):
                    result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'toy':
            for i in range(50):
                if (not filter_same) or (filter_same and (i in self._filtered_idxs)):
                    result.append(self._format_annotation(i,gid.next()))
        return result

    def _format_annotation(self, index, number):
        # Read width, height of image
        img=Image.open(self._base_path+self._image_path(index))
        width, height = img.size

        # Get the internal index and pickle for the sequence
        iidx = self._convert_index(index)
        pkl = self._pkls[iidx[0]][iidx[1]]

        # Look up gender
        gender = pkl['genders'][iidx[2]]

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

        # Look up shape (betas)
        betas=pkl['betas'][iidx[2]]
        if len(betas)>10:
            if (betas[10:]!=0).any():
                print('Warning: Truncating betas')

        # Calculate minimal 2D bbox
        bbox=tdpw_bbox(index, j2x, j2y, j2v, (width, height))

        annotation = {}
        annotation['ID'] = number
        annotation['set'] = '3DPW'
        annotation['path'] = self._image_path(index)
        annotation['gender'] = gender
        if bbox != (0,0,0,0):
            annotation['bbox'] = bbox


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

        # Add shape (betas) - Should generally be first 10 PCA components
        # Confirmed only 10 are used, although some zero-pad out to 300
        annotation['betas']=betas[0:10].tolist()

        # Add silhouette if there is a valid one
        if self._jm_lookup[index][1]:
            silhouette_filename=self._path_to_silhouette+'/'+pkl['sequence']+'/'
            actor=iidx[2]
            frame=iidx[3]
            silhouette_filename+=f'image_{frame:05d}_subj{actor}.png'
            annotation['silhouette']=silhouette_filename
                
        return annotation

    def _filter_same_pose(self, idxs):
        """
        Checks a list of images and returns only those that have at least one joint that
        moves at least 40 mm from its previous position
        """

        # Always add first frame
        unique_pose_idxs = [idxs[0]]

        # Build list of all 3D joints
        j=[]
        for idx in idxs:
            iidx = self._convert_index(idx)
            pkl = self._pkls[iidx[0]][iidx[1]]
            joints = pkl['jointPositions'][iidx[2]][iidx[3]]
            jx=joints[0::3]
            jy=joints[1::3]
            jz=joints[2::3]
            j.append(np.transpose([jx, jy, jz]))

        # Prime loop by loading the first joint, then add images only when they differ sufficiently
        # from last included image
        last_idx=0
        last_j=j[last_idx]
        for idx in idxs[1:]:
            cur_j = j[idx]
            d=np.array([euclidean(a, b) for a, b in zip(last_j, cur_j)]).max()
            if d > JOINT_DIFF_THRESHOLD:
                unique_pose_idxs.append(idx)
                last_idx=idx
                last_j=j[last_idx]

        return unique_pose_idxs

    def _filter_no_joint_annotations(self, idxs):
        """
        Remove image/subject pairs that don't have joint annotations
        """
        new_idxs=[]
        for idx in idxs:
            if self._jm_lookup[idx][0]:
                new_idxs.append(idx)

        return new_idxs

    def _create_jm_lookup(self):
        """
        Create a lookup table to convert from index to joint/mask validity
        """
        infile=open(self._base_path+self._path_to_silhouette_valid, 'rb')
        sil_valid_pkl=pickle.load(infile)
        infile.close()

        jm_lookup={}
        for i in range(MAX_IMAGE+1):
            iidx = self._convert_index(i)
            pkl = self._pkls[iidx[0]][iidx[1]]
            seq = pkl['sequence']
            actor = iidx[2]
            frame = iidx[3]

            pkl_key=f'{seq}/image_{frame:05d}.jpg'

            entry=sil_valid_pkl[pkl_key]
            jm_lookup[i]=(entry[0+actor], entry[2+actor])

        return jm_lookup
