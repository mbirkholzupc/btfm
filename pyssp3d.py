import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image

from utils import plotMultiOnImage, clip_detect, GID

# 311 images total
MIN_IMAGE=0
MAX_IMAGE=310

# Joints are in COCO order. There are 17.
ssp3d_joints = [ 'nose', 'left_eye', 'right_eye',
                'left_ear', 'right_ear', 'left_shoulder',
                'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip',
                'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle' ]

class PySSP3D:
    def __init__(self, base_path, ssp3d_path):
        self._base_path = base_path
        self._ssp3d_path = ssp3d_path
        self._ssp3d_img_path = ssp3d_path+'/ssp_3d/images'
        self._ssp3d_sil_path = ssp3d_path+'/ssp_3d/silhouettes'
        self._ssp3d_data = np.load(base_path+ssp3d_path+'/ssp_3d/labels.npz')
        self._num_samples = self._ssp3d_data['fnames'].shape[0]
        self._img_files = self._ssp3d_data['fnames']
        self._joints = self._ssp3d_data['joints2D']
        self._shapes = self._ssp3d_data['shapes']
        self._poses = self._ssp3d_data['poses']
        self._genders = self._ssp3d_data['genders']

    def _image_path(self, index):
        if index < MIN_IMAGE or index > MAX_IMAGE:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {MAX_IMAGE}]')
        return self._ssp3d_img_path + '/' + self._img_files[index]
        
    def disp_image(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._base_path+self._image_path(index))
        img=img/255
        plt.imshow(img)
        plt.show()

    #def disp_annotations(self, index):
    #    print(self._image_path(index))
    #    # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
    #    # get values in range [0, 1]
    #    img=matplotlib.image.imread(self._base_path+self._image_path(index))
    #    img=img/255
    #    height=img.shape[0]
    #    width=img.shape[1]
    #    print(f'height: {height} // width: {width}')
    #    joints=self._joints[index]
    #    jointsx=self._joints[index][0::3]
    #    jointsy=self._joints[index][1::3]
    #    jointsv=self._joints[index][2::3]

    #    #  Note: Very few images actually need this. Mainly cosmetic, to get rid of
    #    #        some whitespace around the image and keep a 1:1 pixel mapping
    #    if clip_detect(jointsx, 0, width-1):
    #        np.clip(jointsx, 0, width-1, out=jointsx)
    #    if clip_detect(jointsy, 0, height-1):
    #        np.clip(jointsy, 0, height-1, out=jointsy)

    #    jointsxy=np.transpose(np.array([jointsy, jointsx]))

    #    #img=plotOnImage(img, jointsxy, 'ro')
    #    img=plotMultiOnImage(img, zip([jointsxy[0:6], jointsxy[7:]], ['ro', 'bo']))
    #    plt.imshow(img)
    #    plt.show()

    def gather_data(self, which_set, gid=None):
        if gid==None:
            # Start numbering from 0 if no GID given
            gid = GID()
        result = []
        if which_set == 'test':
            for i in range(self._num_samples):
                result.append(self._format_annotation(i,gid.next()))
        return result

    def _format_annotation(self, index, number):
        # Read width, height of image
        img=Image.open(self._base_path+self._image_path(index))
        width, height = img.size

        annotation = {}
        annotation['ID'] = number
        annotation['set'] = 'SSP3D'
        annotation['path'] = self._image_path(index)
        annotation['gender'] = self._genders[index]

        # Create bbox from joints
        jointsx=self._joints[index][:,0]
        jointsy=self._joints[index][:,1]
        bbox=self._joint_minimal_bbox(jointsx, jointsy, (width,height))
        if bbox != (0,0,0,0):
            annotation['bbox'] = bbox

        for j in range(len(ssp3d_joints)):
            for i, xy in enumerate(['x', 'y', 'v']):
                annotation[f'{xy}{j}'] = self._joints[index][j][i]

        # Add shape (betas) - First 10 PCA components
        annotation['betas']=self._shapes[index].tolist()

        annotation['silhouette'] = self._ssp3d_sil_path+'/'+self._img_files[index]
                
        return annotation

    def _valid_joints(self, jointsx, jointsy, dims):
        """
        _valid_joints: Function to decide which joints are valid. Each dataset
                       has different criteria, so can't have a single function. Boo.
        jointsx: ndarray joints x cood
        jointsy: ndarray joints y coord
        dims: (width, height)
        """
        # Note: In this dataset, all joints are valid, whether visible or not
        assert(jointsx.shape==jointsy.shape)

        return np.array([True]*jointsx.shape[0])

    def _joint_minimal_bbox(self, jointsx, jointsy, dims):
        # dims: (width, height)
        jx=jointsx.round(0)
        jy=jointsy.round(0)
        joint_mask=self._valid_joints(jx, jy, dims)

        # Check if we need to clip
        outside_image=(jx<0)|(jx>=dims[0])|(jy<0)|(jy>=dims[1])
        if outside_image.any():
            jx=jx.clip(0, dims[0]-1)
            jy=jy.clip(0, dims[1]-1)

        if joint_mask.any() == True:
            x0=int(jx[joint_mask].min())
            x1=int(jx[joint_mask].max())
            y0=int(jy[joint_mask].min())
            y1=int(jy[joint_mask].max())
        else:
            x0=x1=y0=y1=0
        return (x0, y0, x1, y1)
