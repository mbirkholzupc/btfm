import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
from PIL import Image

from utils import plotMultiOnImage, clip_detect, GID

MIN_IMAGE=1
MAX_IMAGE=10000

lsp_joints = [ 'right ankle', 'right knee', 'right hip', 
               'left hip', 'left knee', 'left ankle', 
               'right wrist', 'right elbow', 'right shoulder', 
               'left shoulder', 'left elbow', 'left wrist', 
               'neck', 'head top']

# These images have annotations that would either require manually fixing or too much logic in the 
# code, so let's drop them
poor_annotations=[6098, 8074]

class PyLSPET:
    def __init__(self, base_path, lsp_path, csv_path, upi_s1h_path):
        self._base_path = base_path
        self._lsp_path = lsp_path
        self._csv_path = csv_path
        self._upi_s1h_path = upi_s1h_path

        self._csv = pd.read_csv(self._base_path+self._csv_path)
        self._image_list = self._csv['image'].to_numpy()
        self._joints = self._csv.drop('image', axis=1).to_numpy()

        # Need to check which images actually have silhouettes
        # Save as a set in order to make it easy to check later
        full_path_upi_s1h=base_path+upi_s1h_path+'/data/lsp_extended'
        # Each filename is exactly 24 characters long, so super-easy to extract
        self._silhouettes = set([str(x)[-24:] for x in sorted(list(Path(full_path_upi_s1h).glob('*')))])


    def _image_path(self, index):
        index += 1  # Convert to 1-based
        if index < MIN_IMAGE or index > MAX_IMAGE:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {MAX_IMAGE}]')
        return self._lsp_path + '/images/im' + f'{index:05d}' + '.jpg'
        
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
        print(f'height: {height} // width: {width}')
        joints=self._joints[index]
        jointsx=self._joints[index][0::3]
        jointsy=self._joints[index][1::3]
        jointsv=self._joints[index][2::3]

        #  Note: Very few images actually need this. Mainly cosmetic, to get rid of
        #        some whitespace around the image and keep a 1:1 pixel mapping
        if clip_detect(jointsx, 0, width-1):
            np.clip(jointsx, 0, width-1, out=jointsx)
        if clip_detect(jointsy, 0, height-1):
            np.clip(jointsy, 0, height-1, out=jointsy)

        jointsxy=np.transpose(np.array([jointsy, jointsx]))

        #img=plotOnImage(img, jointsxy, 'ro')
        #img=plotMultiOnImage(img, zip([jointsxy[0:6], jointsxy[7:]], ['ro', 'bo']))
        img=plotMultiOnImage(img, zip([jointsxy], ['ro']))
        #plt.imshow(img)
        #plt.show()

    def gather_data(self, which_set, gid=None):
        if gid==None:
            # Start numbering from 0 if no GID given
            gid = GID()
        result = []
        if which_set == 'train':
            for i in range(MAX_IMAGE):
                # Images that are beyond recovery. Drop them.
                if i in poor_annotations:
                    continue
                result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'val':
            # No validation in this set
            pass
        elif which_set == 'test':
            # No test in this set
            pass
        if which_set == 'toy':
            for i in range(10):
                result.append(self._format_annotation(i,gid.next()))
        return result

    def _format_annotation(self, index, number):
        # Read width, height of image
        img=Image.open(self._base_path+self._image_path(index))
        width, height = img.size

        annotation = {}
        annotation['ID'] = number
        annotation['path'] = self._image_path(index)

        # Create bbox from joints
        jointsx=self._joints[index][0::3]
        jointsy=self._joints[index][1::3]
        bbox=self._joint_minimal_bbox(jointsx, jointsy, (width,height))
        annotation['bbox'] = bbox

        iidx=0
        for j in range(len(lsp_joints)):
            for xy in ['x', 'y', 'v']:
                annotation[f'{xy}{j}'] = self._joints[index][iidx]
                iidx+=1

        # If a silhouette file exists, write it. If not, skip
        sil_filename=f'im{index+1:05d}_segmentation.png'
        if sil_filename in self._silhouettes:
            annotation['silhouette'] = self._upi_s1h_path+'/data/lsp_extended/' + sil_filename
                
        return annotation

    def _valid_joints(self, jointsx, jointsy, dims):
        """
        _valid_joints: Function to decide which joints are valid. Each dataset
                       has different criteria, so can't have a single function. Boo.
        jointsx: ndarray joints x cood
        jointsy: ndarray joints y coord
        dims: (width, height)
        """
        # Note: In this dataset, joints are signaled invalid by a negative number in one coordinate
        #       and zero in the other or (0,0)
        assert(jointsx.shape==jointsy.shape)

        # Let's filter the negative and 0 case first
        fix_y=False

        negx=jointsx<0
        negy=jointsy<0

        # This case doesn't happen
        for x,y in zip(jointsx[negx], jointsy[negx]):
            if (x<0) and (y!=0):
                #print('Uh oh Y!')
                pass

        # This case happens 3 times
        # If it's -1, make it 0 and call valid. I guess they did this to get around using
        # 0 to signal invalid...
        for x,y in zip(jointsx[negy], jointsy[negy]):
            if (y<0) and (x!=0):
                #print('Uh oh X! Fixing...')
                fix_y=True
                break

        if fix_y:
            jointsy[jointsy==-1]=0
            negy=jointsy<0

        valid_idxs=np.logical_not(negx|negy)

        # Now, filter out (0,0) joints
        all_zero=(jointsx==0)&(jointsy==0)
        valid_idxs=valid_idxs&np.logical_not(all_zero)

        return valid_idxs

    def _joint_minimal_bbox(self, jointsx, jointsy, dims):
        # dims: (width, height)
        jx=jointsx.round(0)
        jy=jointsy.round(0)
        joint_mask=self._valid_joints(jx, jy, dims)

        # Only use valid joints from here
        jx=jx[joint_mask]
        jy=jy[joint_mask]

        if not joint_mask.any():
            assert(1==0)

        # Check if we need to clip
        outside_image=(jx<0)|(jx>=dims[0])|(jy<0)|(jy>=dims[1])
        if outside_image.any():
            jx=jx.clip(0, dims[0]-1)
            jy=jy.clip(0, dims[1]-1)

        if joint_mask.any() == True:
            x0=int(jx.min())
            x1=int(jx.max())
            y0=int(jy.min())
            y1=int(jy.max())
        else:
            x0=x1=y0=y1=0
        return (x0, y0, x1, y1)
