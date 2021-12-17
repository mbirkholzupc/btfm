import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
import pickle

from utils import plotMultiOnImage, clip_detect, GID

MIN_IMAGE=0
MAX_IMAGE=24986

mpii_joints = [ 'r ankle', 'r knee', 'r hip',
                'l hip', 'l knee', 'l ankle',
                'pelvis', 'thorax', 'upper neck',
                'head top', 'r wrist', 'r elbow',
                'r shoulder', 'l shoulder', 'l elbow',
                'l wrist' ]
annorect_map = { 'id' : 0, 'scale' : 1, 'obj_x' : 2, 'obj_y' : 3,
    'head_x1' : 4, 'head_y1' : 5, 'head_x2' : 6, 'head_y2' : 7,
    'joints' : 8 } # Note: joints are from 8 to end (x, y, v; 16*3=48 total)

class PyMPII:
    def __init__(self, path_to_mpii_release, path_to_image):
        self._path_to_release = path_to_mpii_release
        self._path_to_image = path_to_image

        with open(self._path_to_release,'rb') as infile:
            self._release=pickle.load(infile)

        # Convert to pandas for ease of accesss
        self._df_annorect = pd.DataFrame(self._release['annorect'])

    def _image_path(self, index):
        if index < MIN_IMAGE or index > MAX_IMAGE:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {MAX_IMAGE}]')
        return self._path_to_image + '/' + self._release['annolist'][index][1]
        
    def disp_image(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._image_path(index))
        img=img/255
        plt.imshow(img)
        plt.show()

    def disp_annotations(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._image_path(index))
        img=img/255
        height=img.shape[0]
        width=img.shape[1]
        print(f'height: {height} // width: {width}')
        
        # Shorthand
        RELEASE=self._release

        if RELEASE['annorect'][RELEASE['annorect'][:,0]==RELEASE['annolist'][index][0]].shape[0] >= 1:
            # There are annotations to display
            #print('Annotations to display: ' + str(RELEASE['annorect'][RELEASE['annorect'][:,0]==RELEASE['annolist'][index][0]].shape[0]))
            # Just display the first one
            annotation = RELEASE['annorect'][RELEASE['annorect'][:,0]==RELEASE['annolist'][index][0]][0]
            print(annotation)

            jointsx=annotation[8::3]
            jointsy=annotation[9::3]
            jointsv=annotation[10::3]

            if clip_detect(jointsx, 0, width-1):
                np.clip(jointsx, 0, width-1, out=jointsx)
            if clip_detect(jointsy, 0, height-1):
                np.clip(jointsy, 0, height-1, out=jointsy)

            jointsxy=np.transpose(np.array([jointsy, jointsx]))

            bb = np.array([ [annotation[3], annotation[2]], [annotation[5], annotation[4]] ])

            #img=plotOnImage(img, jointsxy, 'ro')
            #img=plotMultiOnImage(img, zip([jointsxy[0:6], jointsxy[7:]], ['ro', 'bo']))
            #img=plotMultiOnImage(img, zip([jointsxy], ['ro']))
            img=plotMultiOnImage(img, zip([jointsxy, bb], ['ro', 'bo']))
        else:
            # There are no annotations. Maybe a test image?
            print('No annotations to display.')

        #joints=self._joints[index]
        #jointsx=self._joints[index][0::3]
        #jointsy=self._joints[index][1::3]
        #jointsv=self._joints[index][2::3]

        ##  Note: Very few images actually need this. Mainly cosmetic, to get rid of
        ##        some whitespace around the image and keep a 1:1 pixel mapping
        #if clip_detect(jointsx, 0, width-1):
        #    np.clip(jointsx, 0, width-1, out=jointsx)
        #if clip_detect(jointsy, 0, height-1):
        #    np.clip(jointsy, 0, height-1, out=jointsy)

        #jointsxy=np.transpose(np.array([jointsy, jointsx]))

        ##img=plotOnImage(img, jointsxy, 'ro')
        ##img=plotMultiOnImage(img, zip([jointsxy[0:6], jointsxy[7:]], ['ro', 'bo']))
        #img=plotMultiOnImage(img, zip([jointsxy], ['ro']))
        plt.imshow(img)
        plt.show()

    def gather_data(self, which_set, gid=None):
        if gid==None:
            # Start numbering from 0 if no GID given
            gid = GID()
        result = []
        if which_set == 'train':
            # This returns a tuple, so extract the part we care about
            train_idxs=np.where(self._release['img_train']==1)[0]
            for i in train_idxs:
                annot_result =self._format_annotation(i,gid.next())
                if annot_result is not None:
                    result.append(annot_result)
                else:
                    # Reuse the last ID since it wasn't assigned
                    gid.rollback()
        elif which_set == 'val':
            # No validation in this set
            pass
        elif which_set == 'test':
            # This returns a tuple, so extract the part we care about
            test_idxs=np.where(self._release['img_train']==0)[0]
            for i in test_idxs:
                result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'toy':
            # Grab the first 10 training instances
            # This returns a tuple, so extract the part we care about
            train_idxs=np.where(self._release['img_train']==1)[0][:10]
            for i in train_idxs:
                result.append(self._format_annotation(i,gid.next()))
        return result

    def _format_annotation(self, index, number):
        image_path = self._image_path(index)
        if image_path[-13:] in ['040348287.jpg', '013401523.jpg',
            '002878268.jpg' ]:
            # This file is missing
            return None
        img=matplotlib.image.imread(self._image_path(index))
        height=img.shape[0]
        width=img.shape[1]

        annotation = {}
        annotation['ID'] = number
        annotation['path'] = self._image_path(index)

        # TODO: If we need bbox, get from openpose. Right now, whole image
        #annotation['bbox_x'] = 0
        #annotation['bbox_y'] = 0
        #annotation['bbox_h'] = height
        #annotation['bbox_w'] = width

        # Load the first annotations for this image
        annolist_index=self._release['annolist'][index][0]
        annorect=self._df_annorect[self._df_annorect[0]==annolist_index]

        if annorect.shape[0] == 0:
            # This is totally normal, so no need to print a warning. Just return None
            #print("Warning: 0 annotations - index/annolist_index: "+str(index) + '/' +str(annolist_index))
            return None
        elif annorect.shape[0] < 0:
            print("Warning: negative annotations")
            assert(1==0)

        # Note: There may be multiple people per image, but we will only take the first one unless
        #       we need more data
        annorect=annorect.iloc[0]

        # Iterate over values from up to head_y2, but drop the first ID because we don't care about it
        for i,k in enumerate(list(annorect_map.keys())[:-1]):
            if k == 'id':
                continue
            annotation[k] = annorect[i]

        # Now deal with the joints
        iidx=8  # Initialize to index of first joint
        for j in range(len(mpii_joints)):
            for xyv in ['x', 'y', 'v' ]:
                annotation[f'{xyv}{j}'] = annorect[iidx]
                iidx += 1

        return annotation
