import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
from skimage import io
from pathlib import Path

from utils import plotMultiOnImage, clip_detect, GID

# Zero-indexed limits
MIN_IMAGE=0
MAX_IMAGE=74619

lsp_joints = [ 'right ankle', 'right knee', 'right hip', 
               'left hip', 'left knee', 'left ankle', 
               'right wrist', 'right elbow', 'right shoulder', 
               'left shoulder', 'left elbow', 'left wrist', 
               'neck', 'head top']

class Py3DPW:
    def __init__(self, path_to_trn_annot, path_to_val_annot, path_to_tst_annot, path_to_img):
        self._trn_path = path_to_trn_annot
        self._val_path = path_to_val_annot
        self._tst_path = path_to_tst_annot
        self._img_path = path_to_img

        # Need to read in all annotations from train, val and test folder pickles and
        # string them all together. Somehow need to create a common index.
        self._pkls = { 'train': [], 'val': [], 'test': [] }
        self._num_imgs = { 'train': 0, 'val': 0, 'test': 0}
        self._pkl_limits = []

        img_count=0
        for d,p in zip([self._trn_path, self._val_path, self._tst_path], self._pkls):
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

        for s in self._pkls:
            for p in self._pkls[s]:
                print(p['sequence'])

        print(self._num_imgs)

        print(self._pkl_limits)

        print('which 0: ' + str(self._convert_index(0)))
        print('which 764: ' + str(self._convert_index(764)))
        print('which 765: ' + str(self._convert_index(765)))
        print('which 1000: ' + str(self._convert_index(1000)))
        print('which 17903: ' + str(self._convert_index(17903)))
        print('which 17904: ' + str(self._convert_index(17904)))
        print('which 23542: ' + str(self._convert_index(23542)))
        print('which 23543: ' + str(self._convert_index(23543)))
        print('which 26412: ' + str(self._convert_index(26412)))
        print('which 26413: ' + str(self._convert_index(26413)))
        print('which 28644: ' + str(self._convert_index(28644)))
        print('which 29230: ' + str(self._convert_index(29230)))
        print('which 29231: ' + str(self._convert_index(29231)))
        print('which 29817: ' + str(self._convert_index(29817)))
        print('which 29818: ' + str(self._convert_index(29818)))
        print('which 52652: ' + str(self._convert_index(52652)))
        print('which 72797: ' + str(self._convert_index(72797)))
        print('which 72798: ' + str(self._convert_index(72798)))
        print('which 74619: ' + str(self._convert_index(74619)))

        print(self._image_path(0))
        print(self._image_path(1000))
        print(self._image_path(17903))
        print(self._image_path(17904))
        print(self._image_path(23542))
        print(self._image_path(23543))
        print(self._image_path(26412))
        print(self._image_path(26413))
        print(self._image_path(28644))
        print(self._image_path(29230))
        print(self._image_path(29231))
        print(self._image_path(29817))
        print(self._image_path(29818))
        print(self._image_path(52652))
        print(self._image_path(72797))
        print(self._image_path(72798))
        print(self._image_path(74619))

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
        img=plotMultiOnImage(img, zip([jointsxy[0:6], jointsxy[7:]], ['ro', 'bo']))
        plt.imshow(img)
        plt.show()

    def gather_data(self, which_set, gid=None):
        if gid==None:
            # Start numbering from 0 if no GID given
            gid = GID()
        result = []
        if which_set == 'train':
            for i in range(self._num_imgs['train']):
                result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'val':
            for i in range(self._num_imgs['train'], self._num_imgs['train']+self._num_imgs['val']):
                result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'test':
            for i in range(self._num_imgs['val'], self._num_imgs['val']+self._num_imgs['test']):
                result.append(self._format_annotation(i,gid.next()))
            pass
        elif which_set == 'toy':
            for i in range(10):
                result.append(self._format_annotation(i,gid.next()))
            pass
        return result

    def _format_annotation(self, index, number):
        img=matplotlib.image.imread(self._image_path(index))
        height=img.shape[0]
        width=img.shape[1]

        annotation = {}
        annotation['ID'] = number
        annotation['path'] = self._image_path(index)
        annotation['bbox_x'] = 0
        annotation['bbox_y'] = 0
        annotation['bbox_h'] = height
        annotation['bbox_w'] = width
        iidx=0
        for j in range(len(lsp_joints)):
            for xy in ['x', 'y', 'v']:
                annotation[f'{xy}{j}'] = self._joints[index][iidx]
                iidx+=1
                
        return annotation
