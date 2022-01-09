import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
from skimage import io
from pycocotools.coco import COCO

from utils import plotOnImage, clip_detect, GID

MIN_IMAGE=0

coco_joints = [ 'nose', 'left_eye', 'right_eye',
                'left_ear', 'right_ear', 'left_shoulder',
                'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip',
                'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle' ]

BBOX_IDX = { 'x' : 0, 'y' : 1, 'w' : 2, 'h' : 3 }

class PyCOCO:
    def __init__(self, base_path, path_to_trn_img, path_to_trn_annot, path_to_val_img, path_to_val_annot, path_to_test_img, path_to_test_info):
        self._coco_trn = PyCOCOTrainVal(base_path, path_to_trn_img, path_to_trn_annot)
        self._coco_val = PyCOCOTrainVal(base_path, path_to_val_img, path_to_val_annot)
        self._coco_tst = PyCOCOTest(base_path, path_to_test_img, path_to_test_info)

    def gather_data(self, which_set, gid=None):
        if gid==None:
            gid = GID()
        result = []

        if which_set == 'train':
            result=self._coco_trn.gather_data(which_set, gid)
        elif which_set == 'val':
            result=self._coco_val.gather_data(which_set, gid)
        elif which_set == 'test':
            result=self._coco_tst.gather_data(gid)
        elif which_set == 'toy':
            result=self._coco_trn.gather_data(which_set, gid)

        return result
    

class PyCOCOTrainVal:
    def __init__(self, base_path, path_to_img, path_to_annot):
        self._base_path = base_path
        self._path_to_img = path_to_img
        self._path_to_annot = path_to_annot

        # Load data set
        self._coco=COCO(self._base_path+self._path_to_annot)

        # Load categories. Code to retrieve names included in comment
        self._cats = self._coco.loadCats(self._coco.getCatIds())
        #nms=[cat['name'] for cat in cats]
        #print('COCO categories: \n{}\n'.format(' '.join(nms)))

        # get all images containing given categories, select one at random
        #catIds = self._coco.getCatIds(catNms=['person','dog','skateboard']);
        self._catIds = self._coco.getCatIds(catNms=['person']);
        self._imgIds = self._coco.getImgIds(catIds=self._catIds );
        self.num_images = len(self._imgIds)
        print('Found images: ' + str(self.num_images))

    def _image_path(self, index):
        if index < MIN_IMAGE or index > self.num_images-1:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {self.num_images-1}]')
        img = self._coco.loadImgs(self._imgIds[index])[0]

        return self._path_to_img + '/' + img['file_name']
        
    def disp_image(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._base_path+self._image_path(index))
        img=img/255
        plt.imshow(img)
        plt.show()

    def disp_annotations(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._base_path+self._image_path(index))
        img=img/255
        height=img.shape[0]
        width=img.shape[1]

        # Annotations
        img_meta = self._coco.loadImgs(self._imgIds[index])[0]
        annIds = self._coco.getAnnIds(imgIds=img_meta['id'], catIds=self._catIds, iscrowd=None)
        anns = self._coco.loadAnns(annIds)
        for a in anns:
            print(a)

        # Let's only work with the first one for now
        joints = np.array(anns[0]['keypoints'])
        print(joints)
        jointsx=joints[0::3]
        jointsy=joints[1::3]
        jointsv=joints[2::3]

        #  Note: Very few images actually need this. Mainly cosmetic, to get rid of
        #        some whitespace around the image and keep a 1:1 pixel mapping
        if clip_detect(jointsx, 0, width-1):
            np.clip(jointsx, 0, width-1, out=jointsx)
        if clip_detect(jointsy, 0, height-1):
            np.clip(jointsy, 0, height-1, out=jointsy)

        jointsxy=np.transpose(np.array([jointsy, jointsx]))

        img=plotOnImage(img, jointsxy, 'ro')
        plt.imshow(img)
        plt.show()

    def gather_data(self, which_set, gid=None):
        if gid==None:
            gid = GID()
        result = []

        if which_set == 'train':
            for i in range(self.num_images):
                result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'val':
            for i in range(self.num_images):
                result.append(self._format_annotation(i,gid.next()))
        elif which_set == 'toy':
            for i in range(10):
                result.append(self._format_annotation(i,gid.next()))
        else:
            print('Invalid set: ' + which_set)
            assert(1==0)

        return result

    def _format_annotation(self, index, number):

        img_meta = self._coco.loadImgs(self._imgIds[index])[0]
        annIds = self._coco.getAnnIds(imgIds=img_meta['id'], catIds=self._catIds, iscrowd=None)
        anns = self._coco.loadAnns(annIds)

        # We will only use the first annotation that shows up for each image. If we need more,
        # update thisannotation that shows up for each image. If we need more,
        # update this.
        if(len(anns)<1):
            print('Unexpected length: ' + str(len(anns)))
            print(anns)
            exit()

        # If height/width needed, uncomment
        #height=self._coco.dataset['images'][index]['height']
        #width=self._coco.dataset['images'][index]['width']

        annotation = {}
        annotation['ID'] = number
        annotation['path'] = self._image_path(index)
        annotation['bbox_x'] = anns[0]['bbox'][BBOX_IDX['x']]
        annotation['bbox_y'] = anns[0]['bbox'][BBOX_IDX['y']]
        annotation['bbox_w'] = anns[0]['bbox'][BBOX_IDX['w']]
        annotation['bbox_h'] = anns[0]['bbox'][BBOX_IDX['h']]
        ann_joints = anns[0]['keypoints']
        iidx=0
        for j in range(len(coco_joints)):
            for xy in ['x', 'y', 'v']:
                annotation[f'{xy}{j}'] = ann_joints[iidx]
                iidx+=1

        return annotation

class PyCOCOTest:
    def __init__(self, base_path, path_to_img, path_to_img_info):
        self._base_path = base_path
        self._path_to_img = path_to_img
        self._path_to_img_info = path_to_img_info

        # Load data set. Need to just work with raw JSON unfortunately.
        infile=open(self._base_path+self._path_to_img_info,'r')
        self._img_info=json.load(infile)
        infile.close()

        # As far as I know, there's no way to filter by category, so we'll
        # just handle all files in the test set
        self.num_images = len(self._img_info['images'])

        print('Found images: ' + str(self.num_images))

    def _image_path(self, index):
        if index < MIN_IMAGE or index > self.num_images-1:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {self.num_images-1}]')
        img = self._img_info['images'][index]

        return self._path_to_img + '/' + img['file_name']
        
    def disp_image(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._base_path+self._image_path(index))
        img=img/255
        plt.imshow(img)
        plt.show()

    def gather_data(self, gid=None):
        if gid==None:
            gid = GID()
        result = []

        # Test set only. Only available info will be path to image.
        for i in range(self.num_images):
            result.append(self._format_annotation(i,gid.next()))

        return result

    def _format_annotation(self, index, number):
        height=self._img_info['images'][index]['height']
        width=self._img_info['images'][index]['width']

        annotation = {}
        annotation['ID'] = number
        annotation['path'] = self._image_path(index)
        #annotation['bbox_x'] = 0
        #annotation['bbox_y'] = 0
        #annotation['bbox_h'] = height
        #annotation['bbox_w'] = width

        return annotation
