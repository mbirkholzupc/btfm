import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import os
from skimage import io
from PIL import Image
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

def coco_valid_joints(jointsx, jointsy, jointsv, dims):
    """
    coco_valid_joints: Function to decide which joints are valid. Each dataset
                       has different criteria, so can't have a single function. Boo.
    jointsx: ndarray joints x cood
    jointsy: ndarray joints y coord
    dims: (width, height)
    """
    assert(jointsx.shape==jointsy.shape)
    assert(jointsx.shape==jointsv.shape)

    val_joints=(jointsv>0)

    # This all seems unnecessary because no images "break the rules"
    # Can keep to check in future if new data added
    #inval_joints=(jointsv==0)
    #zerox=jointsx[inval_joints]
    #zeroy=jointsy[inval_joints]
    #if (zerox|zeroy).any():
    #    print('INVALID BUT NONZERO X/Y')
    #    print(jointsx)
    #    print(jointsy)
    #    print(jointsv)
    #if (jointsx>=dims[0]).any():
    #    print(f'BAD X BIG {index} {dims[0]}')
    #    print(dims)
    #    print(jointsx)
    #    print(jointsy)
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

def coco_bbox(index, jointsx, jointsy, jointsv, dims):
    # dims: (width, height)
    jx=np.array(jointsx).round(0)
    jy=np.array(jointsy).round(0)
    jv=np.array(jointsv)
    joint_mask=coco_valid_joints(jx, jy, jv, dims)

    if joint_mask.any() == True:
        x0=int(jx[joint_mask].min())
        x1=int(jx[joint_mask].max())
        y0=int(jy[joint_mask].min())
        y1=int(jy[joint_mask].max())
    else:
        x0=x1=y0=y1=0

    return (x0, y0, x1, y1)

class PyCOCO:
    def __init__(self, base_path, path_to_trn_img, path_to_trn_annot, path_to_val_img, path_to_val_annot, path_to_test_img, path_to_test_info, path_to_silhouette):
        self._coco_trn = PyCOCOTrainVal(base_path, path_to_trn_img, path_to_trn_annot, path_to_silhouette)
        self._coco_val = PyCOCOTrainVal(base_path, path_to_val_img, path_to_val_annot, path_to_silhouette)
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
    def __init__(self, base_path, path_to_img, path_to_annot, path_to_silhouette):
        self._base_path = base_path
        self._path_to_img = path_to_img
        self._path_to_annot = path_to_annot
        self._path_to_silhouette = path_to_silhouette

        # Create pre-processing dirs if they don't exist
        # Test images don't have keypoints/segmentations, so only train/val/toy
        os.system(f'mkdir -p {base_path}{path_to_silhouette}/train')
        os.system(f'mkdir -p {base_path}{path_to_silhouette}/val')
        os.system(f'mkdir -p {base_path}{path_to_silhouette}/toy')

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
                the_annotations=self._get_annotations(i, which_set)
                self._format_annotations(result,i,the_annotations,gid)
        elif which_set == 'val':
            for i in range(self.num_images):
                the_annotations=self._get_annotations(i, which_set)
                self._format_annotations(result,i,the_annotations,gid)
        elif which_set == 'toy':
            for i in range(10):
                the_annotations=self._get_annotations(i, which_set)
                self._format_annotations(result,i,the_annotations,gid)
        else:
            print('Invalid set: ' + which_set)
            assert(1==0)

        return result

    def _get_annotations(self, index, which_set):
        img_meta = self._coco.loadImgs(self._imgIds[index])[0]
        annIds = self._coco.getAnnIds(imgIds=img_meta['id'], catIds=self._catIds, iscrowd=None)
        anns = self._coco.loadAnns(annIds)

        # Just check that we never see 0 annotations
        if(len(anns)<1):
            print('Unexpected length: ' + str(len(anns)))
            print(anns)
            exit()

        # Image-level stuff: path, width, height
        # Note: height/width in metdata doesn't agree with this but this number is right
        image_path = self._image_path(index)
        image_filename_only=image_path[-16:-4]
        img=Image.open(self._base_path+self._image_path(index))
        width, height = img.size

        annotations=[]

        for i, ann in enumerate(anns):
            annotation = {}
            # Leave out ID in case there aren't any useful annotations
            annotation['path'] = image_path
            annotation['width'] = width
            annotation['height'] = height

            # Calculate minimal bbox from joints
            jointsx=ann['keypoints'][0::3]
            jointsy=ann['keypoints'][1::3]
            jointsv=ann['keypoints'][2::3]
            bbox=coco_bbox(index, jointsx, jointsy, jointsv, (width, height))
            if bbox != (0,0,0,0):
                annotation['jointsx']=ann['keypoints'][0::3]
                annotation['jointsy']=ann['keypoints'][1::3]
                annotation['jointsv']=ann['keypoints'][2::3]
                annotation['bbox'] = bbox

                # See if there is silhouette info available. If so, generate silhouette.
                # Note: There appears to always be segmentation available if there are keypoints
                if 'segmentation' in ann:
                    mask = self._coco.annToMask(ann)
                    filename=self._path_to_silhouette+'/'+which_set+'/'
                    filename+=f'{image_filename_only}_{i:02d}.png'
                    masked_image = np.zeros_like(mask, dtype=np.uint8)
                    masked_image[:,:] = np.where((mask==1), 255, 0)
                    im=Image.fromarray(masked_image)
                    im.save(self._base_path+filename)
                    annotation['silhouette']=filename


                annotations.append(annotation)

        return annotations

    def _format_annotations(self, results, index, annotations, gid):
        for ann in annotations:
            annotation = {}
            annotation['ID'] = gid.next()
            annotation['set'] = 'COCO'
            annotation['path'] = ann['path']

            # Minimal bounding box containing joints
            annotation['bbox']=ann['bbox']

            jointsx=ann['jointsx']
            jointsy=ann['jointsy']
            jointsv=ann['jointsv']
            # TODO: Remove this when confirmed no invalid joints
            jx=np.array(jointsx).round(0)
            jy=np.array(jointsy).round(0)
            jv=np.array(jointsv)
            valj=coco_valid_joints(jx, jy, jv, (ann['width'], ann['height']))
            assert(valj.any())

            # Now deal with the joints
            for j in range(len(coco_joints)):
                annotation[f'x{j}'] = ann['jointsx'][j]
                annotation[f'y{j}'] = ann['jointsy'][j]
                annotation[f'v{j}'] = ann['jointsv'][j]

            if 'silhouette' in ann:
                annotation['silhouette']=ann['silhouette']

            # Not sure if we need this
            # Now, add silhouette info if available
            #if uann[1] is not None:
            #    silhouette_filename = uann[1][2]  # In numpy format, image_name is index 2
            #    silhouette_filename = silhouette_filename[:5]+'_segmentation_full.png'
            #    annotation['silhouette'] = self._base_path+self._upi_s1h_img+'/'+silhouette_filename

            results.append(annotation)

        return

    def _format_annotation(self, index, number):
        # Deprecated, but leaving for reference
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
        # Update: Not sure what these are, but they don't match the image height/width
        #         Use PIL instead
        #cocoheight=self._coco.dataset['images'][index]['height']
        #cocowidth=self._coco.dataset['images'][index]['width']

        # Get image height/width from image (height/width in dataset is not this)
        img=Image.open(self._base_path+self._image_path(index))
        width, height = img.size

        annotation = {}
        annotation['ID'] = number
        annotation['path'] = self._image_path(index)
        #print(f"{annotation['path']} {index}")

        # Calculate minimal bbox from joints
        jointsx=anns[0]['keypoints'][0::3]
        jointsy=anns[0]['keypoints'][1::3]
        jointsv=anns[0]['keypoints'][2::3]
        bbox=coco_bbox(index, jointsx, jointsy, jointsv, (width, height))
        if bbox != (0,0,0,0):
            annotation['bbox'] = bbox
        #if index>5:
        #    exit()

        # COCO-format bboxes, just in case we need it sometime
        #annotation['bbox_x'] = anns[0]['bbox'][BBOX_IDX['x']]
        #annotation['bbox_y'] = anns[0]['bbox'][BBOX_IDX['y']]
        #annotation['bbox_w'] = anns[0]['bbox'][BBOX_IDX['w']]
        #annotation['bbox_h'] = anns[0]['bbox'][BBOX_IDX['h']]

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
        annotation['set'] = 'COCO'
        annotation['path'] = self._image_path(index)

        # If height/width needed, uncomment
        # Get image height/width from image (height/width in dataset is not this)
        #img=Image.open(self._base_path+self._image_path(index))
        #width, height = img.size

        #annotation['bbox_x'] = 0
        #annotation['bbox_y'] = 0
        #annotation['bbox_h'] = height
        #annotation['bbox_w'] = width

        return annotation
