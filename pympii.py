import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
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

def mpii_valid_joints(jointsx, jointsy, dims):
    # dims: (width, height)
    # Note: First comparisons return False for nan, so should filter out missing joints
    in_image=(jointsx>=0)&(jointsy>=0)&(jointsx<=dims[0])&(jointsy<=dims[1])
    nonzero=(jointsx!=0)|(jointsy!=0)
    return in_image & nonzero

def mpii_bbox(jointsx, jointsy, dims):
    # dims: (width, height)
    joint_mask=mpii_valid_joints(jointsx, jointsy, dims)
    if joint_mask.any() == True:
        c0=int(jointsx[joint_mask].min())
        c1=int(jointsx[joint_mask].max())
        r0=int(jointsy[joint_mask].min())
        r1=int(jointsy[joint_mask].max())
    else:
        return None
    return (r0, c0, r1, c1)

def s1h_bbox(ulx, uly, lrx, lry, dims):
    # dims: width, height
    if lrx==dims[0] or lry==dims[1]:
        print('WARNING: s1h_bbox on edge!')
    if (0 <= int(ulx) <= dims[0]) and (0 <= int(lrx) < dims[0]) and \
       (0 <= int(uly) <= dims[1]) and (0 <= int(lry) < dims[1]):
        c0=int(ulx)
        c1=int(lrx)
        r0=int(uly)
        r1=int(lry)
    else:
        return None
    return (r0, c0, r1, c1)

def bbox_iou(box1, box2):
    iou=0

    # Format of each box: (r0, c0, r1, c1)
    # row/colum_up/down/left/right
    r_u=max(box1[0], box2[0])
    c_l=max(box1[1], box2[1])
    r_d=min(box1[2], box2[2])
    c_r=min(box1[3], box2[3])

    l=max(r_d-r_u,0)
    w=max(c_r-c_l,0)
    intersection=l*w

    # This also protects us against divide by zero
    if intersection>0:
        union=((box1[2]-box1[0])*(box1[3]-box1[1])+(box2[2]-box2[0])*(box2[3]-box2[1]))-intersection
        iou=intersection/union

    return iou

# Return 1 for True, 0 for False (want to sum numbers)
def j_in_bb(jx, jy, bb):
    rv=0
    if (bb[1]<=jx<=bb[3]) and (bb[0]<=jy<=bb[2]):
        rv=1
    return rv

def eval_joints_bbox(jx, jy, jval, bbox):
    score=0
    valjx=jx[jval]
    valjy=jy[jval]
    if jval.any()==True:
        score = np.mean([j_in_bb(j_x, j_y, bbox) for j_x, j_y in zip(valjx, valjy)])
    return score

def eval_bbox_iou(mpii_bbox, s1h_bbox):
    return bbox_iou(mpii_bbox, s1h_bbox)

class PyMPII:
    def __init__(self, base_path, path_to_mpii_release, path_to_image, upi_s1h_img, upi_s1h_annotation):
        self._base_path = base_path
        self._path_to_release = path_to_mpii_release
        self._path_to_image = path_to_image
        self._upi_s1h_img = upi_s1h_img

        with open(self._base_path+self._path_to_release,'rb') as infile:
            self._release=pickle.load(infile)

        # Convert to pandas for ease of accesss
        self._df_annorect = pd.DataFrame(self._release['annorect'])

        # Read in S1H data
        self._upi_s1h=pd.read_csv(f'{base_path}{upi_s1h_annotation}')

    def _image_path(self, index):
        if index < MIN_IMAGE or index > MAX_IMAGE:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {MAX_IMAGE}]')
        return self._path_to_image + '/' + self._release['annolist'][index][1]

    def _num_annotations(self, index):
        if index < MIN_IMAGE or index > MAX_IMAGE:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {MAX_IMAGE}]')

        annolist_index=self._release['annolist'][index][0]
        annorect=self._df_annorect[self._df_annorect[0]==annolist_index]

        return annorect.shape[0]

    def _num_up_s1h_annotations(self, index):
        if index < MIN_IMAGE or index > MAX_IMAGE:
            raise Exception(f'Invalid image index: {index}. Must be in range [{MIN_IMAGE}, {MAX_IMAGE}]')

        return len(self._upi_s1h[self._upi_s1h['mpii_id']==index])

    def _unified_annotations(self, index):
        """
        Return list of tuples with (MPII annotation, UP-S1H annotation)
        Needs to compare bboxes to decide which is the best match for each pair
        None will be in place of any missing annotation
        Length of list is exactly number of MPII annotations
        """
        num_annotations=self._num_annotations(index)
        num_s1h_annotations=self._num_up_s1h_annotations(index)

        unified=[]
        unified_dbg_idxs=[]

        image_path = self._image_path(index)
        if image_path[-13:] in ['040348287.jpg', '013401523.jpg', '002878268.jpg' ]:
            # These files are missing, so exit early
            return unified

        # Read width, height of image
        img=Image.open(self._base_path+self._image_path(index))
        width, height = img.size

        # Skip some images that have no MPII annotations
        if num_annotations>0:
            # Read MPII annotations into numpy array
            #  - Each row contains info about a single person in the image
            #  - Format: Index, Scale, Obj Pos X/Y (2), Head Bbox (4), Joints (x,y,v)*16
            annolist_index=self._release['annolist'][index][0]
            mpii_annotations=self._df_annorect[self._df_annorect[0]==annolist_index].to_numpy()

            # Check each MPII annotation and create list of valid ones
            # Valid is:
            #   - There are some valid joints
            #   - Can create a bounding box from joints
            # At the end, we will have a list of joints and bounding boxes including the joints
            mpii_ann_list = []
            for i, ann in enumerate(mpii_annotations):
                jointsx=ann[8::3]
                jointsy=ann[9::3]

                valj=mpii_valid_joints(jointsx, jointsy, (width, height))
                if valj.any():
                    mbb=mpii_bbox(jointsx, jointsy, (width, height))
                    if mbb == None:
                        print(f'WARNING: Could not create mpii_bbox! {index}/{i}')
                        mpii_ann_list.append(None)
                    else:
                        mpii_ann_list.append((jointsx, jointsy, valj, mbb))
                else:
                    #print(f'No valid joints')
                    mpii_ann_list.append(None)

            # Read S1H annotations into numpy array
            #  - Each row contains info about a single person in the image
            #  - Format: MPII ID, MPII Filename, S1H Filename, bbox (4)
            s1h_annotations=self._upi_s1h[self._upi_s1h['mpii_id']==index].to_numpy()

            # Check each S1H annotation and create list of valid ones
            # Valid is:
            #   - Bounding box fits in the image boundaries
            # At the end, we will have a list of valid bounding boxes
            s1h_ann_list = []
            for i, ann in enumerate(s1h_annotations):
                sbb=s1h_bbox(ann[3],ann[4],ann[5],ann[6], (width, height))
                if sbb == None:
                    print(f'Could not create S1H bbox! {index}/{i}')
                    s1h_ann_list.append(None)
                else:
                    # Just adding a 1 to make the code for MPII list and this symmetric in terms of how deep to index
                    s1h_ann_list.append((sbb,1))

            # Now we need to check each combination of MPII/UP-S1H bbox and find best
            # Create arrays to hold scores
            jscore=np.zeros((num_annotations, num_s1h_annotations))
            bbscore=np.zeros((num_annotations, num_s1h_annotations))
            assert(num_annotations==len(mpii_ann_list))
            assert(num_s1h_annotations==len(s1h_ann_list))

            for midx, m in enumerate(mpii_ann_list):
                if m is not None:
                    jointsx=m[0]
                    jointsy=m[1]
                    jointsv=m[2]
                    mbb=m[3]

                    for sidx, s in enumerate(s1h_ann_list):
                        if s is not None:
                            sbb=s[0]

                            # MPII Joints vs. S1H Bbox
                            jscore[midx,sidx]=eval_joints_bbox(jointsx, jointsy, jointsv, sbb)
                            # MPII Joints Minimal Bbox vs. S1H Bbox
                            bbscore[midx,sidx]=eval_bbox_iou(mbb, sbb) 
                        
            # Check the scores and pick best MPII match for each S1H annotation
            best_jscores=np.argmax(jscore, axis=0)
            best_bbscores=np.argmax(bbscore, axis=0)
            if( not np.array_equal(best_jscores, best_bbscores) ):
                print(f"WARNING! J/BB scores not equal!' Index: {index}")
                print('J Scores: ' + str(best_jscores))
                print(jscore)
                print('BB Scores: ' + str(best_bbscores))
                print(bbscore)

            for i, mpi_ann in enumerate(mpii_annotations):
                best_s1h=None
                best_s1h_idx=None
                for j, s1h_ann in enumerate(s1h_annotations):
                    if best_bbscores[j] == i:
                        best_s1h = s1h_ann
                        best_s1h_idx = j
                        break
                unified.append((mpi_ann,best_s1h))
                unified_dbg_idxs.append((i,best_s1h_idx))

        #print(f'DBG ({index}):')
        #print(unified_dbg_idxs)
        return unified
        
    def disp_image(self, index):
        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._base_path+self._image_path(index))
        img=img/255
        plt.imshow(img)
        plt.show()

    def disp_annotations(self, index):
        # TODO: Needs update to work with more than one image and/or display UPi-S1H annotations

        # Read in image and normalize. These are jpeg's, so need to be divided by 255 to
        # get values in range [0, 1]
        img=matplotlib.image.imread(self._base_path+self._image_path(index))
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
                the_annotations=self._unified_annotations(i)
                self._format_annotations(result, i, the_annotations, gid)
        elif which_set == 'val':
            # No validation in this set
            pass
        elif which_set == 'test':
            # This returns a tuple, so extract the part we care about
            test_idxs=np.where(self._release['img_train']==0)[0]
            for i in test_idxs:
                the_annotations=self._unified_annotations(i)
                self._format_annotations(result, i, the_annotations, gid)
        elif which_set == 'toy':
            # Grab the first 10 training instances
            # This returns a tuple, so extract the part we care about
            train_idxs=np.where(self._release['img_train']==1)[0][:10]
            for i in train_idxs:
                the_annotations=self._unified_annotations(i)
                self._format_annotations(result, i, the_annotations, gid)
        return result

    def _format_annotations(self, results, index, unified_anns, gid):
        image_path = self._image_path(index)

        for uann in unified_anns:
            # Read width, height of image
            img=Image.open(self._base_path+self._image_path(index))
            width, height = img.size

            annotation = {}
            annotation['ID'] = gid.next()
            annotation['set'] = 'MPII'
            annotation['path'] = self._image_path(index)

            # Minimal bounding box containing joints
            jointsx=uann[0][8::3]
            jointsy=uann[0][9::3]
            valj=mpii_valid_joints(jointsx, jointsy, (width, height))
            if valj.any():
                mbb=mpii_bbox(jointsx, jointsy, (width, height))
                # Convert to (x0,y0,x1,y1)
                mbb=(mbb[1],mbb[0],mbb[3],mbb[2])
                area=(mbb[2]-mbb[0])*(mbb[3]-mbb[1])
                if area > 0:
                    annotation['bbox'] = mbb
                else:
                    print(f"Zero area! {index} {area}")
                    print(mbb)
                    print(annotation['ID'])
            #annotation['bbox_x'] = 0
            #annotation['bbox_y'] = 0
            #annotation['bbox_h'] = height
            #annotation['bbox_w'] = width

            # Iterate over values from scale up to head_y2, but drop the first ID because we don't care about it
            # Skipping the last value since it's just a placeholder where joints start
            for i,k in enumerate(list(annorect_map.keys())[:-1]):
                if k == 'id':
                    continue
                annotation[k] = uann[0][i]

            # Now deal with the joints
            iidx=8  # Initialize to index of first joint
            for j in range(len(mpii_joints)):
                for xyv in ['x', 'y', 'v' ]:
                    annotation[f'{xyv}{j}'] = uann[0][iidx]
                    iidx += 1

            # Now, add silhouette info if available
            if uann[1] is not None:
                silhouette_filename = uann[1][2]  # In numpy format, image_name is index 2
                silhouette_filename = silhouette_filename[:5]+'_segmentation_full.png'
                annotation['silhouette'] = self._upi_s1h_img+'/'+silhouette_filename

            # Note: UPi-S1H provides the following formats if alternates are needed
            #00001.png - just person in bbox area
            #00001_full.png - full image
            #00001_segmentation.png - person silhouette in bbox area
            #00001_segmentation_full.png - person silhouette in full image

            results.append(annotation)

        return
