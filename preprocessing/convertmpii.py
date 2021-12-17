#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import sys
import pickle

def process_values(k, v):
    rval = v

    if k == 'version':
        pass
    elif k == 'act':
        rval=process_act(v)
    elif k == 'annolist':
        rval=process_annolist(v)
    elif k == 'annorect':
        rval=process_annorect(v)
    elif k == 'img_train':
        rval=process_img_train(v)
    elif k == 'single_person':
        rval=process_single_person(v)
    elif k == 'video_list':
        rval=process_videolist(v)
    else:
        sys.stderr.write(f'Error: Unrecognized key {k}')

    return rval

def process_act(path):
    act_array=np.genfromtxt(path, delimiter=';', dtype=None, encoding='utf8', skip_header=1)
    return act_array

def process_annolist(path):
    al_array=np.genfromtxt(path, delimiter=',', dtype=None, encoding='utf8', skip_header=1,autostrip=True)
    return al_array

def process_annorect(path):
    ar_array=np.genfromtxt(path, delimiter=',', skip_header=1)
    return ar_array

def process_videolist(path):
    with open(path) as invfile:
        videolines=invfile.readlines()
        # skip 1 line for header
        videolines=[line.strip() for line in videolines[1:]]
        invfile.close()
    return videolines

def process_img_train(path):
    img_train_array=np.genfromtxt(path, delimiter=',', dtype=int, skip_header=1)
    return img_train_array


def process_single_person(path):
    single_person_array=np.genfromtxt(path, delimiter=',', dtype=int, skip_header=1)
    # This ragged list is more like the original data but if it's harder to work with,
    # just return single_person_array instead
    ragged_single_person = [x[x!=-1] for x in single_person_array]
    return ragged_single_person


# Command-line arguments
ap=argparse.ArgumentParser(description='Utility to convert flattened MPII data files to a python-readable format. The script convertmpii.m produces the required set of files. The file RELEASE.txt, provided as a command-line argument to this script, contains the full list of files that must be processed.')
ap.add_argument("-r", "--release", required=True, help="Path to RELEASE.txt")
args=vars(ap.parse_args())

release_txt = args['release']
if release_txt[-11:] != 'RELEASE.txt':
    sys.stderr.write('Error: Provided file is not RELEASE.txt\n')
    exit(-1)
release_path = release_txt[:-11]

with open(release_txt) as infile:
    lines = infile.readlines()
    lines = [line.rstrip() for line in lines]

RELEASE={}

for line in lines:
    splitline=line.split(':')
    splitline[0]=splitline[0].strip()
    splitline[1]=splitline[1].strip()
    if splitline[1][-4:] == '.csv':
        splitline[1] = release_path + splitline[1]
    RELEASE[splitline[0]]=process_values(splitline[0],splitline[1])

for k in RELEASE:
    print(f'k: {k}')
    if k == 'annolist' or k == 'img_train' or k == 'annorect':
        print('\t' + str(RELEASE[k].shape))
        print('\t' + str(RELEASE[k]))
    elif k == 'video_list' or k == 'single_person':
        print('\t' + str(len(RELEASE[k])))
        print( '\t' + str(RELEASE[k][0:3]) + ' ... ' + str(RELEASE[k][-3:]) )
    elif k == 'act':
        print('\t' + str(RELEASE[k].shape))
        print( '\t' + str(RELEASE[k][0:3]) + ' ... ' + str(RELEASE[k][14:16]) +  ' ... ' + str(RELEASE[k][-4:]) )
    else:
        print('\t' + str(RELEASE[k]))

    print('')

#print(RELEASE)

# Save as pickle
with open('mpii-RELEASE.pickle', 'wb') as of:
    pickle.dump(RELEASE,of)
