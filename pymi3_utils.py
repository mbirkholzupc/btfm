#!/bin/env python
# Original source from mpi_inf_3dhp/util/mpii_get_sequence_info.m

from collections import namedtuple

MpiiSeqInfo = namedtuple('MpiiSeqInfo', 'bg_augmentable ub_augmentable lb_augmentable chair_augmentable fps num_frames')

def mpii_get_sequence_info( subj, seq ):
    ub_augmentable = False
    lb_augmentable = False
    bg_augmentable = False
    chair_augmentable = False
    fps = 25

    assert(subj in [1, 2, 3, 4, 5, 6, 7, 8])
    assert(seq in [1, 2])

    if subj == 1:
        if seq == 1:
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 6416
        elif seq == 2:
            ub_augmentable = True  # The LB masks are bad, so skip putting textures there and in the BG
            chair_augmentable = True
            num_frames = 12430
            fps = 50
    elif subj == 2:
        if seq == 1:
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 6502
        elif seq == 2:
            bg_augmentable = True
            chair_augmentable = True
            ub_augmentable = True
            lb_augmentable = True
            num_frames = 6081
    elif subj == 3:
        fps = 50
        if seq == 1:
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 12488
        elif seq == 2:
            bg_augmentable = True
            chair_augmentable = True
            ub_augmentable = True
            lb_augmentable = True
            num_frames = 12283
    elif subj == 4:
        if seq == 1:
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 6171
        elif seq == 2:
            chair_augmentable = True # The LB masks are bad, so ski putting textures there and in the BG
            ub_augmentable = True
            num_frames = 6675
    elif subj == 5:
        fps = 50
        if seq == 1:
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 12820
        elif seq == 2:
            chair_augmentable = True
            ub_augmentable = True
            bg_augmentable = True
            lb_augmentable = True
            num_frames = 12312
    elif subj == 6:
        if seq == 1:
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 6188
        elif seq == 2:
            ub_augmentable = True
            lb_augmentable = True
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 6145
    elif subj == 7:
        if seq == 1:
            bg_augmentable = True
            chair_augmentable = True
            ub_augmentable = True
            lb_augmentable = True
            num_frames = 6239
        elif seq == 2:
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 6320
    elif subj == 8:
        if seq == 1:
            bg_augmentable = True
            chair_augmentable = True
            ub_augmentable = True
            lb_augmentable = True
            num_frames = 6468
        elif seq == 2:
            bg_augmentable = True
            chair_augmentable = True
            num_frames = 6054

    seq_info = MpiiSeqInfo(bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable, fps, num_frames)

    return seq_info
