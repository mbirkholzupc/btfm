# BTFM Preprocessing
This directory holds the preprocessing scripts and files for the BTFM dataset, including data preprocessed with MATLAB since MATLAB licensing is too restrictive to be installed everywhere. The files preprocessed with MATLAB are in the data/ directory. There is also a python script to convert the MPII csv files to a pickle file for easier python consumption. Note that this script is run by the initial setup script so there is no need to run any of these manually.

In order to preprocess with MATLAB, the following scripts simply need the PATH variable updated to point to the input file:
 * convertlsp.m
 * convertlspet.m
 * convertmpii.m

A python script is provided to convert the MPII csv files to a pickle:
 * convertmpii.py

