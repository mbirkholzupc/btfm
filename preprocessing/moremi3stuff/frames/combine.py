
import pickle
import numpy as np

SUBJS=[1,2,3,4,5,6,7,8]
SEQS=[1,2]

def mi3_idx(subj,seq):
    return f'/S{subj}/Seq{seq}'

frames={}

for subj in SUBJS:
    for seq in SEQS:
        filename=f'S{subj}_Seq{seq}.txt'
        frames[mi3_idx(subj,seq)]=np.loadtxt(filename,dtype=int)

outfile=open('mi3_pp_frames.pkl', 'wb')
pickle.dump(frames,outfile)
outfile.close()
