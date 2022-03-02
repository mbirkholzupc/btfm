
import pickle
import numpy as np

TSEQS=[1,2,3,4,5,6]

def mi3_tst_idx(seq):
    return f'/TS{seq}'

frames={}

for seq in TSEQS:
    filename=f'TS{seq}.txt'
    frames[mi3_tst_idx(seq)]=np.loadtxt(filename,dtype=int)

outfile=open('mi3_pp_test_frames.pkl', 'wb')
pickle.dump(frames,outfile)
outfile.close()
