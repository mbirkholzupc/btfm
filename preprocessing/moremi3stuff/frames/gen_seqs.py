
import pickle
import numpy as np

from pathlib import Path

SUBJS=[1,2,3,4,5,6,7,8]
SEQS=[1,2]

TSEQS=[1,2,3,4,5,6]

frames={}

for subj in SUBJS:
    for seq in SEQS:
        file_list=sorted(list(Path(f'../S{subj}/Seq{seq}/img').glob('img_0_*')))
        frames=[x.name[-10:-4] for x in file_list]
        filename=f'S{subj}_Seq{seq}.txt'
        outfile=open(filename, 'w')
        for f in frames:
            print(f'{f}', file=outfile)
        outfile.close()

for seq in TSEQS:
        file_list=sorted(list(Path(f'../../mpi_inf_3dhp/mpi_inf_3dhp/download/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/TS{seq}/imageSequence').glob('img_*')))
        frames=[x.name[-10:-4] for x in file_list]
        filename=f'TS{seq}.txt'
        outfile=open(filename, 'w')
        for f in frames:
            print(f'{f}', file=outfile)
            # These are the last valid frames in two sequences
            if (seq==3) and (f=='005838'):
                break
            elif (seq==4) and (f=='006007'):
                break
        outfile.close()

