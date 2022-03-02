import pickle

infile=open('good_3dpw_annotations.pkl','rb')
annotations=pickle.load(infile)
infile.close()
