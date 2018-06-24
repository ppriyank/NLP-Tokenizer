


import skipthoughts
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

import numpy as np 
import os 


def load(fname , s=True):
    f = open(fname,'r')
    data = []
    for line in f.readlines():
    	if s :
    		data.append(line.replace('\n','').split(' '))
    	else:
    		data.append(line.replace('\n',''))
    f.close()
    return data


for this_iter in range(300): 
    fname1= '../../pathak/sentence_dataset/train_%d.txt'%(this_iter)
    fname2= '../../Abstract/train_%d.txt'%(this_iter)
    # fname= '~/pathak/sentence_dataset/train_%d.txt'%(this_iter)
    # this_iter = 56
    print(fname1)
    if (os.path.isfile(fname1)) :    
        A= load(fname1 , False)
        vectors = encoder.encode(A)
        # print (np.shape(vectors ) , np.shape(A ))
        np.save("../../pathak/skip_thought_vectors/Doc/train_%d.npy"%(this_iter),  vectors)
    # print(fname2)
    if (os.path.isfile(fname2)) :    
        A= load(fname2 , False)
        vectors = encoder.encode(A)
        # print (np.shape(vectors ) , np.shape(A ))
        np.save("../../pathak/skip_thought_vectors/Abs/train_%d.npy"%(this_iter),  vectors)
