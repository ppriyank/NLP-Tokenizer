import os
import time
import sys
import re
from subprocess import call
import numpy as np







FASTTEXT_EXEC_PATH = "fast text/sent2vec-master/./fasttext"
MODEL_TORONTOBOOKS_BIGRAMS = "fast text/torontobooks_bigrams.bin"

# for this_iter in range(300):
#  fname= 'sentence_dataset/train_%d.txt'%(this_iter)
#  print(fname)
#  output_file = 'fast_text_sentence_vectors/train_%d.txt'%(this_iter)
#  os.system("fast\ text/sent2vec-master/./fasttext print-sentence-vectors fast\ text/torontobooks_bigrams.bin < " + fname + " > "+ output_file) 	

   

for this_iter in range(300): 
    fname1= '../../../pathak/sentence_dataset/train_%d.txt'%(this_iter)
    fname2= '../../../Abstract/train_%d.txt'%(this_iter)
    print(fname1)
    output_file1 = "../../../pathak/fasttext/Doc/train_%d.npy"%(this_iter)
    output_file2 = "../../../pathak/fasttext/Abs/train_%d.npy"%(this_iter)
    if (os.path.isfile(fname1)):
    	os.system("./fasttext print-sentence-vectors ../torontobooks_bigrams.bin < " + fname1 + " > "+ output_file1) 	
    if (os.path.isfile(fname2)):
    	os.system("./fasttext print-sentence-vectors ../torontobooks_bigrams.bin < " + fname2 + " > "+ output_file2) 	

