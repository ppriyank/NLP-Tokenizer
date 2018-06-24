
import numpy as np 
import pickle
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


from word_pattern import *

import re
import string
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize


import numpy as np 
import pickle
import json

from scipy import sparse, io

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def tokenizer(text):
        token = ""
        tokenized_sentence = []
        for m in word_compile.finditer(text):
            if  len(m.group()) == 0 :
                continue
            else :
                token = m.group()
                token = re.sub(r'( )+$',"", token) 
                token = re.sub(r',|(\.$)',"", token)
                token = re.sub(r'\'[a-z]+',"", token)
                token = re.sub(r'/|\*|^'," ", token)
                if token == "" or token== "re" or token== "Re" or  token == "-" or token == "cnn" or token == "CNN":
                    continue 
                elif token != "" and token[-1] == " " : 
                      token = token[:-1]      
                elif token != "" and token[-1] == "," or token[-1] == "{" or token[-1] == "-" or token[-1] == "}" or token[-1] == ")" :
                        token = token[:-1] 
                elif token != "" and token[0] == "(" or token[0] == "{" or token[0] == "-":
                        token = token[1:]
                elif token != "" and len(re.findall("-", token))>0 : 
                            temp = token.split("-")
                            for i in range(len(temp)):
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                elif token != "" and len(re.findall(" ", token))>0 : 
                            temp = token.split(" ")
                            for i in range(len(temp)):
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                if token != "" : 
                    tokenized_sentence.append(token.lower())
        # tokenized_sentence
        tokenized_headline2_temp = []         
        for tokenized_word in tokenized_sentence :    
            if tokenized_word == "":
                continue
            tokenized_word = re.sub(r"[^A-Za-z0-9(),@!?\'\`]", "", tokenized_word)     
            tokenized_word = re.sub(r"\'s", " \'s", tokenized_word) 
            string = tokenized_word
            string = re.sub(r"cnn", "", string)
            string = re.sub(r"\'ve", "", string) 
            string = re.sub(r"n\'t", "", string) 
            string = re.sub(r"\'re", "", string) 
            string = re.sub(r"\'d", "", string) 
            string = re.sub(r"\'ll", "", string) 
            string = re.sub(r",", "", string) 
            string = re.sub(r"!", "", string) 
            string = re.sub(r"\(", "", string) 
            string = re.sub(r"\)", "", string) 
            string = re.sub(r"\?", "", string) 
            string = re.sub(r"\s{2,}", "", string)
            string = re.sub(r"<[^>]*>", "", string)   
            if string == "" or string == 'p': 
                continue
            tokenized_headline2_temp.append(string)
        return  tokenized_headline2_temp 




def grab_body_headline(article):
    found_headline = re.findall("(?<=Subject:)(.*)", article)
    found_body = re.findall("(?<=Lines:)(?s)(.*)", article)
    # For 58 of the 18,846 articles in the data, no headline and/or body was found.  
    if found_headline and found_body: 
        headline = found_headline[0]
        body = found_body[0]
        return (body, headline)
    else: 
        # Return (None, None) to allow continuation of the pipline, and filter later.
        return (None, None)


articles = fetch_20newsgroups(subset='all').data
bodies_n_headlines = [grab_body_headline(article) for article in articles]




bodies = []
headlines = []
Vocabulary = []
tokenized_headline = []
tokenized_body = []
count = 0 



for body, headline in bodies_n_headlines: 
    if body and headline: 
        count = count + 1
        bodies.append(body)
        headlines.append(headline)
        token = ""
        tokenized_sentence = []
        for m in word_compile.finditer(headline):
            if  len(m.group()) == 0 :
                continue
            else :
                token = m.group()
                token = re.sub(r'( )+$',"", token) 
                token = re.sub(r',|(\.$)',"", token)
                token = re.sub(r'\'[a-z]+',"", token)
                token = re.sub(r'/|\*|^'," ", token)
                if token == "" or token== "re" or token== "Re" or  token == "-":
                    continue 
                elif token != "" and token[-1] == " " : 
                      token = token[:-1]      
                elif token != "" and token[-1] == "," or token[-1] == "{" or token[-1] == "-" or token[-1] == "}" or token[-1] == ")" :
                        token = token[:-1] 
                elif token != "" and token[0] == "(" or token[0] == "{" or token[0] == "-":
                        token = token[1:]
                # elif token != "" and len(re.findall("^[A-Z]", token)) <= 0 :
                #         if len(re.findall("-", token))>0 :
                #             temp = token.split("-")
                #             for i in range(len(temp)):
                #                 if temp[i] not in Vocabulary : 
                #                     Vocabulary.append(temp[i].lower())
                #                 tokenized_sentence.append(temp[i].lower()) 
                #             continue
                elif token != "" and len(re.findall("-", token))>0 : 
                            temp = token.split("-")
                            for i in range(len(temp)):
                                # if temp[i] not in Vocabulary : 
                                #     Vocabulary.append(temp[i].lower())
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                elif token != "" and len(re.findall(" ", token))>0 : 
                            temp = token.split(" ")
                            for i in range(len(temp)):
                                # if temp[i] not in Vocabulary : 
                                #     Vocabulary.append(temp[i].lower())
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                if token != "" : 
                    # print(token)
                    tokenized_sentence.append(token.lower())
                    # if token.lower() not in Vocabulary:
                    #             Vocabulary.append(token.lower()) 
        # print(count , headline ,tokenized_sentence)
        tokenized_headline = tokenized_headline + [tokenized_sentence]
        tokenized_sentence = [] 
        token = ""
        for m in word_compile.finditer(body):
            if  len(m.group()) == 0 :
                continue
            else :
                # if count == 18786:
                #     break
                token = m.group()
                # print(token)
                token = re.sub(r'( )+$',"", token) 
                token = re.sub(r',|(\.$)',"", token)
                token = re.sub(r'\'[a-z]+',"", token)
                token = re.sub(r'/'," ", token)
                if token == "" or token== "re" or token== "Re" or token == "-" or token == "." or token == "!" or token == "?":
                    continue 
                elif token != "" and token[-1] == " " : 
                      token = token[:-1]      
                elif token != "" and token[-1] == "," or token[-1] == "{" or token[-1] == "-" or token[-1] == "}" or token[-1] == ")" :
                        token = token[:-1] 
                elif token != "" and token[0] == "(" or token[0] == "{" or token[0] == "-":
                        token = token[1:]
                # elif token != "" and len(re.findall("^[A-Z]", token)) <= 0 :
                #         if len(re.findall("-", token))>0 :
                #             temp = token.split("-")
                #             for i in range(len(temp)):
                #                 if temp[i] not in Vocabulary : 
                #                     Vocabulary.append(temp[i].lower())
                #                 tokenized_sentence.append(temp[i].lower()) 
                #             continue
                elif token != "" and len(re.findall("-", token))>0 : 
                            temp = token.split("-")
                            for i in range(len(temp)):
                                # if temp[i] not in Vocabulary : 
                                #     Vocabulary.append(temp[i].lower())
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                elif token != "" and len(re.findall(" ", token))>0 : 
                            temp = token.split(" ")
                            for i in range(len(temp)):
                                # if temp[i] not in Vocabulary : 
                                #     Vocabulary.append(temp[i].lower())
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                if token != "" : 
                    # print(token)
                    tokenized_sentence.append(token.lower())
                    # if token.lower() not in Vocabulary:
                    #             Vocabulary.append(token.lower()) 
        # print(count , headline ,tokenized_sentence)
        tokenized_body = tokenized_body + [tokenized_sentence]
        
len(tokenized_body)
len(tokenized_headline)

tokenized_headline2 = [] 
tokenized_body2 = []
for i in range (len(tokenized_headline)) :
    headline = tokenized_headline[i]
    body =  tokenized_body[i]
    tokenized_headline2_temp = []
    tokenized_body2_temp = []
    for tokenized_word in headline :
        if tokenized_word == "":
            continue
        tokenized_word = re.sub(r"[^A-Za-z0-9(),@!?\'\`]", "", tokenized_word)     
        tokenized_word = re.sub(r"\'s", " \'s", tokenized_word) 
        string = tokenized_word
        string = re.sub(r"\'ve", "", string) 
        string = re.sub(r"n\'t", "", string) 
        string = re.sub(r"\'re", "", string) 
        string = re.sub(r"\'d", "", string) 
        string = re.sub(r"\'ll", "", string) 
        string = re.sub(r",", "", string) 
        string = re.sub(r"!", "", string) 
        string = re.sub(r"\(", "", string) 
        string = re.sub(r"\)", "", string) 
        string = re.sub(r"\?", "", string) 
        string = re.sub(r"\s{2,}", "", string)    
        if string == "" :
            continue
        tokenized_headline2_temp.append(string)
    if len(tokenized_headline2_temp) < 3 :
        continue 
    for tokenized_word in body :
        if tokenized_word == "":
            continue
        tokenized_word = re.sub(r"[^A-Za-z0-9(),@!?\'\`]", "", tokenized_word)     
        tokenized_word = re.sub(r"\'s", "\'s", tokenized_word) 
        string = tokenized_word
        string = re.sub(r"\'ve", "", string) 
        string = re.sub(r"n\'t", "", string) 
        string = re.sub(r"\'re", "", string) 
        string = re.sub(r"\'d", "", string) 
        string = re.sub(r"\'ll", "", string) 
        string = re.sub(r",", "", string) 
        string = re.sub(r"!", "", string) 
        string = re.sub(r"\(", "", string) 
        string = re.sub(r"\)", "", string) 
        string = re.sub(r"\?", "", string) 
        string = re.sub(r"\s{2,}", "", string)  
        if string == "" :
            continue  
        tokenized_body2_temp.append(string)
    if len(tokenized_body2_temp) > 2769 :
        continue 
    tokenized_body2.append(tokenized_body2_temp) 
    tokenized_headline2.append(tokenized_headline2_temp)


len(tokenized_body2)
len(tokenized_headline2)


save_obj(tokenized_body2, "20_news_tokenized_body")
save_obj(tokenized_headline2, "20_news_tokenized_headline")


root_folder_cnn = "cnn_data/" 
f = open(root_folder_cnn+'cnn_summaries.txt','r')
stories = f.read().strip().split('\n')
i = 0
for story in stories:
    f1 = open(root_folder_cnn+'summaries/summary_'+str(i)+'.txt','w')
    f1.write(story)
    f1.close()
    i += 1




f = open(root_folder_cnn+'cnn_stories.txt','r')
stories = f.read().strip().split('\n')
i = 0
for story in stories:
    f1 = open(root_folder_cnn+'stories/story_'+str(i)+'.txt','w')
    f1.write(story)
    f1.close()
    i += 1


bodies = []
headlines = []
Vocabulary = []
tokenized_headline = []
tokenized_body = []
count = 0 



for i in range(29842):
    f1 = open(root_folder_cnn+'stories/story_'+str(i)+'.txt','r')
    story = f1.read()
    f2 = open(root_folder_cnn+'summaries/summary_'+str(i)+'.txt','r')
    summary = f2.read()
    if body and headline: 
        count = count + 1
        token = ""
        tokenized_sentence = []
        for m in word_compile.finditer(summary):
            if  len(m.group()) == 0 :
                continue
            else :
                token = m.group()
                token = re.sub(r'( )+$',"", token) 
                token = re.sub(r',|(\.$)',"", token)
                token = re.sub(r'\'[a-z]+',"", token)
                token = re.sub(r'/|\*|^'," ", token)
                if token == "" or token== "re" or token== "Re" or  token == "-" or token == "cnn" or token == "CNN":
                    continue 
                elif token != "" and token[-1] == " " : 
                      token = token[:-1]      
                elif token != "" and token[-1] == "," or token[-1] == "{" or token[-1] == "-" or token[-1] == "}" or token[-1] == ")" :
                        token = token[:-1] 
                elif token != "" and token[0] == "(" or token[0] == "{" or token[0] == "-":
                        token = token[1:]
                elif token != "" and len(re.findall("-", token))>0 : 
                            temp = token.split("-")
                            for i in range(len(temp)):
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                elif token != "" and len(re.findall(" ", token))>0 : 
                            temp = token.split(" ")
                            for i in range(len(temp)):
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                if token != "" : 
                    tokenized_sentence.append(token.lower())
        tokenized_headline = tokenized_headline + [tokenized_sentence]
        tokenized_sentence = [] 
        token = ""
        for m in word_compile.finditer(story):
            if  len(m.group()) == 0 :
                continue
            else :
                token = m.group()
                token = re.sub(r'( )+$',"", token) 
                token = re.sub(r',|(\.$)',"", token)
                token = re.sub(r'\'[a-z]+',"", token)
                token = re.sub(r'/'," ", token)
                if token == "" or token== "re" or token== "Re" or token == "-" or token == "." or token == "!" or token == "?":
                    continue 
                elif token != "" and token[-1] == " " : 
                      token = token[:-1]      
                elif token != "" and token[-1] == "," or token[-1] == "{" or token[-1] == "-" or token[-1] == "}" or token[-1] == ")" :
                        token = token[:-1] 
                elif token != "" and token[0] == "(" or token[0] == "{" or token[0] == "-":
                        token = token[1:]
                elif token != "" and len(re.findall("-", token))>0 : 
                            temp = token.split("-")
                            for i in range(len(temp)):
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                elif token != "" and len(re.findall(" ", token))>0 : 
                            temp = token.split(" ")
                            for i in range(len(temp)):
                                tokenized_sentence.append(temp[i].lower()) 
                            continue
                if token != "" : 
                    tokenized_sentence.append(token.lower())
        tokenized_body = tokenized_body + [tokenized_sentence]




tokenized_headline2 = [] 
tokenized_body2 = []
for i in range (len(tokenized_headline)) :
    headline = tokenized_headline[i]
    body =  tokenized_body[i]
    tokenized_headline2_temp = []
    tokenized_body2_temp = []
    for tokenized_word in headline :
        if tokenized_word == "":
            continue
        tokenized_word = re.sub(r"[^A-Za-z0-9(),@!?\'\`]", "", tokenized_word)     
        tokenized_word = re.sub(r"\'s", " \'s", tokenized_word) 
        string = tokenized_word
        string = re.sub(r"cnn", "", string)
        string = re.sub(r"\'ve", "", string) 
        string = re.sub(r"n\'t", "", string) 
        string = re.sub(r"\'re", "", string) 
        string = re.sub(r"\'d", "", string) 
        string = re.sub(r"\'ll", "", string) 
        string = re.sub(r",", "", string) 
        string = re.sub(r"!", "", string) 
        string = re.sub(r"\(", "", string) 
        string = re.sub(r"\)", "", string) 
        string = re.sub(r"\?", "", string) 
        string = re.sub(r"\s{2,}", "", string)    
        if string == "": 
            continue
        tokenized_headline2_temp.append(string)
    if len(tokenized_headline2_temp) < 3 :
        continue 
    for tokenized_word in body :
        if tokenized_word == "":
            continue
        tokenized_word = re.sub(r"[^A-Za-z0-9(),@!?\'\`]", "", tokenized_word)     
        tokenized_word = re.sub(r"\'s", "\'s", tokenized_word) 
        string = tokenized_word
        string = re.sub(r"cnn", "", string) 
        string = re.sub(r"\'ve", "", string) 
        string = re.sub(r"n\'t", "", string) 
        string = re.sub(r"\'re", "", string) 
        string = re.sub(r"\'d", "", string) 
        string = re.sub(r"\'ll", "", string) 
        string = re.sub(r",", "", string) 
        string = re.sub(r"!", "", string) 
        string = re.sub(r"\(", "", string) 
        string = re.sub(r"\)", "", string) 
        string = re.sub(r"\?", "", string) 
        string = re.sub(r"\s{2,}", "", string)
        if string == "": 
            continue    
        tokenized_body2_temp.append(string)
    # if len(tokenized_body2_temp) > 2769 :
    #     continue 
    tokenized_body2.append(tokenized_body2_temp) 
    tokenized_headline2.append(tokenized_headline2_temp)



len(tokenized_body2)
len(tokenized_headline2)

save_obj(tokenized_body2, "CNN_tokenized_body")
save_obj(tokenized_headline2, "CNN_tokenized_headline")




# A = load_obj("CNN_tokenized_body")
# B = load_obj("CNN_tokenized_headline")


import os
import re 


DUC  = "Processed/"

for SET in os.listdir(DUC):
    f1 = DUC  + SET + "/" + "TOPICS/" 
    if SET == "2004":
        continue
    for topic in os.listdir(f1):
        f2 = f1 + topic + "/"
        if not os.path.isdir(f2):
            continue 
        # f3 = f2 + "documents/"
        for type_ in os.listdir(f2):
            f3 = f2 + type_ + "/"
            if not os.path.isdir(f3):
                continue 
            for file_name in os.listdir(f3):
                f4 = f3 + file_name 
                if f4[-4:] == ".pkl":
                    continue
                file = open(f4, "r") 
                text= file.read()
                name = re.sub("\.txt", "", file_name)
                tokenized = tokenizer(text); 
                save_obj(tokenized, f3 + name )
        


DUC  = "Processed/"
import os 

f1 = DUC  + "2001" + "/" + "Train/"
for topic in os.listdir(f1):
    f2 = f1 + topic + "/"
    if not os.path.isdir(f2):
        continue 
    for type_ in os.listdir(f2):
        f3 = f2 + type_ + "/"
        if not os.path.isdir(f3):
            continue 
        for file_name in os.listdir(f3):
            f4 = f3 + file_name 
            if f4[-4:] == ".pkl":
                continue
            file = open(f4, "r") 
            text= file.read()
            name = re.sub("\.txt", "", file_name)
            tokenized = tokenizer(text); 
            save_obj(tokenized, f3 + name )
        

f1 = DUC  + "2001" + "/" + "Trial/"  

for topic in os.listdir(f1):
    f2 = f1 + topic + "/"
    if not os.path.isdir(f2):
        continue 
    for type_ in os.listdir(f2):
        f3 = f2 + type_ + "/"
        if not os.path.isdir(f3):
            continue 
        for file_name in os.listdir(f3):
            f4 = f3 + file_name 
            if f4[-4:] == ".pkl":
                continue
            file = open(f4, "r") 
            text= file.read()
            name = re.sub("\.txt", "", file_name)
            tokenized = tokenizer(text); 
            save_obj(tokenized, f3 + name )
        




vocab = defaultdict(float)

vocab = {}
max_len =  0



def build_vocab(pick, vocab):
    A = load_obj(pick)
    global max_len 
    for tokenized in A:
        slen = len(tokenized)
        print(slen)
        max_len = max(max_len,slen)
        for word in tokenized:
            vocab[word] = 1
    return vocab

def build_vocab2(pick, vocab):
    A = load_obj(pick)
    global max_len     
    slen = len(A)
    # print(slen)
    max_len = max(max_len,slen)
    for word in A:
        vocab[word] = 1
    return vocab


A = load_obj(pick)
pick = "CNN_tokenized_body"
pick = "20_news_tokenized_body"





vocab = build_vocab("CNN_tokenized_body", vocab)
vocab = build_vocab("20_news_tokenized_body", vocab)

max_len = 2768 

max_len = 0
vocab = build_vocab("CNN_tokenized_headline", vocab)
vocab = build_vocab("20_news_tokenized_headline", vocab)

max_len = 89 

len(vocab)

pick = "Processed/2003/TOPICS/d30040/documents/t_apw19981124.0256"


for SET in os.listdir(DUC):
    f1 = DUC  + SET + "/" + "TOPICS/" 
    if SET == "2004":
        continue
    for topic in os.listdir(f1):
        f2 = f1 + topic + "/"
        if not os.path.isdir(f2):
            continue 
        for type_ in os.listdir(f2):
            if type_ == "documents" :
                continue
            f3 = f2 + type_ + "/"
            if not os.path.isdir(f3):
                continue 
            for file_name in os.listdir(f3):
                f4 = f3 + file_name
                if f4[-4:] != ".pkl":
                    continue
                f4
                vocab = build_vocab2(f4[:-4], vocab)



DUC = "Processed/"
A = ["Train" , "Trial" ]
for a in A :
    f1 = DUC  + "2001" + "/" + a + "/"
    for topic in os.listdir(f1):
        f2 = f1 + topic + "/"
        if not os.path.isdir(f2):
            continue 
        for type_ in os.listdir(f2):
            f3 = f2 + type_ + "/"
            if not os.path.isdir(f3):
                continue 
            for file_name in os.listdir(f3):
                file_name
                f4 = f3 + file_name 
                if f4[-4:] != ".pkl":
                    continue
                vocab = build_vocab2(f4[:-4], vocab)
                    
            


documents
max_len = 4633

summary 
max_len = 333

save_obj(vocab, "vocab")



vocab = load_obj("vocab")
json.dump(vocab, open('DataVocab.txt', 'w'))


word2vec_file = "GoogleNews-vectors-negative300.bin"  


def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs


def get_W(word_vecs, word_vocab,k=300):
    vocab_size = len(word_vocab)
    print(vocab_size)
    word_idx_map = {}
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    temp = np.mean(W[0:i,:],axis=0)
    for word in word_vocab:
        if  word not in word_idx_map:
            W[i] = temp
            word_idx_map[word] = i
            i += 1
    return W, word_idx_map


w2v = load_bin_vec(word2vec_file,vocab)
wordVecs, word_idx_map = get_W(w2v,vocab)

# cPickle.dump([wordVecs, word_idx_map], open('wordVecs_wordIndex_dataset.p', "wb"))


#####################################################################################################
# padding 


stored_data_path = "priyank_vocab.p"
with open(stored_data_path, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    x = u.load()

    
w2v, word_idx_map = x[0], x[1]


max_l = 4633 
k=300
filter_h=5

def make_idx_data_cv(A , word_idx_map, max_l, k=300, filter_h=5):
    train = []
    count = 0 
    for stories in A :
        sent,temp = get_idx_from_sent(stories , word_idx_map, max_l, k, filter_h)   
        train.append(sent)
        count += temp
    print(type(train), len(train))
    train = np.array(train,dtype="int")
    # print (count)
    return train     


def get_idx_from_sent(sent, word_idx_map, max_l, k=300, filter_h=5):
    x = []
    count = 0
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    for word in sent:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            count += 1
    while len(x) < max_l+filter_h:
        x.append(0)
    return np.array(x), count



def padding(pick , max_size, word_idx_map ):
    A = load_obj(pick)
    print (len(A))
    train_matrix = make_idx_data_cv(A,  word_idx_map , max_size, k=300, filter_h=5)
    train = np.array(train_matrix,dtype="int")
    train_file_name = pick + '.npz'
    train_matrix = sparse.csr_matrix(train_matrix)
    save_sparse_csr(train_file_name,train_matrix)


def padding2(A,pick , max_size, word_idx_map ):
    # print (len(A))
    train_matrix = make_idx_data_cv(A,  word_idx_map , max_size, k=300, filter_h=5)
    train = np.array(train_matrix,dtype="int")
    train_file_name = pick + '.npz'
    train_matrix = sparse.csr_matrix(train_matrix)
    save_sparse_csr(train_file_name,train_matrix)

# A = load_sparse_csr("20_news_tokenized_body.npz")


padding("CNN_tokenized_body" , 4633, word_idx_map )
padding("20_news_tokenized_body" , 4633, word_idx_map )

padding("CNN_tokenized_headline" , 333, word_idx_map )
padding("20_news_tokenized_headline" , 333, word_idx_map )


train_matrix = load_sparse_csr('20_news_tokenized_body.npz')
A = train_matrix.toarray()


train_matrix = load_sparse_csr('documents.npz')
A = train_matrix.toarray()

i=0
A[i][np.where(A[i])]


DUC  = "Processed/"
import os 



for SET in os.listdir(DUC):
    f1 = DUC  + SET + "/" + "TOPICS/" 
    if SET == "2004":
        continue
    for topic in os.listdir(f1):
        f2 = f1 + topic + "/"
        if not os.path.isdir(f2):
            continue 
        for type_ in os.listdir(f2):
            if type_ != "documents" :
                continue
            f3 = f2 + type_ + "/"
            if not os.path.isdir(f3):
                continue 
            temp = []
            count = 0
            string = "" 
            for file_name in os.listdir(f3):
                f4 = f3 + file_name
                if f4[-4:] != ".pkl" :
                    continue
                # os.remove(f4)
                A = load_obj(f4[:-4])
                temp = temp + [A]
                string += str(count)+ "  " + file_name[:-4] + "\n"
                count = count + 1
            f5 = f2 + "documents_index.txt"
            file = open( f5 ,"w")
            file.write(string)
            file.close() 
            padding2(temp, f2 + "documents" , 4633 , word_idx_map)



for SET in os.listdir(DUC):
    f1 = DUC  + SET + "/" + "TOPICS/" 
    if SET == "2004":
        continue
    for topic in os.listdir(f1):
        f2 = f1 + topic + "/"
        if not os.path.isdir(f2):
            continue 
        for type_ in os.listdir(f2):
            if type_ == "documents" :
                continue
            f3 = f2 + type_ + "/"
            if not os.path.isdir(f3):
                continue 
            temp = []
            count = 0
            string = "" 
            for file_name in os.listdir(f3):
                f4 = f3 + file_name
                if f4[-4:] != ".pkl" :
                    continue
                # os.remove(f4)
                A = load_obj(f4[:-4])
                temp = temp + [A] 
                string += str(count)+ "  " + file_name[:-4] + "\n"
                count = count + 1
            f5 = f2 + type_ + "_index.txt"
            file = open( f5 ,"w")
            file.write(string)
            file.close() 
            padding2(temp, f2 + type_ , 333 , word_idx_map)


import os 
DUC = "Processed/"
f1 = DUC  + "2001" + "/" + "Trial/"  

for topic in os.listdir(f1):
    f2 = f1 + topic + "/"
    if not os.path.isdir(f2):
        continue 
    # f2
    for type_ in os.listdir(f2):
        f3 = f2 + type_ 
        if not os.path.isdir(f3):
            continue 
        if type_ == "documents" :
                continue  
        f3 = f2 + type_ +"/"
        # f3
        temp = []
        count = 0
        string = ""     
        for file_name in os.listdir(f3):
            f4 = f3 + file_name 
            if f4[-4:] != ".pkl":
                continue
            A = load_obj(f4[:-4])
            if len(A) > 4633: 
                continue
            temp = temp + [A] 
            string += str(count)+ "  " + file_name[:-4] + "\n"
            count = count + 1
            len(A)
        f5 = f2 + type_ + "_index.txt"
        file = open( f5 ,"w")
        file.write(string)
        file.close() 
        padding2(temp, f2 + type_ , 333 , word_idx_map)

        



for i in range(len(train)):
    temp = train[0:i] + train[i+1:]
    try:
        train = np.array(temp,dtype="int")
    except Exception as e:
        i
        continue


from keras.preprocessing.sequence import pad_sequences


data1 = pad_sequences(find_seq1, maxlen=max_body_length , padding='post')
data2 = pad_sequences(find_seq2, maxlen=max_caption_length , padding='post')

indices = np.arange(data1.shape[0])
np.random.shuffle(indices)

data1 = data1[indices]
data2 = data2[indices]

num_validation_samples = .10 * len(indices)
num_validation_samples = int(num_validation_samples)

x_train = data1[:-num_validation_samples]
y_train = data2[:-num_validation_samples]
x_val = data1[-num_validation_samples:]
y_val = data2[-num_validation_samples:]

x_train[0]
y_train[0]

np.save("x_train.npy" , x_train)
np.save("y_train.npy" , y_train)
np.save("x_val.npy" , x_val)
np.save("y_val.npy" , y_val)

