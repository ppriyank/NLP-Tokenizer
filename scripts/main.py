import os
import re
# import nltk
# from nltk.tokenize import sent_tokenize 
# import nltk
# nltk.download('all')
# from nltk.tokenize import word_tokenize
import string 
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet 

# from nltk.stem import PorterStemmer
# from nltk.stem.lancaster import LancasterStemmer
# # from nltk.stem.Snowball import SnowballStemmer
# from nltk.stem import WordNetLemmatizer



# pst = PorterStemmer() 
# lst = LancasterStemmer()
# wlem = WordNetLemmatizer()

# stops=set(stopwords.words('english'))

dest = "../"
    

# wlem.lemmatize("ate")

j=0
till_now = 0
sentences = []
file_list = []
documents = []
Abstract = [] 
i = -1 
src_files = os.listdir(dest)
for file_name in src_files:
    print(file_name)
    i = i + 1
    summary  = os.path.join(dest, file_name)
    if (os.path.isfile(summary)) :
        with open(summary , 'r') as f:
            doc_train = list(f)
            if doc_train[2] == "Introduction:\n":
                temp = doc_train[3:]
                A = "".join(temp)
                B = re.sub("\n" , " " , A)
                B = re.sub(r'[\`\^\*\{\}\[\]\#\<\>\+\=\_\(\)]+',"",B)
                # B = re.sub("," , " " , B)
                B = re.sub("\'\'" , " " , B)
                B = re.sub("``" , " " , B)
                B = re.sub(r'\s| +|--|(- )' , " " , B)
                B = re.sub(r'\'(?![a-z])' , "" , B)
                C= "".join(B)
                documents = documents + [C]
                file_list = file_list + [file_name]
                # if file_name == "128_summ":
                #     break 
            else :
                os.remove(summary)
                continue 
            if doc_train[0] == "Abstract:\n":
                temp = doc_train[1]
                A = "".join(temp)
                B = re.sub("\n" , " " , A)
                B = re.sub(r'[\`\^\*\{\}\[\]\#\<\>\+\=\_\(\)]+',"",B)
                # B = re.sub("," , " " , B)
                B = re.sub("\'\'" , " " , B)
                B = re.sub("``" , " " , B)
                B = re.sub(r'\s| +|--|(- )' , " " , B)
                B = re.sub(r'\'(?![a-z])' , "" , B)
                C= "".join(B)
                Abstract = Abstract + [C]
            else :
                os.remove(summary)
                continue 
                


for  i in range(len(documents)):
    if documents[i] == "\x1a":
        print(i , file_list[i])




samples ="""
Although, …

As a consequence, …

As a result, …

As we have seen, …

At the same time, …

Accordingly, …

An equally significant aspect of…

Another, significant factor in…

Before considering X it is important to note Y

By the same token, …

But we should also consider, …

Despite these criticisms, …it’s popularity remains high.

Certainly, there is no shortage of disagreement within…

Consequently, …

Correspondingly, …

Conversely, …

Chaytor, … in particular, has focused on the

Despite this, …

Despite these criticisms, … the popularity of X remains largely undiminished.

Each of these theoretical positions make an important contribution to our understanding of, …

Evidence for in support of this position, can be found in…,

Evidently,

For this reason, …

For these reasons, …

Furthermore, …

Given, the current high profile debate with regard to, …it is quite surprising that …

Given, the advantages of … outlined in the previous paragraph, …it is quite predictable that …

However, …

Having considered X, it is also reasonable to look at …

Hence, …

In addition to, …

In contrast, …

In this way, …

In this manner, …

In the final analysis, …

In short, …

Indeed, …

It can be seen from the above analysis that, …

It could also be said that, …

It is however, important to note the limitations of…

It is important to note however, that …

It is important however not to assume the applicability of, …in all cases.

It is important however not to overemphasis the strengths of …

In the face of such criticism, proponents of, …have responded in a number of ways.

Moreover, …

Notwithstanding such criticism, ….it’s popularity remains largely undiminished.

Notwithstanding these limitations, ….it worth remains in a number of situations.

Noting the compelling nature of this new evidence, …has suggested that.

Nevertheless, …remains a growing problem.

Nonetheless, the number of, …has continued to expand at an exponential rate.

Despite these criticisms, …it’s popularity remains high.

On the other hand, critics of, …point to its blindness, with respect to.

Of central concern therefore to, …sociologists is explaining how societal processes and institutions…

Proponents of…, have also suggested that…

Subsequently, …

Similarly, …

The sentiment expressed in the quotation, embodies the view that, …

This interpretation of, … has not been without it’s detractors however.

This approach is similar to the, …. position

This critique, unfortunately, implies a singular cause of, …

This point is also sustained by the work of, …

Thirdly, …

This counter argument is supported by evidence from, …

The use of the term, …

Therefore, …

There appears then to be an acceleration in the growth of

There is also, however, a further point to be considered.

These technological developments have greatly increased the growth in, …

Thus, …

To be able to understand, …

"""
prep = ""
first_word = [] 
tokenized_samples = samples.split("\n")

for tokenized in tokenized_samples:
    if tokenized != "":
        # print(tokenized)
        temp = tokenized.split(" ")
        # temp[0]
        if temp[0] not in first_word :
            first_word.append(temp[0])


first_word = first_word + ["Today"]
prep = ""
for words in first_word :
    if words[-1] == ",":
        prep = prep + words +"? |"
    else:    
        prep = prep + words +"(\'[a-z][a-z]?)? |"

prep = prep + "And |Some |[0-9]+(,[0-9]+)?(\.[0-9]+)? "
special = "((Mrs|Ms|Mr|Dr|Sen|Gen)(\.)? | Jr\.|\$|(\?|\.|\!)(?=\"( )+[a-z]))"
accepted1 = "[^ .!\"\'?\n\t`,;:]+(\'[a-z][a-z]?)?"
# accepted2 = "(((\.)?[^ .!\"\'?\n\t`,-]+)*)?"
word_pattern = "(("+special+")|(" + prep + ")|([A-Za-z]\.)+|([A-Z]"+accepted1+"\.(?=,|-| {1,2}[0-9]+))+|(([A-Z]"+accepted1+" ([A-Z]\. )?([A-Z]"+accepted1+"( |,|(\. (?!"+prep +"))))+)( [A-Z]"+accepted1+"\. (?!"+prep +"))?)|" + accepted1 +")"
word_pattern = "(("+special+")|(" + prep + ")|([A-Za-z]\.)+|([A-Z]"+accepted1+"\.(?=,|-| {1,2}[0-9]+))+|(([A-Z]"+accepted1+" ([A-Z]\. )?([A-Z]"+accepted1+"( |,))+))|" + accepted1 +")"
words=[]
word_compile = re.compile(word_pattern)

# word_compile = re.compile(accepted1)

i = 12
file_list[i]
doc_i = documents[i]
doc_i=doc_i[:-1]

# tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
# text = tokenizer.tokenize(doc_i)

# sentences = nltk.sent_tokenize(doc_i)
# tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

sentences =[]
t = 0
sentences.append("")
for m in word_compile.finditer(doc_i):
    if  len(m.group()) == 0 :
        if len(re.findall("\.|\?|\!|;", doc_i[till_now:m.start()]))>0 :
            sentences.append("")
            t= t + 1
        till_now = m.start() 
        continue
    # if m.start() == 1053:
    #     break
    # if len(m.group()) == 1 and m.group() == "_" :
    #     continue
    else :
        token = m.group() 
        if m.group()[-1] == " " : 
          token = m.group()[:-1]      
        if token[-1] == "," or token[-1] == "{" or token[-1] == "-" or token[-1] == "}" or token[-1] == ")" :
            token = token[:-1] 
        if token[0] == "(" or token[0] == "{" or token[0] == "-":
            token = token[1:]
    # if token == "(408" :
    #     break
    print (m.start(), token)
    words = words + [token] 
    if len(re.findall("\.|\?|\!|;", doc_i[till_now:m.start()]))>0 :
        sentences.append("")
        t= t + 1
        # flag = 1 
    sentences[t] = sentences[t] + token + " "
    # print(doc_i[till_now:m.start()])
    till_now = m.start() + len(token)
    # flag = 0 
    

Vocabulary = []
tokenized_sentence = []
max_length = 0
new_sentence = []
count  = 0 
token = ""
entities = []

# new_first_word = []
# for temp in first_word :
#     temp = re.sub(r'( )+$',"", temp)
#     temp = re.sub(r'[\.,\']',"", temp)
#     new_first_word.append(temp.lower())

# new_first_word = new_first_word + ["some" , "that" ,"and"]


for stuff in sentences:
        if stuff != "":
            tokenized_sentence.append([])
            print("---------", stuff)  
            new_sentence  = new_sentence  + [stuff] 
            for m in word_compile.finditer(stuff):
                token  = m.group()
                token = re.sub(r'( )+$',"", token)
                token = re.sub(r',|(\.$)',"", token)
                token = re.sub(r'\'[a-z]+',"", token)
                if len(re.findall("^[A-Z]", token)) <= 0 :
                    if len(re.findall("-", token))>0 :
                        temp = token.split("-")
                        for i in range(len(temp)):
                            if temp[i] not in Vocabulary : 
                                Vocabulary.append(temp[i].lower())
                            tokenized_sentence[count].append(temp[i].lower())    
                        continue
                tokenized_sentence[count].append(token.lower())
                if token.lower() not in Vocabulary:
                    Vocabulary.append(token.lower())    
            print("=======" , tokenized_sentence[count])
            count = count + 1
            # break 


Vectors = {}
Vocabulary_marker =[0 for i in range(len(Vocabulary)) ]

emb_path="lstm based similarity/wiki.en.vec"
for line in open(emb_path):
    l = line.strip().split()
    st = l[-301]
    st=st.lower()
    if st not in Vocabulary :
        continue
    else:
        Vectors[st]=np.asarray(l[-300:] , dtype=float)
        Vocabulary_marker[ Vocabulary.index(st)] =1 

count = 0 
for i in range(len(Vocabulary)):
    if Vocabulary_marker[i] == 0 :
        print(Vocabulary[i])
        Vectors[Vocabulary[i]]=np.array([0 for j in range(count)] + [1] + [0 for j in range(299 - count)] , dtype=float)
        Vocabulary_marker[ Vocabulary.index(Vocabulary[i])] =1 
        count = count + 1


tf_idf = [[0.5 for i in range(len(Vocabulary))] for j in range(len(tokenized_sentence))]
for j in range(len(tokenized_sentence)):
    for token in tokenized_sentence[j]:
        # print(token)
        tf_idf[j][Vocabulary.index(token)] = tf_idf[j][Vocabulary.index(token)] + 1 

from math import log
idf = {}
count = [0 for i in range(len(Vocabulary))]
for j in range(len(Vocabulary)):
    for k in range(len(tokenized_sentence)):
        if Vocabulary[j] in tokenized_sentence[k]:
            count[j]  =count[j] + 1
    idf[Vocabulary[j]] = log(len(tokenized_sentence)/ (1+ count[j]) )

temp = np.max(tf_idf)

for j in range(len(tokenized_sentence)):
    for token in tokenized_sentence[j]:
        # print(token)
        tf_idf[j][Vocabulary.index(token)] = (tf_idf[j][Vocabulary.index(token)] / (temp)) * idf[token]
        




import numpy as np 

max_count = 0

for this_iter in range(len(documents)):
    print(this_iter)
    doc_i = documents[this_iter]
    abs_i = Abstract[this_iter]
    doc_i=doc_i[:-1]
    sentences =[]
    t = 0
    sentences.append("")
    for m in word_compile.finditer(doc_i):
        if  len(m.group()) == 0 :
            if len(re.findall("\.|\?|\!|;", doc_i[till_now:m.start()]))>0 :
                sentences.append("")
                t= t + 1
            till_now = m.start() 
            continue
        else :
            token = m.group() 
            if m.group()[-1] == " " : 
              token = m.group()[:-1]      
            if token[-1] == "," or token[-1] == "{" or token[-1] == "-" or token[-1] == "}" or token[-1] == ")" :
                token = token[:-1] 
            if token[0] == "(" or token[0] == "{" or token[0] == "-":
                token = token[1:]
        words = words + [token] 
        if len(re.findall("\.|\?|\!|;", doc_i[till_now:m.start()]))>0 :
            sentences.append("")
            t= t + 1
            # flag = 1 
        sentences[t] = sentences[t] + token + " "
        till_now = m.start() + len(token)
    tokenized_sentence = []
    new_sentence = []
    count  = 0 
    token = ""
    for stuff in sentences:
            if stuff != "":
                new_sentence  = new_sentence  + [stuff] 
                # tokenized_sentence.append([])                
    np.savetxt('sentence_dataset/train_%d.txt'%(this_iter), new_sentence, delimiter=" ", fmt="%s") 
    sentences =[]
    t = 0
    sentences.append("")
    for m in word_compile.finditer(abs_i):
        if  len(m.group()) == 0 :
            if len(re.findall("\.|\?|\!|;", abs_i[till_now:m.start()]))>0 :
                sentences.append("")
                t= t + 1
            till_now = m.start() 
            continue
        else :
            token = m.group() 
            if m.group()[-1] == " " : 
              token = m.group()[:-1]      
            if token[-1] == "," or token[-1] == "{" or token[-1] == "-" or token[-1] == "}" or token[-1] == ")" :
                token = token[:-1] 
            if token[0] == "(" or token[0] == "{" or token[0] == "-":
                token = token[1:]
        words = words + [token] 
        if len(re.findall("\.|\?|\!|;", abs_i[till_now:m.start()]))>0 :
            sentences.append("")
            t= t + 1
            # flag = 1 
        sentences[t] = sentences[t] + token + " "
        till_now = m.start() + len(token)
    tokenized_sentence = []
    new_sentence = []
    count  = 0 
    token = ""
    for stuff in sentences:
            if stuff != "":
                new_sentence  = new_sentence  + [stuff] 
                # tokenized_sentence.append([])                
    np.savetxt('Abstract_dataset/train_%d.txt'%(this_iter), new_sentence, delimiter=" ", fmt="%s") 
    







                for m in word_compile.finditer(stuff):
                    token  = m.group()
                    token = re.sub(r'( )+$',"", token)
                    token = re.sub(r',|(\.$)',"", token)
                    token = re.sub(r'\'[a-z]+',"", token)
                    # print(token)
                    if len(re.findall("^[A-Z]", token)) <= 0 :
                        if len(re.findall("-", token))>0 :
                            temp = token.split("-")
                            for i in range(len(temp)):
                                tokenized_sentence[count].append(temp[i].lower())    
                            continue
                    tokenized_sentence[count].append(token.lower())
                if len(tokenized_sentence[count]) > max_count:
                    max_count = len(tokenized_sentence[count])
                    print(this_iter, stuff , len(tokenized_sentence[count]))
                count = count + 1
                



