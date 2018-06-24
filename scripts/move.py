import os
import shutil

src1 = "../../data/DUC2001/Summaris/"
src2 = "../../data/DUC2001/"
dest = "../"

    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False




i = -1 
src_files = os.listdir(src1)
for file_name in src_files:
    i = i + 1
    summary  = os.path.join(src1, file_name)
    if (os.path.isfile(summary)) :
    	summary_copy = dest + str(i) +"_summ"
    	shutil.copy(summary, summary_copy)
    print(summary , orig )


    

    
    new_token = x.sub(u'', token)
    if not new_token == u'':
            new_review.append(new_token)
    tokenized_docs_no_punctuation.append(new_review)    
>>> print(tokenized_docs_no_punctuation)









import nltk
i = 5#till 3987
file_list[i]
doc_i = documents[i]
j=0
doc_i=doc_i[:-1]
till_now = 0



for sent in sent_tokenize(doc_i):
    print(sent)
    if len(sent) <= 4:
        continue
    word_tokenize(sent)     
    print("========" , sent)
    for m in word_compile.finditer(sent):
        if  len(m.group()) == 0 :
            continue
        if len(m.group()) == 1 and m.group() == "_" :
            continue
        else : 
            if m.group()[-1] == "," or m.group()[-1] == "{" or m.group()[-1] == "."  :
                token = m.group()[:-1] 
            elif m.group == "Inc" :
                words[j-1] = words[j-1] + " Inc."
            elif m.group()[0] == "(" or m.group()[0] == "{" :
                token = m.group()[1:] 
            else :
                token = m.group() 
            if token[-1] == "}" or token[-1] == ")" : 
                token = token[:-1]     
        print (m.start(), token)
        words = words + [token] 
        j =j + 1



    


# words    
for m in word_compile.finditer(doc_i):
    if  len(m.group()) == 0 :
        continue
    # if m.start() == 5849:
    #     break
    if len(m.group()) == 1 and m.group() == "_" :
        continue
    else : 
        if m.group()[-1] == "," or m.group()[-1] == "{" or m.group()[-1] == "."  :
            token = m.group()[:-1] 
        elif m.group == "Inc" :
            words[j-1] = words[j-1] + " Inc."
        elif m.group()[0] == "(" or m.group()[0] == "{" :
            token = m.group()[1:] 
        else :
            token = m.group() 
        if token[-1] == "}" or token[-1] == ")" : 
          token = token[:-1]     
    # print (m.start(), token)
    words = words + [token] 
    j =j + 1
    if m.start() == 190:
        break
    print(m.start(), "-----" , doc_i[till_now :  m.start()] ,"----")
    till_now = m.start() + len(m.group())




# Vocublary = []

# tf = 
# idf = 


# for  word in in words:
#     if word not in Vocublary:
#         Vocublary = Vocublary + [word]
#         tf[word] = tf[word]  + 1









sentence_temp = []
doc_formed = ""
till_now = 0
j = 1
for m in sentence.finditer(doc_i):
    # print (m.start(), m.group())
    # print( m.group())
    print(i , "==============" , doc_i[till_now :  m.start()])    
    till_now = m.start() + len(m.group())
    j = j + 1



import re

main_sentence_quote = "(([A-Z][a-z]*\.)|([A-Z][A-Z])|(vs\.)|(\.,)|(\.( )?\.( )?\.( )?)|([A-Z]\.)|([a-z]?\.[a-z]?\.([a-z]?\.)?)|([A-Z]\.[A-Z]\.)|(\([^)]*\))|([A-Za-z0-9]*\.[a-z0-9])|(([A-Z][a-z][a-z]?)\.( )?[A-Za-z]*)|[^\.\?\!])*"
# main_sentence_quote = "((Ariz\.)|(Corp\.'s)|(Wash\.)|(Conn\.)|(Mass\.)|([A-Z][A-Z])|([A-Z]\.)|([a-z]?\.[a-z]?\.)|([A-Z]\.[A-Z]\.)|(\([^)]*\))|([A-Za-z0-9]*\.[a-z0-9])|(([A-Z][a-z][a-z]?)\.( )?[A-Za-z]*)|[^\.\?\!])*"
sentence_pattern_quote = "(([A-Z0-9]\.)+|([A-Z][a-z]\.)|[A-Z0-9])" + main_sentence_quote + "([\'\"]?[\.\?\!][\'\"]?)"


sentence = re.compile(sentence_pattern_quote)

sentences = []

# for i in range(len(documents)):
for i in range(100):
    doc_i = documents[i]
    sentence_temp = []
    doc_formed = ""
    till_now = 0 
    for m in sentence.finditer(doc_i):
        # print (m.start(), m.group())
        # print( m.group())
        print(i , "==============" , doc_i[till_now :  m.start()])    
        till_now = m.start() + len(m.group())





for i in range(100):
    doc_i = documents[i]
    sentence_temp = []
    doc_formed = "" 
    for m in sentence.finditer(doc_i):
        print (m.start(), m.group())
        sentence_temp.append(m.group())
    sentences.append(sentence_temp)
    


dot_checker = "[\.\?\!]"
for j in range(100):
    for i in sentences[j]:
        if i 
        print("====", i)



words= []

word_pattern  = "(([A-Z][^ ,]* )|[^ ,]*)"
word = re.compile(word_pattern)

for i in sentences[1]:
    print("---", i)




    till_now = 0 
    for m in word.finditer(i):
        if m.group() != "":
            print( m.group() ,"------------" ,i[till_now : m.start()])
            till_now = m.start() + len(m.group()            )


    
    



for i in range(0,10):
    doc_i = documents[i]
    till_now = 0 
    for m in word.finditer(doc_i):
        print(i , "==============" , doc_i[till_now :  m.start()])    
        till_now = m.start() + len(m.group())




for i in range(len(documents)):
    doc_i = documents[i]
    word_temp = []
    doc_formed = ""
    for m in word.finditer(doc_i):
        print (m.start(), m.group())
        sentence_temp = sentence_temp + [m.group]
    sentences.append(sentence_temp)



