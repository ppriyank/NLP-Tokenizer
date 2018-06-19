
import re

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
