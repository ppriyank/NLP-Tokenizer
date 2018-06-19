
import tensorflow as tf
import numpy as np
import copy
import random
import numpy as np
from scipy import sparse, io
import pickle


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

stored_data_path = 'priyank_vocab.p'

with open(stored_data_path, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    x = u.load()


    
w2v_embeddings, word_idx_map = x[0], x[1]


embed_size = 300
batch_size = 50
vocab_size = w2v_embeddings.shape[0]

is_training = True
learning_rate = 1e-5
decay_rate_big = 0.1
filter_sizes=[1,2,3,4,5,6,7] #[1,2,3,4,5,6,7]
num_filters = 256 
clip_gradients = 5
num_filters_total = num_filters * len(filter_sizes)
decay_rate = 0.65
decay_steps = 6000

num_epochs = 100
validate_every = 1

l1 = 50
Final_layer = 100
# add placeholder (X,label)

sequence_length1 = 338
sequence_length2= 4638

input_x3 = tf.placeholder(tf.int32, [None, sequence_length1], name="input_x3")  # X
input_x1 = tf.placeholder(tf.int32, [None, sequence_length1], name="input_x1")  # X
input_x2 = tf.placeholder(tf.int32, [None, sequence_length2], name="input_x2")  # X
prediction_label1 = tf.placeholder(tf.float32, [None, 1], name="input_read1")  # Y
prediction_label2 = tf.placeholder(tf.float32, [None, 1], name="input_read2")  # Y



dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")


global_step = tf.Variable(0, trainable=False, name="Global_Step")
epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
epoch_increment=tf.assign(epoch_step,tf.add(epoch_step,tf.constant(1)))
decay_steps, decay_rate = decay_steps, decay_rate

with tf.device('/gpu:0'):
    initializer = tf.random_normal_initializer(stddev=0.1)
    Embedding = tf.get_variable("Embedding",initializer=w2v_embeddings) 
    embedded_words3 = tf.nn.embedding_lookup(Embedding,input_x3)
    embedded_words1 = tf.nn.embedding_lookup(Embedding,input_x1)
    embedded_words2 = tf.nn.embedding_lookup(Embedding,input_x2)
    sentence_embeddings_expanded3=tf.expand_dims(embedded_words3,-1   )
    sentence_embeddings_expanded1=tf.expand_dims(embedded_words1,-1   )
    sentence_embeddings_expanded2=tf.expand_dims(embedded_words2,-1   )
    filter1=tf.get_variable("read",[35,embed_size,1,1],initializer=initializer)
    conv1=tf.nn.conv2d(sentence_embeddings_expanded1, filter1, strides=[1,1,1,1], padding="VALID", name="read_conv1" )
    conv2=tf.nn.conv2d(sentence_embeddings_expanded2, filter1, strides=[1,1,1,1], padding="VALID", name="read_conv2" )
    conv3=tf.nn.conv2d(sentence_embeddings_expanded3, filter1, strides=[1,1,1,1], padding="VALID", name="read_conv3" )
    b=tf.get_variable("read_",[1])
    h1=tf.nn.relu(tf.nn.bias_add(conv1,b),"read_relu")
    h2=tf.nn.relu(tf.nn.bias_add(conv2,b),"read_relu")
    h3=tf.nn.relu(tf.nn.bias_add(conv3,b),"read_relu")
    hread1 = tf.reshape(h1,[-1, h1.get_shape().as_list()[1]])
    hread2 = tf.reshape(h2,[-1, h2.get_shape().as_list()[1]])
    hread3 = tf.reshape(h3,[-1, h3.get_shape().as_list()[1]])
    h_read1=tf.concat([hread1 ,hread2] ,1)
    h_read2=tf.concat([hread3 ,hread2] ,1)
    W_read = tf.get_variable("read_W",shape=[h_read2.get_shape().as_list()[1], 1],initializer=initializer) #[embed_size,label_size]
    b_read = tf.get_variable("read_b",shape=[1])       #[label_size] #ADD 2017.06.09
    read_real = tf.matmul(h_read1,W_read) + b_read
    read_fake = tf.matmul(h_read2,W_read) + b_read
    read_real = tf.nn.sigmoid(read_real)
    read_fake = tf.nn.sigmoid(read_fake)

theta_read = [v for v in tf.trainable_variables() ]

pooled_outputs3 = []
pooled_outputs1 = []
pooled_outputs2 = []
temp = 5



# with tf.device('/gpu:0'):
#     temp = 5
#     for i,filter_size in enumerate(filter_sizes):
#         i , filter_size
#         with tf.name_scope("convolution-pooling-%s" %filter_size):
#             filter=tf.get_variable("filter-%s"%filter_size,[filter_size,embed_size,1,num_filters],initializer=initializer)
#             conv1=tf.nn.conv2d(sentence_embeddings_expanded1, filter, strides=[1,1,1,1], padding="VALID", name="conv1" )
#             conv3 =tf.nn.conv2d(sentence_embeddings_expanded3, filter, strides=[1,1,1,1], padding="VALID", name="conv3" )
#             conv2 = tf.nn.conv2d(sentence_embeddings_expanded2, filter, strides=[1,1,1,1], padding="VALID", name="conv2" ) 
#             b=tf.get_variable("b-%s"%filter_size,[num_filters])
#             h1=tf.nn.relu(tf.nn.bias_add(conv1,b),"relu")
#             h2=tf.nn.relu(tf.nn.bias_add(conv2,b),"relu")
#             h3=tf.nn.relu(tf.nn.bias_add(conv3,b),"relu")
#             pooled1=tf.nn.max_pool(h1, ksize=[1,sequence_length1-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool1")
#             pooled2=tf.nn.max_pool(h2, ksize=[1,sequence_length2-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool2")
#             pooled3=tf.nn.max_pool(h3, ksize=[1,sequence_length1-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool3")
#             pooled_outputs1.append(pooled1)
#             pooled_outputs2.append(pooled2)
#             pooled_outputs3.append(pooled3)
    
with tf.device('/gpu:0'):
    for i,filter_size in enumerate(filter_sizes):
        i , filter_size
        with tf.name_scope("convolution-pooling-%s" %filter_size):
            filter=tf.get_variable("filter-%s"%filter_size,[filter_size,embed_size,1,num_filters],initializer=initializer)
            conv1=tf.nn.conv2d(sentence_embeddings_expanded1, filter, strides=[1,1,1,1], padding="VALID", name="conv1" )
            conv3 =tf.nn.conv2d(sentence_embeddings_expanded3, filter, strides=[1,1,1,1], padding="VALID", name="conv3" )
            b=tf.get_variable("b-%s"%filter_size,[num_filters])
            h1=tf.nn.relu(tf.nn.bias_add(conv1,b),"relu")
            h3=tf.nn.relu(tf.nn.bias_add(conv3,b),"relu")
            pooled1=tf.nn.max_pool(h1, ksize=[1,sequence_length1-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool1")
            pooled3=tf.nn.max_pool(h3, ksize=[1,sequence_length1-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool3")
            pooled_outputs1.append(pooled1)
            pooled_outputs3.append(pooled3)
        with tf.name_scope("convolution-pooling2-%s" %filter_size):
            filter=tf.get_variable("filter2-%s"%filter_size,[filter_size,embed_size,1,num_filters],initializer=initializer)
            conv2 = tf.nn.conv2d(sentence_embeddings_expanded2, filter, strides=[1,1,1,1], padding="VALID", name="conv2" ) 
            b=tf.get_variable("b2-%s"%filter_size,[num_filters])
            h2=tf.nn.relu(tf.nn.bias_add(conv2,b),"relu")
            pooled2=tf.nn.max_pool(h2, ksize=[1,sequence_length2-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool2")
            pooled_outputs2.append(pooled2)
    h_pool1=tf.concat(pooled_outputs1,3)
    h_pool3=tf.concat(pooled_outputs3,3)
    h_pool_flat1=tf.reshape(h_pool1,[-1, (temp + 1 )*num_filters_total])
    h_pool_flat3=tf.reshape(h_pool3,[-1,(temp + 1 )* num_filters_total])
    h_pool2=tf.concat(pooled_outputs2,3)
    h_pool_flat2=tf.reshape(h_pool2,[-1,(temp + 1 )*num_filters_total])
    W_projection1 = tf.get_variable("W_projection1",shape=[(temp + 1 )*num_filters_total, Final_layer],initializer=initializer) #[embed_size,label_size]
    b_projection1 = tf.get_variable("b_projection1",shape=[Final_layer])       #[label_size] #ADD 2017.06.09
    W_H1 = tf.get_variable("W_final1",shape=[Final_layer * 2, l1],initializer=initializer) #[embed_size,label_size]
    b_H1 = tf.get_variable("b_final1",shape=[l1])       #[label_size] #ADD 2017.06.09
    W_H2 = tf.get_variable("W_final2",shape=[l1, 1],initializer=initializer) #[embed_size,label_size]
    b_H2 = tf.get_variable("b_final2",shape=[1])       #[label_size] #ADD 2017.06.09



# with tf.device('/gpu:0'):
#     h_pool1=tf.concat(pooled_outputs1,3)
#     h_pool3=tf.concat(pooled_outputs3,3)
#     h_pool_flat1=tf.reshape(h_pool1,[-1, (temp + 1 )*num_filters_total])
#     h_pool_flat3=tf.reshape(h_pool3,[-1,(temp + 1 )* num_filters_total])
#     h_pool2=tf.concat(pooled_outputs2,3)
#     h_pool_flat2=tf.reshape(h_pool2,[-1,(temp + 1 )*num_filters_total])
#     W_projection1 = tf.get_variable("W_projection1",shape=[(temp + 1 )*num_filters_total, Final_layer],initializer=initializer) #[embed_size,label_size]
#     b_projection1 = tf.get_variable("b_projection1",shape=[Final_layer])       #[label_size] #ADD 2017.06.09
#     W_H1 = tf.get_variable("W_final1",shape=[Final_layer * 2, l1],initializer=initializer) #[embed_size,label_size]
#     b_H1 = tf.get_variable("b_final1",shape=[l1])       #[label_size] #ADD 2017.06.09
#     W_H2 = tf.get_variable("W_final2",shape=[l1, 1],initializer=initializer) #[embed_size,label_size]
#     b_H2 = tf.get_variable("b_final2",shape=[1])       #[label_size] #ADD 2017.06.09


    
theta_topic = [v for v in tf.trainable_variables()  if 'read' not in v.name]

with tf.name_scope("dropout"):
            h_drop1=tf.nn.dropout(h_pool_flat1,keep_prob=dropout_keep_prob)
            h_drop3=tf.nn.dropout(h_pool_flat3,keep_prob=dropout_keep_prob)
            h_drop2=tf.nn.dropout(h_pool_flat2,keep_prob=dropout_keep_prob)


with tf.name_scope("output"):
            logits1 = tf.matmul(h_drop1,W_projection1) + b_projection1  #shape:[None, num_classes]==tf.matmul([None,embed_size],[embed_size,num_classes])
            logits2 = tf.matmul(h_drop2,W_projection1) + b_projection1
            logits3 = tf.matmul(h_drop3,W_projection1) + b_projection1

output_true=tf.concat([logits1 , logits2],1)
output_fake=tf.concat([logits3 , logits2],1)




def discriminator(x):
    global W_H1
    global W_H2
    global b_H1
    global b_H2
    D_h1 = tf.nn.relu(tf.matmul(x, W_H1) + b_H1)
    D_logit = tf.matmul(D_h1, W_H2) + b_H2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

D_real, D_logit_real = discriminator(output_true)
D_fake, D_logit_fake = discriminator(output_fake)

l2_lambda=0.001 




delta1 = D_real - D_fake 
delta2 = read_real - read_fake


L_read= 100 * tf.reduce_sum(tf.log(1 + tf.exp( -10 * delta2 ))) 
L2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()  if 'bias' not in v.name]) * l2_lambda



# ##################################################



loss_read_total = L_read + tf.add_n([tf.nn.l2_loss(v) for v in theta_read if 'bias' not in v.name]) * l2_lambda

loss_topic = 100 * tf.reduce_sum(tf.log(1 + tf.exp(-10  * delta1))) 
loss_topic_total = loss_topic + tf.add_n([tf.nn.l2_loss(v) for v in theta_topic if 'bias' not in v.name]) * l2_lambda

# ##################################################

condition1 = tf.greater(read_fake, 0.1)
delta3  = tf.where(condition1, delta2/400  , delta2 )
L_read= 100 * tf.reduce_sum(tf.log(1 + tf.exp( -10 * delta3 ))) 
L_read = L_read

condition2 = tf.greater(D_real, 0.5)
val_main2  = tf.where(condition2, delta1  , delta1/400 )


condition3 = tf.less(delta1, 0.3)
val_main3  = tf.where(condition3, delta1/400 , delta1  )
loss_topic1 = tf.log(1 + tf.exp(-10  * val_main2))
loss_topic2 = tf.log(1 + tf.exp(-10  * val_main3))

penalty = 100 * (loss_topic1 + loss_topic2)
loss_topic = tf.reduce_sum(penalty) ; 


# ##################################################
# ##################################################



learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps,decay_rate, staircase=True)




opti1 = tf.train.AdamOptimizer(1e-4 , beta1=0.5) 
opti2 = tf.train.AdamOptimizer(1e-4 , beta1=0.5) 
train_op1 = tf.contrib.layers.optimize_loss(L_read +  L2_reg, global_step=global_step,learning_rate=learning_rate, optimizer=opti1,clip_gradients=float(clip_gradients) , variables= theta_read)
train_op2 = tf.contrib.layers.optimize_loss(loss_topic +  L2_reg, global_step=global_step,learning_rate=learning_rate, optimizer=opti2,clip_gradients=float(clip_gradients), variables= theta_topic )


config=tf.ConfigProto()
config.gpu_options.allow_growth=True

x_train = np.load("trainX.npy")
y_train= np.load("trainY.npy" )
x_val = np.load("testX.npy" )
y_val = np.load("testY.npy" )




sess = tf.Session(config = config)    
# sess = tf.Session()    

# saver = tf.train.Saver()
# saver.restore(sess, "final5/model.ckpt")



sess.run(tf.global_variables_initializer())
curr_epoch=sess.run(epoch_step)
number_of_training_data=len(x_train)
trainX = x_train
trainY = y_train
testX = x_val
testY = y_val
temp = copy.deepcopy(trainY)

zeros =np.array([[0.0] for i in range(batch_size)]) 
ones = np.array([[1.0] for i in range(batch_size)])

Tens = np.array([[10.0] for i in range(batch_size)])
sevens = np.array([[0.7] for i in range(batch_size)])
nines = np.array([[0.95] for i in range(batch_size)])
fives = np.array([[0.5] for i in range(batch_size)])
# coherency check
twentys = np.array([[20.0] for i in range(batch_size)])
fives2 = np.array([[5] for i in range(batch_size)])


for epoch in range(curr_epoch,num_epochs):
        loss, counter =  0.0, 0
        for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
            # start , end
            dummy_input2 = temp[start:end]
            for i in range(len(dummy_input2)):
                idx = np.nonzero(dummy_input2[i])
                dummy_input2[i][idx] = np.random.permutation(dummy_input2[i][idx])    
            ram = random.sample(range(number_of_training_data), batch_size)
            dummy_input1 =trainY[ram]
            feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1 ,\
            prediction_label1: sevens , prediction_label2: nines } 
            feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2 }
            feed_dict3 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input1 , input_x3: dummy_input2 } 
            feed_dict4 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1 ,\
            prediction_label1: fives , prediction_label2: nines } 
            curr_loss1,_=sess.run([loss_topic,train_op2  ],feed_dict1)
            curr_loss2,_,_=sess.run([L_read,train_op1 , train_op2],feed_dict2)
            curr_loss4,_=sess.run([loss_topic,train_op2  ],feed_dict4)
            curr_loss3,lr,_=sess.run([L_read,learning_rate,train_op1  ],feed_dict3)
            # loss,counter=loss+curr_loss1+curr_loss2+curr_loss3,counter+1
            counter=counter+1
            if counter %50==0:
                print("Epoch %d\t , Loss =%f " %(epoch , curr_loss1+curr_loss2 + curr_loss3 + curr_loss4))
        sess.run(epoch_increment)
        # print(epoch,validate_every,(epoch % validate_every==0))



        

sess.run(W_H2)



saver = tf.train.Saver()
saver.save(sess, "final4/model.ckpt")


saver = tf.train.Saver()
saver.save(sess, "final5/model.ckpt")

saver = tf.train.Saver()
saver.save(sess, "final8/model.ckpt")

# ###########################################################################

saver = tf.train.Saver()
saver.restore(sess, "final/model.ckpt")





testX = x_val
testY = y_val


number_examples=len(testY)
batch_size = 50


indices = np.arange(number_examples)
np.random.shuffle(indices)
garbage  = testY[indices]
from rouge import Rouge 
rouge = Rouge()

temp2 = copy.deepcopy(testY)


def compute(d):
    global number_examples
    global batch_size
    global testY
    global testX
    global S_G
    global S_IN
    global S_GA
    global S_sparse
    global read_input__
    global read_incoerent__
    global read_garbage__
    global read_sparse__
    for start, end in zip(range(0, number_examples, batch_size),range(batch_size, number_examples, batch_size)):
                print(start,end)
                temp = copy.deepcopy(testY[start:end])
                temp = np.transpose(temp)
                np.random.shuffle(temp)
                temp = np.transpose(temp)
                dummy_input2 = temp2[start:end]
                for i in range(len(dummy_input2)):
                    idx = np.nonzero(dummy_input2[i])
                    dummy_input2[i][idx] = np.random.permutation(dummy_input2[i][idx])    
                feed_dict1 = {input_x2: testX[start:end],dropout_keep_prob: d , input_x1: testY[start:end] , input_x3: temp}
                feed_dict2 = {input_x2: testX[start:end],dropout_keep_prob: d , input_x3: garbage[start:end]}
                feed_dict3 = {input_x2: testX[start:end],dropout_keep_prob: d , input_x1: dummy_input2 }
                pred2,pred3,read_input , read_sparse =sess.run([D_logit_real, D_logit_fake, read_real, read_fake ],feed_dict1)
                pred4,read_garbage =sess.run([D_logit_fake , read_fake],feed_dict2)
                pred9, read_incoerent =sess.run([D_logit_real, read_real],feed_dict3)
                # if d == 1:
                #     if (not np.array_equal(pred1, pred5)) or (not np.array_equal(pred2, pred6)):
                #         print("FUCK IT!!")
                #         break 
                pred2= pred2.tolist()
                S_G += [j[0] for j in pred2]
                pred3 = pred3.tolist()
                S_sparse += [j[0] for j in pred3]
                pred4 = pred4.tolist()
                S_GA += [j[0] for j in pred4]
                pred9 = pred9.tolist()
                S_IN += [j[0] for j in pred9]
                read_input = read_input.tolist()
                read_input__ += [j[0] for j in read_input]
                read_incoerent = read_incoerent.tolist()
                read_incoerent__ += [j[0] for j in read_incoerent]
                read_garbage = read_garbage.tolist()
                read_garbage__ += [j[0] for j in read_garbage]
                read_sparse = read_sparse.tolist()
                read_sparse__ += [j[0] for j in read_sparse]
                                

                
# S_G = []
# S_IN = []
# S_GA = []
# S_sparse = []
# read_input__ = []
# read_incoerent__ = []
# read_garbage__ = []
# read_sparse__ = []
    
# compute(0.5)

# Scores  = [S_G , S_GA , S_IN,  S_sparse , read_input__, read_garbage__, read_incoerent__ , read_sparse__]
# Scores = np.array(Scores)
# np.save("Scores_cnn5.npy" , Scores)



S_G = []
S_IN = []
S_GA = []
S_sparse = []
read_input__ = []
read_incoerent__ = []
read_garbage__ = []
read_sparse__ = []
    
compute(1)

Scores  = [S_G , S_GA , S_IN,  S_sparse , read_input__, read_garbage__, read_incoerent__ , read_sparse__]
Scores = np.array(Scores)
np.save("Scores_cnn1.npy" , Scores)










import numpy as np
from matplotlib.pyplot import *
from numpy.random import *



l1 = "Golden"
l2 = "Garbage"
l3 = "Incoherent Golden"
l8 = "Incoherent Sparse"

l4 = "readibility_incoherent"
l5 = "readibility_garbage"
l6 = "readibility_golden"
l9 = "readibility_sparse"




Scores = np.load("Scores_cnn1.npy")
indices = np.arange(50)
np.random.shuffle(indices)

Scores.shape 

[C,D,E ,F] = plot(Scores[0:4].T[indices])
legend([C,D,E , F], [ l1,l2 , l3 , l8 ], loc=2)
show() 

[A,B,C ,D] = plot(Scores[4:].T[indices])
legend([A,B,C,D], [ l6,l5 , l4 , l9], loc=2)
show()



i = 0 ;
file = open( "Scores.txt" ,"w") 
while True  :
    file.write(str(Scores[4][i]) + "\t" + str(Scores[7][i])) 
    file.write("\n") 
    i=i+1 



A = Scores[4].reshape(1,2300)
B = Scores[6].reshape(1,2300)


import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


A = np.concatenate((Scores[4].reshape(1,2300), Scores[6].reshape(1,2300)), axis=0)

A = A.T
for i in range(A.shape[0]):
    A[i][0] = sigmoid(A[i][0])
    A[i][1] = sigmoid(A[i][1])


np.savetxt("Scores.txt" , A ,  fmt='%f %f' )
