
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


folder = "CNN-Dailymail/"

summary_path = folder + 'summary_embedding.npz'
story_path = folder + 'story_embedding.npz'
stored_data_path = folder + 'dataset.p'


story_embeddings = load_sparse_csr(story_path)
print (story_embeddings.shape)

summary_embeddings = load_sparse_csr(summary_path)
print (summary_embeddings.shape)

story_embeddings = story_embeddings.toarray()
print (story_embeddings[0][24])

summary_embeddings = summary_embeddings.toarray()
print (summary_embeddings[0][24])

with open(stored_data_path, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    x = u.load()
    

w2v_embeddings, word_idx_map = x[0], x[1]

embed_size = 300
batch_size = 10
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


Final_layer = 100
# add placeholder (X,label)

sequence_length1 = 99
sequence_length2= 1906

input_x3 = tf.placeholder(tf.int32, [None, sequence_length1], name="input_x3")  # X
input_x1 = tf.placeholder(tf.int32, [None, sequence_length1], name="input_x1")  # X
input_x2 = tf.placeholder(tf.int32, [None, sequence_length2], name="input_x2")  # X
prediction_label = tf.placeholder(tf.float32, [None, 1], name="input_x2")  # Y
dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

global_step = tf.Variable(0, trainable=False, name="Global_Step")
epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
epoch_increment=tf.assign(epoch_step,tf.add(epoch_step,tf.constant(1)))
decay_steps, decay_rate = decay_steps, decay_rate

with tf.device('/gpu:1'):
    initializer = tf.random_normal_initializer(stddev=0.1)
    Embedding = tf.get_variable("Embedding",initializer=w2v_embeddings) #[vocab_size,embed_size] tf.random_uniform([vocab_size, embed_size],-1.0,1.0)
    W_projection2 = tf.get_variable("W_projection2",shape=[num_filters_total, Final_layer],initializer=initializer) #[embed_size,label_size]
    b_projection2 = tf.get_variable("b_projection2",shape=[Final_layer])       #[label_size] #ADD 2017.06.09
    embedded_words3 = tf.nn.embedding_lookup(Embedding,input_x3)
    embedded_words1 = tf.nn.embedding_lookup(Embedding,input_x1)
    embedded_words2 = tf.nn.embedding_lookup(Embedding,input_x2)
    sentence_embeddings_expanded3=tf.expand_dims(embedded_words3,-1   )
    sentence_embeddings_expanded1=tf.expand_dims(embedded_words1,-1   )
    sentence_embeddings_expanded2=tf.expand_dims(embedded_words2,-1   )
    pooled_outputs3 = []
    pooled_outputs1 = []
    pooled_outputs2 = []
    temp = 9
    for i,filter_size in enumerate(filter_sizes):
        i , filter_size
        temp2 = 10
        with tf.name_scope("convolution-pooling-%s" %filter_size):
            filter=tf.get_variable("filter-%s"%filter_size,[filter_size,embed_size,1,temp2],initializer=initializer)
            conv1=tf.nn.conv2d(sentence_embeddings_expanded1, filter, strides=[1,1,1,1], padding="VALID", name="conv1" )
            conv3 =tf.nn.conv2d(sentence_embeddings_expanded3, filter, strides=[1,1,1,1], padding="VALID", name="conv3" )
            conv2 = tf.nn.conv2d(sentence_embeddings_expanded2, filter, strides=[1,1,1,1], padding="VALID", name="conv2" ) 
            b=tf.get_variable("b-%s"%filter_size,[temp2])
            h1=tf.nn.relu(tf.nn.bias_add(conv1,b),"relu")
            h2=tf.nn.relu(tf.nn.bias_add(conv2,b),"relu")
            h3=tf.nn.relu(tf.nn.bias_add(conv3,b),"relu")
            filter2=tf.get_variable("filter-2%s"%filter_size,[sequence_length1-filter_size+1 - temp,1,10,256],initializer=initializer)
            filter3=tf.get_variable("filter2-2%s"%filter_size,[sequence_length2-filter_size+1- temp,1,10,256],initializer=initializer)
            conv_1=tf.nn.conv2d(h1, filter2, strides=[1,1,1,1], padding="VALID", name="conv1" )
            conv_3=tf.nn.conv2d(h3, filter2, strides=[1,1,1,1], padding="VALID", name="conv1" )
            conv_2=tf.nn.conv2d(h2, filter3, strides=[1,1,1,1], padding="VALID", name="conv1" )
            b1=tf.get_variable("b-1%s"%filter_size,[num_filters])
            h_1=tf.nn.relu(tf.nn.bias_add(conv_1,b1),"relu")
            h_3=tf.nn.relu(tf.nn.bias_add(conv_3,b1),"relu")
            h_2=tf.nn.relu(tf.nn.bias_add(conv_2,b1),"relu")
            pooled_outputs2.append(h_2)
            pooled_outputs1.append(h_1)
            pooled_outputs3.append(h_3)
        

for i,filter_size in enumerate(filter_sizes):
    i , filter_size
    with tf.name_scope("convolution-pooling-%s" %filter_size):
        filter=tf.get_variable("filter-%s"%filter_size,[filter_size,embed_size,1,num_filters],initializer=initializer)
        conv1=tf.nn.conv2d(sentence_embeddings_expanded1, filter, strides=[1,1,1,1], padding="VALID", name="conv1" )
        conv3 =tf.nn.conv2d(sentence_embeddings_expanded3, filter, strides=[1,1,1,1], padding="VALID", name="conv3" )
        b=tf.get_variable("b-%s"%filter_size,[num_filters])
        h1=tf.nn.relu(tf.nn.bias_add(conv1,b),"relu")
        h3=tf.nn.relu(tf.nn.bias_add(conv3,b),"relu")
        filter2=tf.get_variable("filter-2%s"%filter_size,[sequence_length1-filter_size+1 - 9,1,256,256],initializer=initializer)
        conv_1=tf.nn.conv2d(h1, filter2, strides=[1,1,1,1], padding="VALID", name="conv1" )
        conv_3=tf.nn.conv2d(h3, filter2, strides=[1,1,1,1], padding="VALID", name="conv1" )
        # conv_3 , conv_1
        b1=tf.get_variable("b-1%s"%filter_size,[num_filters])
        h_1=tf.nn.relu(tf.nn.bias_add(conv_1,b1),"relu")
        h_3=tf.nn.relu(tf.nn.bias_add(conv_3,b1),"relu")  
        pooled_outputs1.append(h_1)
        pooled_outputs3.append(h_3)
    with tf.name_scope("convolution-pooling2-%s" %filter_size):
        filter=tf.get_variable("filter2-%s"%filter_size,[filter_size,embed_size,1,num_filters],initializer=initializer)
        conv2 = tf.nn.conv2d(sentence_embeddings_expanded2, filter, strides=[1,1,1,1], padding="VALID", name="conv2" ) 
        b=tf.get_variable("b2-%s"%filter_size,[num_filters])
        h2=tf.nn.relu(tf.nn.bias_add(conv2,b),"relu")
        filter2=tf.get_variable("filter2-2%s"%filter_size,[sequence_length2-filter_size+1- 9,1,256,256],initializer=initializer)
        conv_2=tf.nn.conv2d(h2, filter2, strides=[1,1,1,1], padding="VALID", name="conv1" )
        b1=tf.get_variable("b-2%s"%filter_size,[num_filters])
        h_2=tf.nn.relu(tf.nn.bias_add(conv_2,b1),"relu")
        pooled_outputs2.append(h_2)
    

h_pool1=tf.concat(pooled_outputs1,3)
h_pool3=tf.concat(pooled_outputs3,3)
h_pool_flat1=tf.reshape(h_pool1,[-1, (temp + 1 )*num_filters_total])
h_pool_flat3=tf.reshape(h_pool3,[-1,(temp + 1 )* num_filters_total])

h_pool2=tf.concat(pooled_outputs2,3)
h_pool_flat2=tf.reshape(h_pool2,[-1,(temp + 1 )*num_filters_total])

W_projection1 = tf.get_variable("W_projection1",shape=[(temp + 1 )*num_filters_total, Final_layer],initializer=initializer) #[embed_size,label_size]
b_projection1 = tf.get_variable("b_projection1",shape=[Final_layer])       #[label_size] #ADD 2017.06.09


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

l1 = 20
W_H1 = tf.get_variable("W_final1",shape=[Final_layer * 2, l1],initializer=initializer) #[embed_size,label_size]
b_H1 = tf.get_variable("b_final1",shape=[l1])       #[label_size] #ADD 2017.06.09

W_H2 = tf.get_variable("W_final2",shape=[l1, 1],initializer=initializer) #[embed_size,label_size]
b_H2 = tf.get_variable("b_final2",shape=[1])       #[label_size] #ADD 2017.06.09


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

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))


learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps,decay_rate, staircase=True)

l2_lambda=0.001 


L2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()  if 'bias' not in v.name]) * l2_lambda

loss_val = 100*(G_loss + D_loss) + L2_reg

opti = tf.train.AdamOptimizer(1e-4 , beta1=0.5)
# opti = tf.train.RMSPropOptimizer(1e-3, momentum=0.4, decay=0.2)
train_op = tf.contrib.layers.optimize_loss(loss_val, global_step=global_step,learning_rate=learning_rate, optimizer=opti,clip_gradients=float(clip_gradients))

config=tf.ConfigProto()
config.gpu_options.allow_growth=True


x_train = np.load("x_train_cnn.npy")
y_train= np.load("y_train_cnn.npy" )
x_val = np.load("x_val_cnn.npy" )
y_val = np.load("y_val_cnn.npy" )


sess = tf.Session(config = config)    
sess.run(tf.global_variables_initializer())
curr_epoch=sess.run(epoch_step)
number_of_training_data=len(x_train)
trainX = x_train
trainY = y_train
testX = x_val
testY = y_val
temp = copy.deepcopy(trainY)



# coherency check
for epoch in range(curr_epoch,num_epochs):
        loss, counter =  0.0, 0
        for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
            # start , end
            temp = np.transpose(temp)
            np.random.shuffle(temp)
            temp = np.transpose(temp)
            dummy_input2 = temp[start:end]
            ram = random.sample(range(number_of_training_data), batch_size)
            dummy_input1 =trainY[ram]
            feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1}
            feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2}
            curr_loss1,lr,_=sess.run([loss_val,learning_rate,train_op],feed_dict1)
            curr_loss2,lr,_=sess.run([loss_val,learning_rate,train_op],feed_dict2)
            loss,counter=loss+curr_loss1+curr_loss2,counter+1
            if counter %50==0:
                print("Epoch %d\t , Loss =%f " %(epoch , curr_loss1+curr_loss2))
        sess.run(epoch_increment)
        # print(epoch,validate_every,(epoch % validate_every==0))





saver = tf.train.Saver()
saver.save(sess, "model_strided_sharing_Kmax/model.ckpt")



saver = tf.train.Saver()
saver.restore(sess, "model_cnn/model.ckpt")



testX = x_val
testY = y_val
number_examples=len(testY)
batch_size = 50


indices = np.arange(number_examples)
np.random.shuffle(indices)

garbage  = testY[indices]

from rouge import Rouge 
rouge = Rouge()



# coherency check
def compute(d):
    global number_examples
    global batch_size
    global testY
    global testX
    global S_G_S
    global S_GA_S
    global S_G_S
    global S_G
    global S_IN_S
    global S_IN
    global S_GA_S
    global S_GA
    global garbage
    for start, end in zip(range(0, number_examples, batch_size),range(batch_size, number_examples, batch_size)):
                print(start,end)
                temp = copy.deepcopy(testY[start:end])
                temp = np.transpose(temp)
                np.random.shuffle(temp)
                temp = np.transpose(temp)
                feed_dict1 = {input_x2: testX[start:end],dropout_keep_prob: d , input_x1: testY[start:end] , input_x3: temp}
                feed_dict2 = {input_x2: testX[start:end],dropout_keep_prob: d , input_x1: testY[start:end] , input_x3: garbage[start:end]}
                pred1,pred2,pred3,pred4=sess.run([D_real,D_logit_real,D_fake , D_logit_fake],feed_dict1)
                pred5,pred6,pred7,pred8=sess.run([D_real,D_logit_real,D_fake , D_logit_fake],feed_dict2)
                if d == 1:
                    if (not np.array_equal(pred1, pred5)) or (not np.array_equal(pred2, pred6)):
                        print("FUCK IT!!")
                        break 
                pred1 = pred1.tolist()
                S_G_S += [j[0] for j in pred1]
                pred2= pred2.tolist()
                S_G += [j[0] for j in pred2]
                pred3 = pred3.tolist()
                S_IN_S += [j[0] for j in pred3]
                pred4 = pred4.tolist()
                S_IN += [j[0] for j in pred4]
                pred7 = pred7.tolist()
                S_GA_S += [j[0] for j in pred7]
                pred8 = pred8.tolist()
                S_GA += [j[0] for j in pred8]
                


S_G = []
S_G_S = []
S_GA = []
S_GA_S = []
S_IN = []
S_IN_S = []

compute(0.5)
len(S_G )     
len(S_GA )     
len(S_G_S )        
len(S_GA_S )     
len(S_IN)
len(S_IN_S)

Scores  = [S_G , S_GA , S_IN , S_G_S , S_GA_S , S_IN_S ]
Scores = np.array(Scores)
np.save("Scores_cnn5.npy" , Scores)

S_G = []
S_G_S = []
S_GA = []
S_GA_S = []
S_IN = []
S_IN_S = []

compute(1)
len(S_G )     
len(S_GA )     
len(S_G_S )        
len(S_GA_S )     
len(S_IN)
len(S_IN_S)

Scores  = [S_G , S_GA , S_IN , S_G_S , S_GA_S , S_IN_S ]
Scores = np.array(Scores)
np.save("Scores_cnn1.npy" , Scores)




import numpy as np
from matplotlib.pyplot import *
from numpy.random import *


l1 = "Golden"
l2 = "Garbage"
l3 = "Incoherent Golden"


Scores = np.load("Scores_cnn5.npy")
indices = np.arange(50)
np.random.shuffle(indices)

Scores.shape 

[A,B , C] = plot(Scores[0:3].T[indices])
legend([A,B, C], [ l1,l2 , l3], loc=2)
show()


[A,B, C] = plot(Scores[3:6].T[indices])
legend([A,B, C], [ l1,l2 , l3], loc=2)
show()


