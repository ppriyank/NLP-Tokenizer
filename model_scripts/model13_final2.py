
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

l1 = 20
Final_layer = 100
# add placeholder (X,label)

sequence_length1 = 99
sequence_length2= 1906

input_x3 = tf.placeholder(tf.int32, [None, sequence_length1], name="input_x3")  # X
input_x1 = tf.placeholder(tf.int32, [None, sequence_length1], name="input_x1")  # X
input_x2 = tf.placeholder(tf.int32, [None, sequence_length2], name="input_x2")  # X
prediction_label1 = tf.placeholder(tf.float32, [None, 1], name="input_read1")  # Y
prediction_label2 = tf.placeholder(tf.float32, [None, 1], name="input_read2")  # Y

topic_label1 = tf.placeholder(tf.float32, [None, 1], name="input_read1")  # Y
topic_label2 = tf.placeholder(tf.float32, [None, 1], name="input_read2")  # Y


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

theta_read = [v for v in tf.trainable_variables() ]

pooled_outputs3 = []
pooled_outputs1 = []
pooled_outputs2 = []
temp = 9


embedded_words1_transpose = tf.transpose(embedded_words1, perm=[0, 2, 1])
embedded_words2_transpose = tf.transpose(embedded_words2, perm=[0, 2, 1])
embedded_words3_transpose = tf.transpose(embedded_words3, perm=[0, 2, 1])


sentence_embeddings_expanded_transpose1=tf.expand_dims(embedded_words1_transpose,-1   )
sentence_embeddings_expanded_transpose2=tf.expand_dims(embedded_words2_transpose,-1   )
sentence_embeddings_expanded_transpose3=tf.expand_dims(embedded_words3_transpose,-1   )


with tf.device('/gpu:0'):
    temp = 9
    for i,filter_size in enumerate(filter_sizes):
        i , filter_size
        with tf.name_scope("convolution-pooling-%s" %filter_size):
            filter=tf.get_variable("filter-%s"%filter_size,[embed_size , filter_size,1,num_filters],initializer=initializer)
            b=tf.get_variable("b-%s"%filter_size,[num_filters])
            conv1=tf.nn.conv2d(sentence_embeddings_expanded_transpose1, filter, strides=[1,1,1,1], padding="VALID", name="conv1" )
            h1=tf.nn.relu(tf.nn.bias_add(conv1,b),"relu")
            conv1_transpose  = tf.transpose(h1, perm=[0,1, 3, 2])
            pooled1 = tf.nn.top_k( conv1_transpose, k=9, sorted=False )
            conv2=tf.nn.conv2d(sentence_embeddings_expanded_transpose2, filter, strides=[1,1,1,1], padding="VALID", name="conv1" )
            h2=tf.nn.relu(tf.nn.bias_add(conv2,b),"relu")
            conv2_transpose  = tf.transpose(h2, perm=[0,1, 3, 2])
            pooled2 = tf.nn.top_k( conv2_transpose, k=9, sorted=False )
            conv3=tf.nn.conv2d(sentence_embeddings_expanded_transpose3, filter, strides=[1,1,1,1], padding="VALID", name="conv1" )
            h3=tf.nn.relu(tf.nn.bias_add(conv3,b),"relu")
            conv3_transpose  = tf.transpose(h3, perm=[0,1, 3, 2])
            pooled3 = tf.nn.top_k( conv3_transpose, k=9, sorted=False )
            pooled_outputs1.append(pooled1[0])
            pooled_outputs2.append(pooled2[0])
            pooled_outputs3.append(pooled3[0])
    


with tf.device('/gpu:0'):
    h_pool1=tf.concat(pooled_outputs1,2)
    h_pool3=tf.concat(pooled_outputs3,2)
    h_pool2=tf.concat(pooled_outputs2,2)
    h_pool_flat1=tf.reshape(h_pool1,[-1, (temp )*num_filters_total ])
    h_pool_flat2=tf.reshape(h_pool2,[-1, (temp  )*num_filters_total ])
    h_pool_flat3=tf.reshape(h_pool3,[-1,(temp  )* num_filters_total])
    W_projection1 = tf.get_variable("W_projection1",shape=[(temp )*num_filters_total, Final_layer],initializer=initializer) #[embed_size,label_size]
    b_projection1 = tf.get_variable("b_projection1",shape=[Final_layer])       #[label_size] #ADD 2017.06.09
    W_H1 = tf.get_variable("W_final1",shape=[Final_layer * 2, l1],initializer=initializer) #[embed_size,label_size]
    b_H1 = tf.get_variable("b_final1",shape=[l1])       #[label_size] #ADD 2017.06.09
    W_H2 = tf.get_variable("W_final2",shape=[l1, 1],initializer=initializer) #[embed_size,label_size]
    b_H2 = tf.get_variable("b_final2",shape=[1])       #[label_size] #ADD 2017.06.09


    
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


# ################################################
G_loss = tf.reduce_mean(tf.nn.l2_loss((D_logit_real - topic_label1)))
D_loss = tf.reduce_mean(tf.nn.l2_loss((D_logit_fake - topic_label2)))

learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps,decay_rate, staircase=True)
l2_lambda=0.001 
L_read= 100 * ( tf.nn.l2_loss((read_real - prediction_label1)) + tf.nn.l2_loss((read_fake - prediction_label2))  )
L2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()  if 'bias' not in v.name]) * l2_lambda



loss_val = 100*(G_loss + D_loss) + tf.reduce_sum(L_read)  + L2_reg

opti = tf.train.AdamOptimizer(1e-4 , beta1=0.5) 
train_op = tf.contrib.layers.optimize_loss(loss_val, global_step=global_step,learning_rate=learning_rate, optimizer=opti,clip_gradients=float(clip_gradients) )



config=tf.ConfigProto()
config.gpu_options.allow_growth=True

x_train = np.load("x_train_cnn.npy")
y_train= np.load("y_train_cnn.npy" )
x_val = np.load("x_val_cnn.npy" )
y_val = np.load("y_val_cnn.npy" )





# ##########################################

delta = D_real - D_fake 

loss_topic = 100 * tf.reduce_sum(tf.log(1 + tf.exp(-10 * delta))) 


learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps,decay_rate, staircase=True)

l2_lambda=0.001 

# L_read= 100 * ( tf.nn.l2_loss((read_real - prediction_label1)) + tf.nn.l2_loss((read_fake - prediction_label2))  )

delta2 = read_real - read_fake
L_read= 100 * tf.reduce_sum(tf.log(1 + tf.exp(-10 * delta2))) 


L2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()  if 'bias' not in v.name]) * l2_lambda

# loss_val = 100*(loss_topic  + L_read)  + L2_reg

opti1 = tf.train.AdamOptimizer(1e-4 , beta1=0.5) 
opti2 = tf.train.AdamOptimizer(1e-4 , beta1=0.5) 
train_op1 = tf.contrib.layers.optimize_loss(L_read +  L2_reg, global_step=global_step,learning_rate=learning_rate, optimizer=opti1,clip_gradients=float(clip_gradients) , variables= theta_read)
train_op2 = tf.contrib.layers.optimize_loss(loss_topic +  L2_reg, global_step=global_step,learning_rate=learning_rate, optimizer=opti2,clip_gradients=float(clip_gradients), variables= theta_topic )

# ##########################################





config=tf.ConfigProto()
config.gpu_options.allow_growth=True

x_train = np.load("x_train_cnn.npy")
y_train= np.load("y_train_cnn.npy" )
x_val = np.load("x_val_cnn.npy" )
y_val = np.load("y_val_cnn.npy" )

sess = tf.Session(config = config)    
# sess = tf.Session()    
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
# coherency check


# similarity distance

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
            feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1 } 
            feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2 }
            feed_dict3 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input1 , input_x3: dummy_input2 } 
            feed_dict4 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1 } 
            curr_loss1,_=sess.run([loss_topic,train_op2  ],feed_dict1)
            curr_loss2,_=sess.run([L_read,train_op1 ],feed_dict2)
            curr_loss4,_=sess.run([loss_topic,train_op2  ],feed_dict4)
            curr_loss3,lr,_=sess.run([L_read,learning_rate,train_op1  ],feed_dict3)
            # loss,counter=loss+curr_loss1+curr_loss2+curr_loss3,counter+1
            counter=counter+1
            if counter %50==0:
                print("Epoch %d\t , Loss =%f " %(epoch , curr_loss1+curr_loss2 + curr_loss3 + curr_loss4))
        sess.run(epoch_increment)
        # print(epoch,validate_every,(epoch % validate_every==0))

# l2 and similarity distance 
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
            prediction_label1: ones , prediction_label2: ones } 
            feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2 ,\
            prediction_label1: ones , prediction_label2: zeros    }
            feed_dict3 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1 ,\
            prediction_label1: zeros , prediction_label2: ones }   
            curr_loss1,_,_=sess.run([loss_topic,train_op2 , train_op1  ],feed_dict1)
            curr_loss2,_=sess.run([L_read,train_op1 ],feed_dict2)
            curr_loss3,_,_ , lr=sess.run([loss_topic,train_op2 , train_op1 , learning_rate ],feed_dict3)
            # loss,counter=loss+curr_loss1+curr_loss2+curr_loss3,counter+1
            counter=counter+1
            if counter %50==0:
                print("Epoch %d\t , Loss =%f " %(epoch , curr_loss1+curr_loss2 + curr_loss3 ))
        sess.run(epoch_increment)
        # print(epoch,validate_every,(epoch % validate_every==0))



# new embedding
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
            feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1, \
            prediction_label1:ones ,prediction_label2:ones ,topic_label1 : ones , topic_label2:zeros } 
            feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2 , \
            prediction_label1: ones ,prediction_label2: zeros , topic_label1 : ones , topic_label2:ones }
            feed_dict3 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1, \
            prediction_label1:zeros ,prediction_label2:ones , topic_label1 : ones , topic_label2:zeros } 
            curr_loss1,_=sess.run([loss_val,train_op  ],feed_dict1)
            curr_loss2,_=sess.run([loss_val,train_op  ],feed_dict2)
            curr_loss3,lr,_=sess.run([loss_val,learning_rate,train_op ],feed_dict3)
            loss,counter=loss+curr_loss1+curr_loss2+curr_loss3,counter+1
            if counter %50==0:
                print("Epoch %d\t , Loss =%f " %(epoch , curr_loss1+curr_loss2+curr_loss3))
        sess.run(epoch_increment)
        # print(epoch,validate_every,(epoch % validate_every==0))



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
            feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1, \
            prediction_label1:ones ,prediction_label2:ones ,topic_label1 : ones , topic_label2:zeros } 
            feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2 , \
            prediction_label1: ones ,prediction_label2: zeros , topic_label1 : ones , topic_label2:ones }
            feed_dict3 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1, \
            prediction_label1:zeros ,prediction_label2:ones , topic_label1 : ones , topic_label2:zeros } 
            curr_loss1,_=sess.run([loss_val,train_op  ],feed_dict1)
            curr_loss2,_=sess.run([loss_val,train_op  ],feed_dict2)
            curr_loss3,lr,_=sess.run([loss_val,learning_rate,train_op ],feed_dict3)
            loss,counter=loss+curr_loss1+curr_loss2+curr_loss3,counter+1
            if counter %50==0:
                print("Epoch %d\t , Loss =%f " %(epoch , curr_loss1+curr_loss2+curr_loss3))
        sess.run(epoch_increment)
        # print(epoch,validate_every,(epoch % validate_every==0))





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
            feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1, \
            prediction_label1:ones ,prediction_label2:ones ,topic_label1 : ones , topic_label2:zeros } 
            feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2 , \
            prediction_label1: ones ,prediction_label2: zeros , topic_label1 : ones , topic_label2:ones }
            feed_dict3 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1, \
            prediction_label1:zeros ,prediction_label2:ones , topic_label1 : ones , topic_label2:zeros } 
            curr_loss1,_,_=sess.run([loss_VAL,train_op1 , train_op2 ],feed_dict1)
            curr_loss2,_,_=sess.run([loss_VAL,train_op1 , train_op2 ],feed_dict2)
            curr_loss3,lr,_,_=sess.run([loss_VAL,learning_rate,train_op2 , train_op1  ],feed_dict3)
            loss,counter=loss+curr_loss1+curr_loss2+curr_loss3,counter+1
            if counter %50==0:
                print("Epoch %d\t , Loss =%f " %(epoch , curr_loss1+curr_loss2+curr_loss3))
        sess.run(epoch_increment)
        # print(epoch,validate_every,(epoch % validate_every==0))

saver = tf.train.Saver()
saver.save(sess, "1opti_wo_sigmoid_L2/model.ckpt")


saver = tf.train.Saver()
saver.save(sess, "diffemb_l2_wosigmoid/model.ckpt")


saver = tf.train.Saver()
saver.save(sess, "softmax_sigmoid/model.ckpt")


saver = tf.train.Saver()
saver.save(sess, "noshare_l2_wosigmoid/model.ckpt")



saver = tf.train.Saver()
saver.save(sess, "noshare_diff/model.ckpt")



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








[A,B , C , D, E,F,G] = plot(Scores[0:7].T[indices])
legend([A,B, C, D, E,F,G], [ l1,l2 , l3 , l4 , l5 , l6, l7], loc=2)
show()


[A,B, C] = plot(Scores[3:6].T[indices])
legend([A,B, C], [ l1,l2 , l3], loc=2)
show()

