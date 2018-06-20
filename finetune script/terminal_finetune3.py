import tensorflow as tf
import numpy as np
import copy
import random
import numpy as np
import pickle


stored_data_path = 'priyank_vocab.p'

with open(stored_data_path, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    x = u.load()


gpu = 0 


w2v_embeddings, word_idx_map = x[0], x[1]

gpu2 = 0
start_rate = 1e-3
read_weight = 0.0001
topic_weight = 0.01
classifier_weight = 100 



model_select = 4

if model_select == 1 or model_select == 2:
    temp = 9 
else: 
    temp = 5 



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

# with tf.variable_scope('added_layer'):
output_z = tf.placeholder(tf.int32, [None, 5], name="input_x2")  # Z
dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
learning_rate = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, trainable=False, name="Global_Step")
epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
epoch_increment=tf.assign(epoch_step,tf.add(epoch_step,tf.constant(1)))
# decay_steps, decay_rate = decay_steps, decay_rate
initializer = tf.random_normal_initializer(stddev=0.1)


with tf.device('/gpu:%d'%(gpu) ):
    # initializer = tf.random_normal_initializer(stddev=0.1)
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



Theta_embedding = [v for v in tf.trainable_variables() if 'Embedding' in v.name]
Theta_read_filter = [v for v in tf.trainable_variables() if ('read:' in v.name or "read_:" in v.name)]
Theta_read_final = [v for v in tf.trainable_variables() if 'read_' in v.name and v not in Theta_read_filter ]


theta_read = [v for v in tf.trainable_variables() ]

pooled_outputs3 = []
pooled_outputs1 = []
pooled_outputs2 = []

with tf.device('/gpu:%d'%(gpu)):
    # temp = 5
    for i,filter_size in enumerate(filter_sizes):
        i , filter_size
        with tf.name_scope("convolution-pooling-%s" %filter_size):
            filter=tf.get_variable("filter-%s"%filter_size,[filter_size,embed_size,1,num_filters],initializer=initializer)
            conv1=tf.nn.conv2d(sentence_embeddings_expanded1, filter, strides=[1,1,1,1], padding="VALID", name="conv1" )
            conv3 =tf.nn.conv2d(sentence_embeddings_expanded3, filter, strides=[1,1,1,1], padding="VALID", name="conv3" )
            conv2 = tf.nn.conv2d(sentence_embeddings_expanded2, filter, strides=[1,1,1,1], padding="VALID", name="conv2" ) 
            b=tf.get_variable("b-%s"%filter_size,[num_filters])
            h1=tf.nn.relu(tf.nn.bias_add(conv1,b),"relu")
            h2=tf.nn.relu(tf.nn.bias_add(conv2,b),"relu")
            h3=tf.nn.relu(tf.nn.bias_add(conv3,b),"relu")
            pooled1=tf.nn.max_pool(h1, ksize=[1,sequence_length1-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool1")
            pooled2=tf.nn.max_pool(h2, ksize=[1,sequence_length2-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool2")
            pooled3=tf.nn.max_pool(h3, ksize=[1,sequence_length1-filter_size+1 - temp,1,1], strides=[1,1,1,1], padding='VALID',name="pool3")
            pooled_outputs1.append(pooled1)
            pooled_outputs2.append(pooled2)
            pooled_outputs3.append(pooled3)


Theta_L = [v for v in tf.trainable_variables() if 'filter-' in v.name or 'b-' in v.name  ] 


if gpu2 == 1:
    gpu = gpu +1



# gpu = gpu +1
with tf.device('/gpu:%d'%(gpu)):
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



Theta_projection = [v for v in tf.trainable_variables() if 'projection' in v.name]
Theta_final1 = [v for v in tf.trainable_variables() if 'final1' in v.name]
Theta_final2 = [v for v in tf.trainable_variables() if 'final2' in v.name]



    
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


config=tf.ConfigProto()
config.gpu_options.allow_growth=True

value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

H__1 =  tf.concat([h_pool_flat1 , h_pool_flat2 ],1)
H__expand1 =tf.expand_dims(H__1,-1)
H_1_max = tf.layers.max_pooling1d(   H__expand1, H__expand1.get_shape().as_list()[1] , 1 )
H_1_avg = tf.layers.average_pooling1d( H__expand1, H__expand1.get_shape().as_list()[1] , 1  )
H_1_avg = tf.reshape(H_1_avg , [-1 , 1]) 


H_class1 = tf.concat([D_logit_real , output_true , H_1_avg ] ,1)

# sess.run(epoch_step )
with tf.variable_scope('added_layer'):
    epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
    epoch_increment=tf.assign(epoch_step,tf.add(epoch_step,tf.constant(1)))
    learning_rate1  =tf.Variable(start_rate,trainable=False,name="lr")
    learning_rate2  =tf.Variable(start_rate / 100 ,trainable=False,name="lr")
    ratio=tf.Variable(32.0,trainable=False,name="Ratio")
    num_epochs_=tf.Variable(12.0,trainable=False,name="NO_Epoch_Step")
    cut  = num_epochs_ * 0.1 + 1
    cut = tf.round(cut)
    epoch_temp= epoch_step + 1 
    cond = tf.less_equal(epoch_temp , tf.cast(cut, tf.int32) )
    p = tf.where(cond , tf.cast(epoch_temp,tf.float32) / cut , 1-(tf.cast(epoch_temp,tf.float32) - cut)/(cut * (ratio-1)) )
    # lr_update = tf.assign(learning_rate, 1e-2 * ((1+ p * (ratio-1))/ ratio)     )
    learning_rate1 = start_rate * ((1+ p * (ratio-1))/ ratio)
    learning_rate2 = start_rate/100 * ((1+ p * (ratio-1))/ ratio)
    l_rate = tf.train.exponential_decay(start_rate, epoch_step, decay_steps,decay_rate, staircase=False)
    decay_steps, decay_rate = decay_steps, decay_rate
    with tf.device('/gpu:%d'%(gpu)):
        W_class1 = tf.get_variable("W_class1",shape=[202, 50],initializer=initializer) #[embed_size,label_size]
        b_class1 = tf.get_variable("b_class1",shape=[50])       #[label_size] #ADD 2017.06.09
        W_class2 = tf.get_variable("W_class2",shape=[50, 5],initializer=initializer) #[embed_size,label_size]
        b_class2 = tf.get_variable("b_class2",shape=[5])       #[label_size] #ADD 2017.06.09
        L1 = tf.matmul(H_class1,W_class1) + b_class1
        # batch_normed = tf.layers.batch_normalization(L1, training=True)
        # L1_output = tf.nn.dropout(batch_normed,keep_prob=dropout_keep_prob)
        L1_output = tf.nn.dropout(L1,keep_prob=dropout_keep_prob)
        L1_output = tf.nn.relu(L1_output,"relu")
        L2 = tf.matmul(L1_output,W_class2) + b_class2
        # L2_output = tf.nn.dropout(batch_normed2,keep_prob=dropout_keep_prob)
        L2_output = L2
        # L2_output = tf.nn.sigmoid(L2)

New_weights = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="added_layer" ) 
output_z = tf.placeholder(tf.float32, [None, 5], name="input_x2")  # Z

read_weight = 0.0001
topic_weight = 0.01
classifier_weight = 100 



with tf.variable_scope('optimizers_loss'):
    lr = learning_rate1
    opread = tf.train.AdamOptimizer(learning_rate2 / (10*10) , beta1=0.5) # Theta_read_final
    opread2 = tf.train.AdamOptimizer(learning_rate2 / (10*10 *10) , beta1=0.5) #Theta_read_filter
    opread3 = tf.train.AdamOptimizer(learning_rate2 / (10*10*10*10) , beta1=0.5)  #Theta_embedding 
    optopic = tf.train.AdamOptimizer(lr , beta1=0.5)  #Theta_final2
    optopic2 = tf.train.AdamOptimizer(lr/2.6 , beta1=0.5) #Theta_final1
    optopic3 = tf.train.AdamOptimizer(lr/(2.6*2.6) , beta1=0.5) #Theta_projection
    optopic4 = tf.train.AdamOptimizer(lr/(2.6*2.6*2.6) , beta1=0.5) #Theta_L
    optopic5 = tf.train.AdamOptimizer(lr/(2.6*2.6*2.6*2.6) , beta1=0.5) # Theta_embedding
    optopic6 = tf.train.AdamOptimizer(1e-4 , beta1=0.5) 
    l2_lambda=0.001 
    delta2 = read_real - read_fake
    L_read= read_weight * tf.reduce_sum(tf.log(1 + tf.exp(-10 * delta2))) 
    loss_read_total = L_read + tf.add_n([tf.nn.l2_loss(v) for v in theta_read if 'bias' not in v.name]) * l2_lambda
    delta1 = D_real - D_fake     
    if model_select == 1 or model_select == 2 :
        loss_topic = topic_weight * tf.reduce_sum(tf.log(1 + tf.exp(-1 * prediction_label1  * delta1))) 
    elif model_select == 4 or model_select == 5:
        confirm =  0
        if confirm == 0 :
            condition1 = tf.greater(read_fake, 0.2)
            condition2 = tf.greater(D_real, prediction_label1)
            val_main1  = tf.where(condition2, delta1  , delta1/400 )
            loss_topic1 = tf.log(1 + tf.exp(-10  * val_main1))
            penalty  = tf.where(condition1, 10 * (loss_topic1 ), loss_topic1)
            loss_topic = tf.reduce_sum(penalty) ; 
        else:
            loss_topic = topic_weight * tf.reduce_sum(tf.log(1 + tf.exp(-1 * prediction_label1  * delta1))) 
    loss_topic_total = loss_topic + tf.add_n([tf.nn.l2_loss(v) for v in theta_topic if 'bias' not in v.name]) * l2_lambda
    # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = output_z, logits = L2_output)
    losses = tf.nn.softmax_cross_entropy_with_logits(labels = output_z, logits = L2_output)
    Loss_total = classifier_weight* tf.reduce_sum(losses) + loss_topic_total 
    train_op1 = tf.contrib.layers.optimize_loss(loss_read_total, global_step=epoch_step,learning_rate=learning_rate2 / (10*10), \
      optimizer=opread,clip_gradients=float(clip_gradients) , variables= Theta_read_final , increment_global_step=False)
    train_op2 = tf.contrib.layers.optimize_loss(loss_read_total, global_step=epoch_step,learning_rate=learning_rate2 /(10*10*10), \
      optimizer=opread2,clip_gradients=float(clip_gradients) , variables= Theta_read_filter , increment_global_step=False )
    train_op3 = tf.contrib.layers.optimize_loss(loss_read_total, global_step=epoch_step,learning_rate=learning_rate2 /(10*10*10*10), \
      optimizer=opread3,clip_gradients=float(clip_gradients) , variables= Theta_embedding , increment_global_step=False)
    train_op4 = tf.contrib.layers.optimize_loss(Loss_total, global_step=epoch_step,learning_rate=lr, \
      optimizer=optopic,clip_gradients=float(clip_gradients) , variables= Theta_final2 , increment_global_step=False)
    train_op5 = tf.contrib.layers.optimize_loss(Loss_total, global_step=epoch_step,learning_rate=lr/2.6, \
      optimizer=optopic2,clip_gradients=float(clip_gradients) , variables= Theta_final1 , increment_global_step=False)
    train_op6 = tf.contrib.layers.optimize_loss(Loss_total, global_step=epoch_step,learning_rate=lr/(2.6*2.6), \
      optimizer=optopic3,clip_gradients=float(clip_gradients) , variables= Theta_projection , increment_global_step=False)
    train_op7 = tf.contrib.layers.optimize_loss(Loss_total, global_step=epoch_step,learning_rate=lr/(2.6*2.6 * 2.6), \
      optimizer=optopic4,clip_gradients=float(clip_gradients) , variables= Theta_L , increment_global_step=False)
    train_op8 = tf.contrib.layers.optimize_loss(Loss_total, global_step=epoch_step,learning_rate=lr /(2.6*2.6 * 2.6 * 2.6), \
      optimizer=optopic5,clip_gradients=float(clip_gradients) , variables= Theta_embedding , increment_global_step=False)
    train_op_main = tf.contrib.layers.optimize_loss(tf.reduce_sum(Loss_total), global_step=epoch_step,learning_rate=l_rate, \
        optimizer=optopic6,clip_gradients=float(clip_gradients),  variables= New_weights , increment_global_step=False)
    TOPIC = tf.group(train_op_main, train_op8 , train_op7 , train_op6 , train_op5, train_op4)
    READ = tf.group(train_op1, train_op2 , train_op3)


trainX = np.load("finetune_trainX.npy"  )
trainY =np.load( "finetune_trainY.npy"   )
trainZ = np.load("finetune_trainZ.npy"   )
testX = np.load( "finetune_testX.npy" )
testY = np.load("finetune_testY.npy"  )
testZ = np.load( "finetune_testZ.npy"  )

num_epochs = 12 
	
value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


sess = tf.Session(config = config)    
saver = tf.train.Saver()

model_2_variables_list = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="added_layer" ) + \
    tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES,  scope="optimizers_loss" )

sess.run(tf.variables_initializer(model_2_variables_list))


# #####################################################
folder_name = "fine_tuned_11"
saver = tf.train.Saver()
saver.restore(sess, folder_name +"/model.ckpt")

folder_name = "fine_tuned_44"
saver = tf.train.Saver()
saver.restore(sess, folder_name +"/model.ckpt")

folder_name = "fine_tuned_121"
saver = tf.train.Saver()
saver.restore(sess, folder_name +"/model.ckpt")

folder_name = "fine_tuned_L2_11"
saver = tf.train.Saver()
saver.restore(sess, folder_name +"/model.ckpt")

folder_name = "fine_tuned_class_1"
saver = tf.train.Saver()
saver.restore(sess, folder_name +"/model.ckpt")

folder_name = "fine_tuned_complete_1"
saver = tf.train.Saver()
saver.restore(sess, folder_name +"/model.ckpt")

folder_name = "fine_tuned_complete_4"
saver = tf.train.Saver()
saver.restore(sess, folder_name +"/model.ckpt")

# #####################################################



number_of_training_data=len(trainX)
batch_size = 30
temp = copy.deepcopy(trainY)
Tens = np.array([[10.0] for i in range(batch_size)])
SCORE_ZERO = [[1 , 0 , 0 , 0, 0 ] for i in range(batch_size)]
SCORE_ONE = [[0 , 0 , 0 , 0, 1 ] for i in range(batch_size)]
# prediction_label2 = tf.placeholder(tf.float32, [None, 1], name="input_read2")  # Y



sevens = np.array([[0.7] for i in range(batch_size)])
nines = np.array([[0.95] for i in range(batch_size)])
fives = np.array([[0.5] for i in range(batch_size)])


print(  sess.run([epoch_step , learning_rate1  , learning_rate2 ]))



for epoch in range(0,10):
        epoch = epoch + 1 
        print(  sess.run([epoch_step , learning_rate1  , learning_rate2 ]))
        loss, counter =  0.0, 0
        for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
            dummy_input2 = temp[start:end]
            print(counter , end='\r')
            for i in range(len(dummy_input2)):
                idx = np.nonzero(dummy_input2[i])
                dummy_input2[i][idx] = np.random.permutation(dummy_input2[i][idx])    
            ram = random.sample(range(number_of_training_data), batch_size)
            dummy_input1 =trainY[ram]
            # "======="
            if model_select == 1 or model_select == 2 :  
                feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2 ,\
                    prediction_label1: Tens  }
                feed_dict3 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input1 , input_x3: dummy_input2 ,\
                    prediction_label1: Tens  }    
                feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1 ,\
                    prediction_label1: Tens , output_z: trainZ[start:end] } 
                feed_dict4 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1 ,\
                    prediction_label1: Tens , output_z: trainZ[start:end]  } 
            else:
                feed_dict2 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input2 ,\
                    prediction_label1: Tens  }
                feed_dict3 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input1 , input_x3: dummy_input2 ,\
                    prediction_label1: Tens  }    
                feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1 ,\
                    prediction_label1: sevens , prediction_label2: nines  , output_z: trainZ[start:end] } 
                feed_dict4 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1 ,\
                    prediction_label1: fives , prediction_label2: nines , output_z: trainZ[start:end]  }            
            # "======="
            # sess.run([epoch_step ])
            curr_loss1,_ =sess.run([L_read,READ ],feed_dict2)
            # sess.run([epoch_step ])
            curr_loss1_temp,_=sess.run([L_read,READ ],feed_dict3)     
            # sess.run([epoch_step ])
            curr_loss1 += curr_loss1_temp 
            curr_loss3 , curr_loss2,_ =sess.run([loss_topic ,losses ,TOPIC  ],feed_dict4)
            # sess.run([epoch_step ])
            curr_loss3_temp, curr_loss2_temp,_ =sess.run([loss_topic ,losses ,TOPIC  ],feed_dict1)   
            # sess.run([epoch_step ])
            curr_loss2 += curr_loss2_temp
            curr_loss3 += curr_loss3_temp
            counter=counter+1
            if counter %50==0:
                print("Epoch %d\t"%(epoch), "read1:" , curr_loss1 , "\tTOPIC:" , curr_loss3  ,"\tCLASSIFIER:" , np.sum(curr_loss2))
        sess.run([epoch_increment ])

    

# import os    
# folder_name  = "fine_tuned%d" %(model_select)
# if not os.path.isdir(folder_name) :
#     os.makedirs(folder_name)


# saver = tf.train.Saver()
# saver.save(sess, folder_name +"/model.ckpt")
  

# train_op_main
epoch = -1
while True :
        epoch = epoch + 1  
        print(  sess.run([epoch_step , learning_rate1  , learning_rate2 ]))
        loss, counter =  0.0, 0
        for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
            # counter
            dummy_input2 = temp[start:end]
            print(counter , end='\r')
            for i in range(len(dummy_input2)):
                idx = np.nonzero(dummy_input2[i])
                dummy_input2[i][idx] = np.random.permutation(dummy_input2[i][idx])    
            ram = random.sample(range(number_of_training_data), batch_size)
            dummy_input1 =trainY[ram] 
            if model_select == 1 or model_select == 2 :
                    feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1 ,\
                        prediction_label1: Tens , output_z: trainZ[start:end] } 
                    feed_dict4 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1 ,\
                        prediction_label1: Tens , output_z: trainZ[start:end]  } 
            else:
                feed_dict1 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input1 ,\
                    prediction_label1: sevens , prediction_label2: nines  , output_z: trainZ[start:end] } 
                feed_dict4 = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: dummy_input2 , input_x3: dummy_input1 ,\
                    prediction_label1: fives , prediction_label2: nines , output_z: trainZ[start:end]  } 
            curr_loss1 , curr_loss3 , curr_loss2,_ = sess.run([L_read, loss_topic ,losses , train_op_main  ],feed_dict4)
            curr_loss3_temp, curr_loss2_temp,_ =sess.run([loss_topic ,losses ,train_op_main  ],feed_dict1)   
            curr_loss2 += curr_loss2_temp
            curr_loss3 += curr_loss3_temp
            counter=counter+1
            if counter %50==0:
                print("Epoch %d\t"%(epoch), "read1:" , curr_loss1 , "\tTOPIC:" , curr_loss3  ,"\tCLASSIFIER:" , np.sum(curr_loss2))
                    # print( , Loss =%f " %(epoch , curr_loss1+curr_loss2  ))      
        sess.run([epoch_increment ])



save =  input('Want to save?')
while(save != '1' and save != '0'):
    save =  input('Want to save?')
save = int(save)
if save == 1:
    folder_name  = "fine_tuned%d" %(model_select)
    if not os.path.isdir(folder_name) :
        os.makedirs(folder_name)
    saver = tf.train.Saver()
    saver.save(sess, folder_name +"/model.ckpt")
    sys.exit()
else:
    sys.exit()



import os 

folder_name  = "fine_tuned_complete_4" 
if not os.path.isdir(folder_name) :
    os.makedirs(folder_name)


saver = tf.train.Saver()
saver.save(sess, folder_name +"/model.ckpt")