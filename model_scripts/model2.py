
import tensorflow as tf
import numpy as np

import pickle
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def do_eval(sess,evalX,evalY):
    number_examples=len(evalX)
    batch_size=1
    corr1 = 0
    corr3 = 0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {input_x1: evalX[start:end], input_x2: evalY[start:end] , prediction_label:[1 for i in range(batch_size)], dropout_keep_prob: 1.0}
        pred= sess.run([D_prob],feed_dict)#curr_eval_acc--->textCNN.accuracy
         

def get_label_using_logits(pred_index,top_number=3):
    index_list=np.argsort(pred_index)[-top_number:]
    index_list=index_list[::-1]
    return index_list



Main_Voc = load_obj("Main_Voc" )

x_train = np.load("x_train.npy"  )
y_train = np.load("y_train.npy"  )
x_val = np.load("x_val.npy" )
y_val = np.load("y_val.npy" )


batch_size = 10
sequence_length = 75
vocab_size = len(Vocabulary)

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


Final_layer = 50
# add placeholder (X,label)

sequence_length1 = 49
sequence_length2= 2742

input_x3 = tf.placeholder(tf.int32, [None, sequence_length1], name="input_x3")  # X
input_x1 = tf.placeholder(tf.int32, [None, sequence_length1], name="input_x1")  # X
input_x2 = tf.placeholder(tf.int32, [None, sequence_length2], name="input_x2")  # X
prediction_label = tf.placeholder(tf.float32, [None, 1], name="input_x2")  # Y
dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

global_step = tf.Variable(0, trainable=False, name="Global_Step")
epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
epoch_increment=tf.assign(epoch_step,tf.add(epoch_step,tf.constant(1)))
decay_steps, decay_rate = decay_steps, decay_rate


embedding_matrix = np.load("embedding_matrix50.npy")
embedding_matrix =  np.asarray(embedding_matrix ,dtype=np.float32)


initializer = tf.random_normal_initializer(stddev=0.1)

Embedding = tf.get_variable("Embedding",initializer=embedding_matrix) #[vocab_size,embed_size] tf.random_uniform([vocab_size, embed_size],-1.0,1.0)
W_projection1 = tf.get_variable("W_projection1",shape=[num_filters_total, Final_layer],initializer=initializer) #[embed_size,label_size]
b_projection1 = tf.get_variable("b_projection1",shape=[Final_layer])       #[label_size] #ADD 2017.06.09


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
embed_size = 50

for i,filter_size in enumerate(filter_sizes):
    i , filter_size
    with tf.name_scope("convolution-pooling-%s" %filter_size):
        filter=tf.get_variable("filter-%s"%filter_size,[filter_size,embed_size,1,num_filters],initializer=initializer)
        conv1=tf.nn.conv2d(sentence_embeddings_expanded1, filter, strides=[1,1,1,1], padding="VALID", name="conv" )
        conv3=tf.nn.conv2d(sentence_embeddings_expanded3, filter, strides=[1,1,1,1], padding="VALID", name="conv3" )
        b=tf.get_variable("b-%s"%filter_size,[num_filters])
        h1=tf.nn.relu(tf.nn.bias_add(conv1,b),"relu")
        h3=tf.nn.relu(tf.nn.bias_add(conv3,b),"relu")
        pooled1=tf.nn.max_pool(h1, ksize=[1,sequence_length1-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool1")
        pooled3=tf.nn.max_pool(h3, ksize=[1,sequence_length1-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool3")
        pooled_outputs1.append(pooled1)
        pooled_outputs3.append(pooled3)
    with tf.name_scope("convolution-pooling2-%s" %filter_size):
        filter=tf.get_variable("filter2-%s"%filter_size,[filter_size,embed_size,1,num_filters],initializer=initializer)
        conv=tf.nn.conv2d(sentence_embeddings_expanded2, filter, strides=[1,1,1,1], padding="VALID",name="conv2" )
        b=tf.get_variable("b2-%s"%filter_size,[num_filters])
        h=tf.nn.relu(tf.nn.bias_add(conv,b),"relu2")
        pooled=tf.nn.max_pool(h, ksize=[1,sequence_length2-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool2")
        pooled_outputs2.append(pooled)


h_pool1=tf.concat(pooled_outputs1,3)
h_pool3=tf.concat(pooled_outputs3,3)
h_pool_flat1=tf.reshape(h_pool1,[-1,num_filters_total])
h_pool_flat3=tf.reshape(h_pool3,[-1,num_filters_total])

h_pool2=tf.concat(pooled_outputs2,3)
h_pool_flat2=tf.reshape(h_pool2,[-1,num_filters_total])


with tf.name_scope("dropout"):
            h_drop1=tf.nn.dropout(h_pool_flat1,keep_prob=dropout_keep_prob)
            h_drop3=tf.nn.dropout(h_pool_flat3,keep_prob=dropout_keep_prob)
            h_drop2=tf.nn.dropout(h_pool_flat2,keep_prob=dropout_keep_prob)

with tf.name_scope("output"):
            logits1 = tf.matmul(h_drop1,W_projection1) + b_projection1  #shape:[None, num_classes]==tf.matmul([None,embed_size],[embed_size,num_classes])
            logits2 = tf.matmul(h_drop2,W_projection2) + b_projection2
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



l2_lambda=0.001 



with tf.name_scope("loss"):
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=prediction_label, logits=D_logit);#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).
            # losses=tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch
            # loss=tf.reduce_mean(losses)         #shape=().   average loss in the batch
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss_val=100* (G_loss + D_loss) +l2_losses
        

learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps,decay_rate, staircase=True)


opti = tf.train.AdamOptimizer(1e-4 , beta1=0.5)

# opti = tf.train.RMSPropOptimizer(1e-3, momentum=0.4, decay=0.2)

train_op = tf.contrib.layers.optimize_loss(loss_val, global_step=global_step,learning_rate=learning_rate, optimizer=opti,clip_gradients=float(clip_gradients))

config=tf.ConfigProto()
config.gpu_options.allow_growth=True


x_train = np.load("x_train.npy")
y_train= np.load("y_train.npy" )
x_val = np.load("x_val.npy" )
y_val = np.load("y_val.npy" )


import random

sess = tf.Session(config = config)    
sess.run(tf.global_variables_initializer())
curr_epoch=sess.run(epoch_step)
number_of_training_data=len(x_train)
trainX = x_train
trainY = y_train
testX = x_val
testY = y_val

for epoch in range(curr_epoch,num_epochs):
        loss, counter =  0.0, 0
        for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
            ram = random.sample(range(number_of_training_data), batch_size)
            dummy_input =trainY[ram]
            feed_dict = {input_x2: trainX[start:end],dropout_keep_prob: 0.5 , input_x1: trainY[start:end] , input_x3: dummy_input}
            curr_loss,lr,_=sess.run([loss_val,learning_rate,train_op],feed_dict)
            loss,counter=loss+curr_loss,counter+1
            if counter %50==0:
                print("Epoch %d\t , Loss =%f " %(epoch , curr_loss))
        sess.run(epoch_increment)
        print(epoch,validate_every,(epoch % validate_every==0))
        # if epoch % validate_every==0:
        #     do_eval(sess,testX,testY)
            # save model to checkpoint
            # save_path="checkpoint"+"model.ckpt"
            # saver.save(sess,save_path,global_step=epoch)


no sharing , cross entropy loss : 

do_eval(sess,testX,testY)


saver = tf.train.Saver()

saver.restore(sess, "model2/model.ckpt")


D_real, D_logit_real 
D_fake, D_logit_fake

no sharing , cross entropy loss : 


testX = x_val
testY = y_val
number_examples=len(testX)
batch_size = 50


from rouge import Rouge 
rouge = Rouge()

S_G = []
S_G_S = []
S_GA = []
S_GA_S = []

S_R_l_G = []
S_R_l_GA = []

S_R_1_G = []
S_R_1_GA = []

S_R_2_G = []
S_R_2_GA = []


for start, end in zip(range(0, number_examples, batch_size),range(batch_size, number_examples, batch_size)):
            ram = random.sample(range(number_examples), batch_size)
            dummy_input = testY[ram]
            feed_dict = {input_x2: testX[start:end],dropout_keep_prob: 0.5 , input_x1: testY[start:end] , input_x3: dummy_input}
            pred1,pred2,pred3,pred4=sess.run([D_real,D_logit_real,D_fake , D_logit_fake],feed_dict)
            pred1 = pred1.tolist()
            S_G_S += [j[0] for j in pred1]
            pred2= pred2.tolist()
            S_G += [j[0] for j in pred2]
            pred3 = pred3.tolist()
            S_GA_S += [j[0] for j in pred3]
            pred4 = pred4.tolist()
            # if isinstance(pred4, list):
            S_GA += [j[0] for j in pred4]
            for i in range(batch_size):
                Ref = testX[start + i][np.where(testX[start + i] > 0 )[0]].tolist()            
                summary = testY[start + i][np.where(testY[start + i] > 0 )[0]].tolist()            
                Garbage = dummy_input[i][np.where(dummy_input[i] > 0 )[0]].tolist()            
                headline = [str(d) for d in summary]
                headline = " ".join(headline)
                body = [str(d) for d in Ref]
                body = " ".join(body)
                Garbage_summary = [str(d) for d in Garbage]
                Garbage_summary = " ".join(Garbage_summary)
                scores = rouge.get_scores(body, headline)
                S_R_l_G += [scores[0]['rouge-l']['r']]
                S_R_1_G  += [scores[0]['rouge-1']['r']]
                S_R_2_G += [scores[0]['rouge-2']['r']]
                scores = rouge.get_scores(body, Garbage_summary)
                S_R_l_GA += [scores[0]['rouge-l']['r']]
                S_R_1_GA  += [scores[0]['rouge-1']['r']]
                S_R_2_GA += [scores[0]['rouge-2']['r']]

len(S_G )     
len(S_GA )     
len(S_G_S )     
len(S_G )     
len(S_GA_S )     

len(S_R_l_G )     
len(S_R_l_GA )     

len(S_R_1_G )     
len(S_R_1_GA )

len(S_R_2_G )     
len(S_R_2_GA )

S_G = np.array(S_G)
S_G_S = np.array(S_G_S)
S_GA = np.array(S_GA)
S_GA_S = np.array(S_GA_S)

S_R_l_G = np.array(S_R_l_G)
S_R_l_GA = np.array(S_R_l_GA)

S_R_1_G = np.array(S_R_1_G)
S_R_1_GA = np.array(S_R_1_GA)

S_R_2_G = np.array(S_R_2_G)
S_R_2_GA = np.array(S_R_2_GA)



np.save("model2_golden_without_signoid.npy" , S_G_S)
np.save("model2_garbage_without_signoid.npy" , S_GA_S)


np.save("model2_golden.npy" , S_G_S)
np.save("model2_garbage.npy" , S_GA_S)

np.save("model2_Rl.npy" , S_R_l_G)
np.save("model2_Rl_garbage.npy" , S_R_l_GA)

np.save("model2_R1.npy" , S_R_1_G)
np.save("model2_R1_garbage.npy" , S_R_1_GA)

np.save("model2_R2.npy" , S_R_2_G)
np.save("model2_R2_garbage.npy" , S_R_2_GA)


S_G.shape
S_G_S.shape
S_GA.shape
S_GA_S.shape

S_R_l_G.shape
S_R_l_GA.shape

S_R_1_G.shape
S_R_1_GA.shape

S_R_2_G.shape
S_R_2_GA.shape



A = S_G.reshape(1850,1)
B=  S_GA.reshape(1,1850)
C = np.concatenate((A, B.T), axis=1)
np.savetxt('Model1.csv', C, delimiter=',', fmt='%s')

A = S_G_S.reshape(1850,1)
B=  S_GA_S.reshape(1,1850)
C = np.concatenate((A, B.T), axis=1)
np.savetxt('Model2.csv', C, delimiter=',', fmt='%s')


A = S_R_l_G.reshape(1850,1)
B=  S_R_l_GA.reshape(1,1850)
C = np.concatenate((A, B.T), axis=1)
np.savetxt('Rouge-l.csv', C, delimiter=',', fmt='%s')

A = S_R_1_G.reshape(1850,1)
B=  S_R_1_GA.reshape(1,1850)
C = np.concatenate((A, B.T), axis=1)
np.savetxt('Rouge-1.csv', C, delimiter=',', fmt='%s')

A = S_R_2_G.reshape(1850,1)
B=  S_R_2_GA.reshape(1,1850)
C = np.concatenate((A, B.T), axis=1)
np.savetxt('Rouge-2.csv', C, delimiter=',', fmt='%s')

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 1, 1850)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')