# Author : Cesar Leonardo Clemente Lopez @ ITESM, Campus MTY.
#CNN that tries to replicate results from paper "Decoding EEG and LFP signals using deep learning: Heading Truenorth".
# Data reference: Nurse ES, Karoly PJ, Grayden DB, Freestone DR (2015) A Generalizable Brain-Computer Interface (BCI)
# Using Machine Learning for Feature Discovery. PLoS ONE 10(6): e0131328. doi: 10.1371/journal.pone.0131328

import scipy.io as io
import numpy
import tensorflow as tf
import numpy as np
from eegClass import eegData as datasets
import time
import random
import matplotlib as mp
# matplotlib inline
import matplotlib.pyplot as plt
import pickle
import netVis as nv
import math


# Hyperparameters
nChannels = 46
nSamples = 100

classes = 2

wConv1  = [10,5,1,40] # kernel size, kernel size, input dimension (maps), output dimension;
wConv2  = [5,5, 40,100] # kernel size, kernel size, input dimension (maps), output dimension;
fcSize = [12*25*100,300] 
outsideLayer = [fcSize[1],2]


# Data directories (change it to yours)
nom = 'raw2'
tensorboardDir = '/Users/leonardo/Desktop/IBMcnn/ibm/tb' 
imgDir='/Users/leonardo/Desktop/IBMcnn/ibm/images'
singleKernelDir = '/Users/leonardo/Desktop/IBMcnn/ibm/images/singleKernel.png'
weightDir = '/Users/leonardo/Desktop/IBMcnn/ibm/weights'
txtFiles = '/Users/leonardo/Desktop/IBMcnn/ibm/txt'

f = open('/Users/leonardo/Desktop/IBMcnn/ibm/txt'+'/'+ nom+'.txt','w')
f.write('nChannels {0} \n nSamples {1} \n Classes {2} \n first conv parameters {3} \n second conv parameters {4} \n FC size  {5} \n  output layer size {6}'.format(nChannels, nSamples, classes, wConv1, wConv2, fcSize, outsideLayer))
f.close
# Loading samples (SEE eegClass)
ds = datasets()
ds.validationSet()
sess=tf.InteractiveSession()



def weight_variable(shape, nom):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = nom )
    
def bias_variable(shape,nom):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name= nom)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        
def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    # tf.histogram_summary(name, var)


# Visualization functions

def getActivations(layer,image):
	units = layer.eval(session = sess, feed_dict = {signals:image,keep_prob:1.0})
	return units

def plotLayer(units):
	filters = units.shape[3]
	plt.figure(1,figsize=(15,30))
	for i in xrange(0,filters):
		plt.subplot(7,6,i+1)
		plt.title('Filter'+ str(i))
		plt.imshow(units[0,:,:,i], interpolation = "nearest", cmap = "gray")
	plt.savefig(singleKernelDir)

'''The first two dimensions are the patch size, the next is the number of input
channels, and the last is the number of output channels'''

with tf.name_scope('input'):
	signals = tf.placeholder(tf.float32, shape=[None, nChannels*nSamples])
	y_ = tf.placeholder(tf.float32, shape=[None, classes])
	image = tf.reshape(signals, [-1, nChannels, nSamples, 1])
	


with tf.name_scope('firstConv'):
	W_conv1 = weight_variable(wConv1, 'w1')
	variable_summaries(W_conv1,'W1')
	b_conv1 = bias_variable([wConv1[3]],'b1')
	h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)




with tf.name_scope('secondConv'):
	W_conv2 = weight_variable(wConv2, 'w2')
	b_conv2 = bias_variable([wConv2[3]], 'b2')
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)


# Fully connected network (500 neurons)
with tf.name_scope('fullyConnected'):
	W_fc1 = weight_variable(fcSize,'w3')  # Might overfit
	b_fc1 = bias_variable([fcSize[1]], 'b3')
	h_pool2_flat = tf.reshape(h_pool2, [-1, fcSize[0]])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout

keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Output LAYER
with tf.name_scope('output'):
	W_fc2 = weight_variable(outsideLayer, 'w4')
	b_fc2 = bias_variable([outsideLayer[1]], 'b5')
	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Cost function
with tf.name_scope('xent'):
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0e5)), reduction_indices=[1]))
	
	


#train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


# Predictions
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))



accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


accSummary = tf.scalar_summary("Accuracy", accuracy)
costSummary = tf.scalar_summary("Cost", cross_entropy)
predSummary = tf.histogram_summary("Prediction", y_conv)
# Tensorboard data
merged=tf.merge_all_summaries()
train_writer=tf.train.SummaryWriter(tensorboardDir)
saver=tf.train.Saver()
coord=tf.train.Coordinator()


sess.run(tf.initialize_all_variables())
train_writer.add_graph(sess.graph)



rate = 0.1
print numpy.shape(ds.vsSamples)
for i in range(500):
	if i%100 == 0 and i > 0 :
		rate= rate/10
	#ds.nextBatch()
	ds.nextBatchRandom()
	if random.random() > .6:
		ds.addNoise()
	summary, _, acc,ent = sess.run([merged, train_step, accuracy,cross_entropy], feed_dict = {signals: ds.batch, y_: ds.batchTargets, keep_prob : 0.5, learning_rate : rate} )
	train_writer.add_summary(summary,i)
	if i%10 == 0:
		acc = sess.run([accuracy], feed_dict = {signals: ds.vsSamples, y_: ds.vsTargets, keep_prob : 1} )
		print('Accuracy is {0}'.format(acc))
		print('Entropy = {0}'.format(ent))
    	
    	
# Saving weight data
w1,w2,w3,w4 = sess.run([W_conv1,W_conv2,W_fc1,W_fc2], feed_dict = {signals: ds.vsSamples, y_: ds.vsTargets, keep_prob : 1} )  
  	 	
w1F= open(weightDir + '/cnnTestw1'+nom+'.pickle','w')
w2F=open(weightDir + '/cnnTestw2+'+nom+'.pickle','w')
w3F=open(weightDir + '/cnnTestw3'+nom+'.pickle','w')
w4F=open(weightDir + '/cnnTestw4'+nom+'.pickle','w')
pickle.dump(w1,w1F)
pickle.dump(w2,w2F)
pickle.dump(w3,w3F)
pickle.dump(w4,w4F)
w1F.close()
w2F.close()
w3F.close()
w4F.close()


# Saving some images
units1=getActivations(h_conv1,numpy.reshape(ds.vsSamples[0,:],[1,nChannels*nSamples]))
units2=getActivations(h_conv2,numpy.reshape(ds.vsSamples[0,:],[1,nChannels*nSamples]))

nv.convLayerImg(units1,imgDir,'/layer1'+nom)
nv.convLayerImg(units2,imgDir,'/layer2'+nom)
'''
dpi=529
plt.figure(2,figsize=(1,1),dpi=46)
plt.imshow(np.reshape(ds.vsSamples[0,:],[46,46]), interpolation = "nearest", cmap="gray")
#plt.show()
plt.savefig(imgDir'/testplot2.png')
'''
    # train_step.run(feed_dict={signals: data[:, :, i], y_: yy, keep_prob : 0.5})
