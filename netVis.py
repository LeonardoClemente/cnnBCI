# Neural Network visualization functions for Tensorflow
# Author : Cesar Leonardo Clemente Lopez   -  a01139264@itesm.mx / clemclem1991@gmail.com
import scipy.io as io
import numpy
import numpy as np
import tensorflow as tf
import time
import math as mt

import matplotlib as mp
# matplotlib inline
import matplotlib.pyplot as plt


def convLayerImg(units,directory,name):
	filters = units.shape[3]
	n = units.shape[1]
	m = units.shape[2]
	nFiltersx=5
	nFiltersy=int(5+filters/5)
	plt.figure(1, figsize=(10, nFiltersy*5), dpi=np.max([n,m]*2))
	for i in xrange(0,filters):
		plt.subplot(nFiltersy,nFiltersx,i+1)
		plt.subplots_adjust( hspace=.001)
		plt.title('Filter'+ str(i))
		plt.imshow(units[0,:,:,i], interpolation = "nearest", cmap = "gray")
	plt.savefig(directory + '/' + name + '.png')
