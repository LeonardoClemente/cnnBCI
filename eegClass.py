import scipy.io as io
import numpy
import matplotlib as mp
# fr%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math as mt
import time

class eegData:
	def __init__(self):
	
	
		# Experiment data
		self.nChannels = 46
		self.sampling = 100
		self.nClasses = 2
		# Loading data
		'''
		dataMat = io.loadmat('haufeDataND.mat') # Path to your data matrix
		targetsMat =  io.loadmat('haufeScoresND.mat')
		dataPre = dataMat['haufeDataND'] 
		targetsPre =   targetsMat['haufeScoresND']
		
		'''
		dataMat = io.loadmat('ibmDataRaw.mat') # Path to your data matrix
		targetsMat =  io.loadmat('ibmScores.mat')
		dataPre = dataMat['ibmData'] 
		targetsPre =   targetsMat['ibmScores']
		
		''' HAUFE 
		print np.shape(dataPre)
		# 1-hot-encoding
		indices = []
		samples=[]
		targets=[]
		for i,v in enumerate(targetsPre[:]):
			k = numpy.zeros([1,self.nClasses])
			k[0,v]=1
			targets.append(k)
			samples.append(dataPre[i,:])
			indices.append(i)
		# indices = [index for index, value in enumerate(targetsPre[:][0]) if value > 1 ];	
		self.nSamples = numpy.size(indices)
		self.targets = numpy.vstack(targets)
		self.samples =  numpy.vstack(samples)
		
		print np.shape(self.samples), np.shape(self.targets)
		time.sleep(10)
		
		'''
		print np.shape(dataPre[0,:])
		# 1-hot-encoding
		indices = []
		samples=[]
		targets=[]
		for i,v in enumerate(targetsPre[0,:]):
			if v < 3:
				indices.append(i)
				k = numpy.zeros([1,self.nClasses])
				k[0,v-1]=1
				targets.append(k)
				samples.append(dataPre[i,:])
		# indices = [index for index, value in enumerate(targetsPre[:][0]) if value > 1 ];	
		self.nSamples = numpy.size(indices)
		self.targets = numpy.vstack(targets)
		self.samples =  numpy.vstack(samples)
		
		# OPTIONAL :  mean substracting and normalizing
		
		
		
		print  'hola',numpy.shape(self.targets)
		# Batch training variables
		self.batchSize = 160;				
		self.batchCounter= 0;
		self.batch = numpy.zeros([self.batchSize, self.nChannels*self.sampling])
		self.batchTargets = numpy.zeros([self.batchSize,2])
		
	def validationSet(self,size=180):
		self.vsSize = size
		self.vsSamples = self.samples[0:size,:]
		self.vsTargets = self.targets[0:size,:]
		self.tsSamples = self.samples[size:self.nSamples,:]
		self.tsTargets = self.targets[size:self.nSamples,:]
		self.tsSize= self.nSamples-size
		
	def nextBatch(self):
		s = self.batchSize
		f = (self.batchCounter + s) - self.tsSize
		if  f > 0 :
			self.batch[0:s-f-1,:] = self.tsSamples[self.batchCounter:self.batchCounter+(s-f-1),:]
			self.batch[s-f:s] = self.tsSamples[0:f,:]
			self.batchTargets[0:s-f-1,:] = self.tsTargets[self.batchCounter:self.batchCounter+(s-f-1),:]
			self.batchTargets[s-f:s] = self.tsTargets[0:f,:]
			self.batchCounter = f
		else :
			self.batch[:,:]=self.tsSamples[self.batchCounter:self.batchCounter+s,:]
			self.batchTargets[:,:]=self.tsTargets[self.batchCounter:self.batchCounter+s,:]
			self.batchCounter = self.batchCounter + s
	
	def setCounter(self,set=0):
		self.batchCounter = set
		
	def nextBatchRandom(self):
		indices = np.random.randint(self.tsSize,size=(self.batchSize))
		self.batch[:,:]=self.tsSamples[indices,:]	
		self.batchTargets[:,:]=self.tsTargets[indices,:]
	
	def addNoise(self, ratio = 6):
		batch = self.batch
		mean = numpy.mean(batch)
		zeroMeanBatch = batch - mean
		zmBatchVar = numpy.var(zeroMeanBatch)
		noise =(np.random.normal(0,1,numpy.shape(batch)))
		noiseVar = numpy.var(noise)
		scalar = (zmBatchVar)/((noiseVar)*ratio)
		noise = mt.sqrt(scalar)*noise
		self.batch =self.batch + noise
