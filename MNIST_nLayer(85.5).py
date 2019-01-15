# generic structure for n layers
# C is cost, dC is change in cost
# W is a layer of weights, dW like above
# a lot of places i commented int i meant float

# LEARNING
# change w based on dC/dW, not dW/dC
# (maybe, when using dropout) limit norm of weights connected to a 
# single node in the next layer by a constant c

# ALL CODE FROM mnist IS NOT MINE.  Taken from outdated library but still works
import mnist
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy


# for printing a list/array
def Print(M):
	exit = False
	for row in M:
		for col in M:
			if (col > 100):
				exit = True
			print("%6.3f"%col,end="")
		print()
	if (exit):
		print("some number is too large")
		sys.exit()
	print()

# for printing images
def PrintImg(img):
	for i in range(28):
		for j in range(28):
			print("%6.3f"%img[28*i+j],end="")
		print()

# (list) -> int
# returns total of all costs
def Error(vect):
	return sum(vect)

# (float list,int) -> list
# returns list of costs
def Cost(guess, ans):
	costs = []
	for i in range(len(guess)):
		if i != ans: costs.append(guess[i] * guess[i])
		else: costs.append((1 - guess[i]) * (1 - guess[i]))
	return costs

# (float list,int) -> ndarray
# returns ndarray of changes to Cost WRT changes in final activation layer
# vertical list of dC/dL_i
def dCdGuess(guess, ans):
	dCdG = []
	for i in range(len(guess)):
		if i != ans: dCdG.append(2 * guess[i])
		else: dCdG.append(2 * (guess[i] - 1))
	return np.asarray([dCdG]).transpose()

# (float list) -> float list
# standard RELU.  Puts stuff closer to 1
def RELU(layer):
	if np.any(layer) > 0:
		for neuron in range(len(layer)):
			if layer[neuron] < 0:
				layer[neuron] /= 10
			elif layer[neuron] > 1:
				layer[neuron] = 2/3 + layer[neuron] / 3
	else:
		for i in range(len(layer)):
			layer[i] += 1
		return RELU(layer)
	return layer

# (ndarray,int,ndarray list,ndarray list) -> ndarray list
# starts with currentLayer = img, then update layer by layer
def ProduceActLayers(currentLayer,numLayers,weightList,masks=None):
	# multiply through layers and RELU each layer
	tempLayers = []
	for i in range(numLayers):
		currentLayer = (currentLayer @ weightList[i])
		if i != numLayers - 1 and masks != None:
			currentLayer = currentLayer * masks[i+1]
		currentLayer = [np.asarray(RELU(np.ndarray.tolist(currentLayer)[0]))]
		tempLayers.append(currentLayer[0])
	return tempLayers

# (int list list,float) -> ndarray list
# produces masks for dropout, 
# minP and maxP are min/max probability of a node not dropping out
def DropOut(dimLayers,minP,maxP):
	rng = np.random.RandomState(random.randint(1,200))
	p = random.uniform(minP,maxP) # probability of weight not dropping out
	masks = []
	for layerShape in dimLayers:
		tempMask = rng.binomial(size=(layerShape[0],1),n=1,p=p)
		masks.append(tempMask.T)
	return masks

def Training(weightList, dimLayers):

	images = mnist.train_images()
	labels = mnist.train_labels()
	images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))

	epochs = 4
	iterations = 50 # number of batches
	batchSize = 1000
	masks = None
	dropOut = True
	minP = .5
	maxP = .5

	# get random permutation of images and labels
	combo = []
	for j in range(iterations*batchSize):
		combo.append((images[j],labels[j]))
	random.shuffle(combo)

	numLayers = len(weightList)
	numNeurons = sum([pair[0]*pair[1] for pair in dimLayers])
	counts = []
	example = 0
	for k in range(epochs):
		for batch in range (iterations):
			changes = [] # changes made to weights after 'batchsize' trials
			error = 0
			count = 0
			if batch%10 == 0: print(batch)
			# create masks for dropout
			if dropOut: 
				masks = DropOut(dimLayers,minP,maxP)
			for trial in range(batchSize):
				# shuffle which ones drop out, faster than generating a new set of masks
				if dropOut:
					for mask in masks: np.random.shuffle(mask[0])
				tempChanges = [] # has the changes backwards
				actLayers = []
				idx = batchSize * batch + trial
				(img,ans) = combo[idx]
				img = np.asarray([img]) / 255 # make pixel values between 0 and 1
				if dropOut: img = img * masks[0]
				actLayers = ProduceActLayers(img,numLayers,weightList,masks)
				# Backpropagation
				guess = actLayers.pop()
				# ensure we don't divide by 0.  Should never be triggered
				if np.linalg.norm(guess) == 0: print(guess)
				guess /= np.linalg.norm(guess) # normalize guess
				if np.linalg.norm(guess) == 0 or np.linalg.norm(guess) > 1000:
					print("norm too big")
					sys.exit()
				guess = np.ndarray.tolist(guess)
				# if guess.index(max(guess)) == ans:
				# 	count += 1
				costs = Cost(guess, ans)
				error += Error(costs)
				dCdG = dCdGuess(guess, ans) # change in cost WRT final act layer
				# change the learning rate
				if example < 300 * batchSize: delta = 0.03
				elif example < 450 * batchSize: delta = 0.015
				elif example < 600 * batchSize: delta = 0.005
				elif example < 800 * batchSize: delta = 0.001
				else: delta = 0.0005
				# dCdL is vertical, 10x1
				dCdL = dCdG # dCdL is dCdG multiplied by some activation Layers in reverse
				dCdL = delta * np.multiply(np.asarray([costs]).T,dCdL)
				c = .06
				mult=[c,78.4*c]
				for layer in range(numLayers):
					# dC/dW = dC/dL * dL/dW.  dL/dW is the previous actLayer's neurons
					if len(actLayers) != 0:
						dCdW = np.multiply(dCdL, actLayers.pop()).transpose()#*mult[layer]
					else:
						dCdW = np.multiply(dCdL, img).transpose()#*mult[layer]
					
					# for row in dCdW: # make each change thingy 1/(dC/dW) if it's not 0
					# 	for i in range(len(row)):
					# 		if row[i] != 0: row[i] = delta * row[i]
					tempChanges.append(-dCdW)
					# dC/dL^(i-1) = dL^i/dL^(i-1) * dC/dL^i. the second term is W^(i-1)
					if (layer != numLayers - 1):
						dCdL = (weightList[numLayers-1-layer] @ dCdL)
						if dropOut:
							dCdL *= masks[numLayers-1-layer].T
				# average changes over a batch
				if len(changes) == 0: # the first trial of the batch
					for i in range(numLayers):
						changes.append(tempChanges[numLayers - 1 - i])
				else:
					for i in range(numLayers):
						changes[i] += tempChanges[numLayers - 1 - i]
				example += 1
			for layer in range(numLayers): # make changes
				weightList[layer] += changes[layer]
			counts.append(error)
	plt.plot(counts)
	plt.show()
	return weightList

# (weights,number of tests)
def Testing(weightList,n):
	images = mnist.test_images()
	labels = mnist.test_labels()
	images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
	# get random permutation of images and labels
	combo = []
	for j in range(n):
		combo.append((images[j],labels[j]))
	random.shuffle(combo)

	numLayers = len(weightList)
	count = 0
	for i in range(n):
		(img,ans) = combo[i]
		img = np.asarray([img]) / 255
		guess = (ProduceActLayers(img,numLayers,weightList)).pop()
		guess = np.ndarray.tolist(guess)
		if guess.index(max(guess)) == combo[i][1]:
			count += 1
	print("accuracy =",count/n)


# (int pair list) -> bool
# checks if layers' dimensions align
def DimCheck(dimLayers):
	for i in range(len(dimLayers)-1):
		if dimLayers[i][1] != dimLayers[i+1][0]:
			print("misaligned dimensions")
			return False
	return True

# (int pair list) -> ndarray list
# makes layers of weights with dimensions according to dimLayers
def MakeLayers(dimLayers):
	weightList = []
	for layer in dimLayers:
		mean = 0
		numRows = layer[0]
		stdDev = 1 / numRows**0.5 # Xavier initialization
		tempLayer = []
		for col in range(layer[1]):
			tempCol = np.random.normal(mean,stdDev,numRows)
			tempLayer.append(np.ndarray.tolist(tempCol))
		weightList.append(np.asarray(tempLayer).transpose())
	return weightList

def main():
	# create dimensions of layers
	height = 28*28
	width = 10
	dimLayers = [(height,width)]
	# check that dimensions of layers align
	if not DimCheck(dimLayers):
		sys.exit()
	# generate each layer with Xavier Initialization
	weightList = MakeLayers(dimLayers) # holds all the layers of weights
	# print(weightList[1])
	updatedWeights = Training(weightList,dimLayers)
	testNum = 2000
	Testing(updatedWeights,testNum)

	
	
		

main()