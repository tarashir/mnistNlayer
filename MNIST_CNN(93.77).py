import mnist
import numpy as np
import copy
import random
from matplotlib import pyplot as plt
import sys
import ConvolutionalPreprocessing as CPR

def testValues(values, W_L, dontTest, trainImages,trainLabels,idx):
    minChange = 0.014 # the change we make if a value is less than 1
    valChange = 0.010 # % change to values
    
    # make adjustments to values
    # scores for each child of values
    scores = [[0,0] for h in range(len(values))]
    
    # this will only happen on the first time.  'dontTest' is so that we dont retest a network we just had
    # if dontTest != None:
    #     for pair in dontTest: 
    #         scores[pair[0]][pair[1]] = None
    
    for row in range(len(scores)):
        for col in [0,1]:
            if scores[row][col] == None:
                continue
            # adjust values
            temp = values[row] # store the original value so we know what to values[row] to
            if values[row] < .5: # if its less than 1, it might want to go negative, so we adjust by this much
                values[row] += minChange*(2*col-1)
            else:
                values[row] *= 1+valChange*(2*col-1)
            
            costs = [0,0]
            for p in range(20):
                image = np.ndarray.tolist(trainImages[10 * idx + p].reshape(28,28)/254)
                a_L_1 = CPR.preprocessing(image,values)
                # transformed vector.  All values between 0 and probably 1
                a_L_0 = W_L * a_L_1
               
                # # compute score by accuracy
                # # check if largest value in a_L_0 corresponds to correct value
                # results = np.ndarray.tolist(a_L_0)
                # if results.index(max(results)) == trainLabels.item(10 * idx + p):
                #     # use a minus here so we dont have to change formulae below.  thus minimum is still what we seek
                #     scores[row][col] -= 1
                
                # compute score by cost.  0<sum(costs)<10
                for k in range(10):
                    # in the case that we are calculating cost of correct weights
                    if k == trainLabels.item(10 * idx + p):
                        # costs[k] = (1-a_L_0.item(k)**2)**2\
                        costs[1] += 9*(1-a_L_0.item(k))**2
                    else:
                        # costs[k] = a_L_0.item(k)**4
                        costs[0] += a_L_0.item(k)**2
                        
            # read adjust values
            values[row] = temp
            # # record score
            scores[row][col] += sum(costs)
    
    # calculate parent score
    parentScore = 0
    costs = [0,0]
    for p in range(20):
        image = np.ndarray.tolist(trainImages[10 * idx + p].reshape(28,28)/254)   
        a_L_1 = CPR.preprocessing(image,values)
        # transformed vector.  All values between 0 and probably 1
        a_L_0 = W_L * a_L_1
        
        # compute score by accuracy
        results = np.ndarray.tolist(a_L_0)
        if results.index(max(results)) == trainLabels.item(10 * idx + p):
            # use a minus here so we dont have to change formulae below.  thus minimum is still what we seek
            parentScore -= 1
        
        # compute score by costs.  0<sum(costs)<10
        for k in range(10):
            # in the case that we are calculating cost of correct weights
            if k == trainLabels.item(10 * idx + p):
                # costs[k] = (1-a_L_0.item(k)**2)**2\
                costs[1] += 9*(1-a_L_0.item(k))**2 # now its weighted as much as the others
            else:
                # costs[k] = a_L_0.item(k)**4
                costs[0] += a_L_0.item(k)**2
    
    parentScore = sum(costs)
    
    # turn scores into linear array without None values
    linearValues = []
    for i in scores:
        for j in i:
            linearValues += [j]
    
    # find minimum values in linear array
    minCosts = []
    cpy = copy.copy(linearValues)
    # number of children selected for modifying next gen parent
    takeLowestNCosts = 28
    for n in range(takeLowestNCosts):
        if parentScore > min(cpy): # check to make sure we're not getting worse than the parent
            minCosts += [cpy.pop(cpy.index(min(cpy)))] # add the minimum costs to the array
        else:
            break
    # print(minCosts)
    
    dontTest = [] # dontTest for next generation
    # find min values in networkArrayCosts
    # then add it to list of dontTest for next generation and adjust that element in parent array
    for i in range(len(scores)):
        for cost in minCosts: # change the values that caused cost to be lowest
            if cost in scores[i]:
                second = scores[i].index(cost)
                dontTest += [[i,1-second]] # the 1-third is so we dont test the sister of the best ones next time
                # adjust the X most influential elements of parent Network by some amount
                if values[i] < .5: # if its less than 1, it might want to go negative, so we adjust by this much
                    values[i] += minChange*(2*second-1)
                else:
                    values[i] *= 1+valChange*(2*second-1)
    
    # # readable output for changes and parent array
    # changes = [[0,0] for i in range(len(values))]
    # for i in range(len(minCosts)):
    #     changes[dontTest[i][0]][dontTest[i][1]] = 1
    # print(changes)
    
    return values,dontTest
    
############################## 

def main():
    # # show plot for mastercosts
    # masterCosts = np.ndarray.tolist(np.loadtxt("MasterCosts.txt"))
    # plt.plot(masterCosts)
    # plt.show()
    
    # make sure original Network is right shape and original values are correct
    originalNeuralNet = np.matrix([[(random.random()-.5) / 4.1 for j in range(196)] for i in range(10)])
    np.savetxt("OriginalNeuralNetwork.txt",originalNeuralNet)
    originalValues = np.matrix([1.63,2,2,2,2.5,3,1.65,1.8,1.9,1.8,2.7,3.1,1.2,2.5,1.2,2.5,3.9,4.9,1.2,2.5,1.3,2.5,4.78,4.4,3,2,3,2,150/254])
    np.savetxt("OriginalConvoValues.txt",originalValues)
    
    # get training data
    trainImages = mnist.train_images()
    trainLabels = mnist.train_labels()
    trainImages = trainImages.reshape((trainImages.shape[0], trainImages.shape[1] * trainImages.shape[2]))
    
    # load network from file, set up neural network (list of weights) as 10x784 ndarray
    W_L = np.matrix(np.loadtxt("OriginalNeuralNetwork.txt"))
    values = np.ndarray.tolist(np.loadtxt("OriginalConvoValues.txt")) # load values to use as normal list
    
    # track costs as time goes on
    masterCosts = []
    
    # the directions of the adjustments that we just made
    dontTest = None
    
    # train neural Network in groups of 10 images to speed things up
    for i in range(1,5999):
        # # groups of 10 tests to improve neural net
        changes = [[0] * (196) for n in range(10)] # changes to be made to each weight
        for p in range(10):
            image = np.ndarray.tolist(trainImages[10 * i + p].reshape(28,28)/254)      
            a_L_1 = CPR.preprocessing(image,values)
            
            # transformed vector.  All values between 0 and probably 1
            a_L_0 = W_L * a_L_1
            
            # compute costs.  0<cost<10
            costs = [0]*10
            for k in range(10):
                # in the case that we are calculating cost of correct weights
                if k == trainLabels.item(10 * i + p):
                    # costs[k] = (1-a_L_0.item(k)**2)**2\
                    costs[k] = (1-a_L_0.item(k))**2
                else:
                    # costs[k] = a_L_0.item(k)**4
                    costs[k] = a_L_0.item(k)**2
            # compile changes in list
            for k in range(10): # rows of Network
                # a = a_L_0.item(k)**3
                b = a_L_0.item(k)
                for j in range(196):
                    # in the case that we are adjusting the correct weights
                    if k == trainLabels.item(10 * i + p):
                        # changes[k][j] -= (a-b) * costs[k] * a_L_1.item(j) / 600 # can use square or non square for cost, about same results
                        changes[k][j] -= (b-1) * costs[k] * a_L_1.item(j) / 20
                    else:
                        # changes[k][j] -= a * costs[k] * a_L_1.item(j) / 600
                        changes[k][j] -= b * costs[k] * a_L_1.item(j) / 40
        
        # make adjustments to neural network
        W_L += np.matrix(changes)
        
        # train values
        values,dontTest = testValues(values, W_L, dontTest, trainImages,trainLabels, i)
        
        # every so often, progress check
        if i%2 == 0:
            print("Generation Number:",i+1)
            
            for p in range(10):
                image = np.ndarray.tolist(trainImages[10 * i + p].reshape(28,28)/254)      
                a_L_1 = CPR.preprocessing(image,values)
                
                # transformed vector.  All values between 0 and probably 1
                a_L_0 = W_L * a_L_1
                # compute costs.  0<cost<10
                costs = [0]*10
                for k in range(10):
                    # in the case that we are calculating cost of correct weights
                    if k == trainLabels.item(10 * i + p):
                        # costs[k] = (1-a_L_0.item(k)**2)**2\
                        costs[k] = (1-a_L_0.item(k))**2
                    else:
                        # costs[k] = a_L_0.item(k)**4
                        costs[k] = a_L_0.item(k)**2
            print("cost =",sum(costs)) # print sum of 10 most recent costs
            masterCosts += [sum(costs)]
            # write network to file **LONG TERM**
            # # if i%10 == 0:
            # #     np.savetxt("MasterCosts.txt",np.matrix(masterCosts))
            # #     np.savetxt("NeuralNetwork.txt",W_L)
            # #     np.savetxt("ConvoValues.txt",np.matrix(values))
            
            # # write network to file
            if i % 50 == 0 and i != 0: # SHORT TERM
                np.savetxt("NeuralNetwork.txt",W_L)
                np.savetxt("ConvoValues.txt",np.matrix(values))
                plt.plot(masterCosts)
                plt.show()
                return
    
    # write network to file
    np.savetxt("NeuralNetwork.txt",W_L)
    np.savetxt("ConvoValues.txt",np.asarray(values))
    plt.plot(masterCosts)
    plt.show()
            
    
main()