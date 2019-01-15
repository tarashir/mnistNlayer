# ALL CODE FROM mnist IS NOT MINE.  Taken from outdated library but still works
import mnist 
import numpy as np
import copy
from matplotlib import pyplot as plt
import sys


def edgeDetectExtreme(image,V):
    temp = [[0]*len(image[0]) for i in range(len(image))]
    for row in range(1,len(image)-1):
        # # checking all 8 squares around a square
        # for col in range(1,len(image[row])-1):
        #     # a number is an edge if geq V[28] and there is a number near it less than V[28]
        #     if image[row][col] >= V[28]:
        #         for dRow in [-1,0,1]:
        #             if image[row+dRow][col-1] < V[28] or image[row+dRow][col] < V[28] or image[row+dRow][col+1] < V[28]:		
        #                 temp[row][col] = 1
        #                 break
        
        # checking only up down left and right squares
        for col in range(1,len(image[row])-1):
            # a square is an edge if geq V[28] and there is a square near it less than V[28]
            if image[row][col] >= V[28]:
                for change in [[-1,0],[0,1],[1,0],[0,-1]]:
                    if image[row+change[0]][col+change[1]] < V[28]:
                        temp[row][col] = 1
                        break
    
    # # print image
    # for k in range(28):
    #     for j in range(28):
    #         if temp[k][j] == 1:
    #             print("\033[1m\033[4m%3s\033[0m" %"WW",end = "")
    #         else:
    #             print("%3i" % (254*image[k][j]),end = "")
    #     print()
    # print()
    
    return temp
    
# edge detection but we aint learning. assume fixed value for min cut off
def edgeDetect(image):
    temp = [[0]*len(image[0]) for i in range(len(image))]
    for row in range(1,len(image)-1):
        # # checking all 8 squares around a square
        # for col in range(1,len(image[row])-1):
        #     # a number is an edge if geq 145/254 and there is a number near it less than 145/254
        #     if image[row][col] >= 145/254:
        #         for dRow in [-1,0,1]:
        #             if image[row+dRow][col-1] < 145/254 or image[row+dRow][col] < 145/254 or image[row+dRow][col+1] < 145/254:
        #                 temp[row][col] = 1
        #                 break
        
        # checking only up down left and right squares
        for col in range(1,len(image[row])-1):
            # a square is an edge if geq 145/254 and there is a square near it less than 145/254
            if image[row][col] >= 145/254:
                for change in [[-1,0],[0,1],[1,0],[0,-1]]:
                    if image[row+change[0]][col+change[1]] < 145/254:
                        temp[row][col] = 1
                        break
    return temp
            

# used when dy or dx is positive.  When using dx, its the same as dy just with the image transposed
def posShift(height,width,image2,dy):
    maxDy = 0
    deny = True # turns false if we found a value greater than 5/254
    while maxDy < dy:
        for value in image2[height-maxDy]: # check all values in row
            if value > 5/254:
                deny = False # if value greater than 5/254 found, break and stop trying to change dy
                break
        if not deny: # if we found a value greater than 5/254, break
            break
        maxDy += 1
    # if maxDy != 0, shift image vertically by maxDy
    if maxDy != 0:
        image2 = image2[height-maxDy+1:height+1]+image2[0:height-maxDy+1]
        for row in range(maxDy):
            image2[row] = [0]*len(image2[row])
    return image2

def negShift(height,width,image2,dy):
    minDy = 0
    deny = True # turns false if we found a value greater than 5/254
    while minDy > dy:
        for value in image2[-minDy]: # check all values in row
            if value > 5/254:
                deny = False # if value greater than 5/254 found, break and stop trying to change dy
                break
        if not deny: # if we found a value greater than 5/254, break
            break
        minDy -= 1
    # if minDy != 0, shift image vertically by maxDy
    minDy *= -1 # minor efficiency, just changing signs
    if minDy != 0:
        image2 = image2[minDy:height+1]+image2[0:minDy]
        for row in range(minDy):
            image2[height-row] = [0]*len(image2[height-row])
    return image2

def shiftCOM(image,dx,dy):
    height = len(image)-1
    width = len(image[0])-1
    image2 = copy.deepcopy(image)
    # calculate how much we can move image while not displacing a pixel
    # with value greater than 10.  if we move one with less than 10, make it 0
    if dy > 0: # we shift array down
        image2 = posShift(height,width,image2,dy)
    elif dy < 0: # shift array up
        image2 = negShift(height,width,image2,dy)
    if dx > 0: # shift array right
        # transpose the image, then shift, then transpose
        image2 = np.ndarray.tolist(np.asarray(image2).transpose())
        image2 = posShift(height,width,image2,dx)
        image2 = np.ndarray.tolist(np.asarray(image2).transpose())
    elif dx < 0: # shift array left
        # transpose the image, then shift, then transpose
        image2 = np.ndarray.tolist(np.asarray(image2).transpose())
        image2 = negShift(height,width,image2,dx)
        image2 = np.ndarray.tolist(np.asarray(image2).transpose())
    return image2

def convolute(image):# passed a 28x28 ndarray
    # different convolutions
    TLSU = [] # diagonal starting top left and subtract 2 x upper
    TLSB = [] # subract 3 x bottom instead
    TRSU = []
    TRSB = []
    HSU = [] # horizontal subtract 2 x upper
    HSB = [] # above but subtract 2 x bottom
    VSL = [] # vertical subtract 2 x left
    VSR = []
    
    for hrow in range(0,13):
        for hcol in range(0,13):
            row = 2 * hrow
            col = 2 * hcol
            
            # compute convolutions
            # if these are both 0, theres really no point in checking this 4x4
            if image[row + 2][col + 1] == image[row + 2][col + 2] == 0:
                TLSU += [0]
                TLSB += [0]
                TRSU += [0]
                TRSB += [0]
                HSU += [0]
                HSB += [0]
                VSL += [0]
                VSR += [0]
                continue
                
            # compute TL convolutions by doing positive parts then negatives separate
            TLup = image[row][col] # top left subtract upper
            for dRow in [1,2,3]:
                TLup += image[row+dRow][col+dRow-1] + image[row+dRow][col+dRow] # loop thru the 2 ones in the same row
            TLbt = TLup # positive parts of both are equal.  TLbt is TLSB
            # make it so that we have more to subtract so that we dont get too many values that register as diagonals
            if hrow != 0 and hcol != 12:
                TLup -= 1.63 * (image[row][col+1]+image[row+1][col+2]+image[row+2][col+3]+(image[row+3][col+4]+image[row-1][col])/2)
            else:
                TLup -= 2 * (image[row][col+1]+image[row+1][col+2]+image[row+2][col+3])
            if hrow != 12 and hrow != 0:
                TLbt -= 2 * (image[row+2][col]+image[row+3][col+1]+(image[row+1][col-1]+image[row+4][col+2])/2)
            else:
                TLbt -= 3 * (image[row+2][col]+image[row+3][col+1])
            TLSU += [TLup/2.5]
            TLSB += [TLbt/3]
            
            # compute TR convolutions with pos parts first
            TRup = image[row][col+3]
            for dRow in [1,2,3]:
                for dCol in [-1,0]:
                    TRup += image[row+dRow][col+3-dRow-dCol]
            TRbt = TRup # positive parts of both are equal
            # make it so that we have more to subtract so that we dont get too many values that register as diagonals
            if hrow != 0 and hcol != 0:
                TRup -= 1.65 * (image[row][col+2]+image[row+1][col+1]+image[row+2][col]+(image[row+3][col-1]+image[row-1][col+3])/1.8)
            else:
                TRup -= 2 * (image[row][col+2]+image[row+1][col+1]+image[row+2][col])
            if hrow != 12 and hcol != 12:
                TRbt -= 1.9 * (image[row+2][col+3]+image[row+3][col+2]+(image[row+4][col+1]+image[row+1][col+4])/1.8)
            else:
                TRbt -= 3 * (image[row+2][col+3]+image[row+3][col+2])
            TRSU += [TRup/2.7]
            TRSB += [TRbt/3.1]
            
            # compute Horizontal convolutions
            Hup,Hbt = 0,0
            for dCol in [0,1,2,3]:
                # ensure we dont check out of bounds
                if hrow == 12:
                    Hup += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-2.5*(image[row][col+dCol]+image[row-1][col+dCol])
                    Hbt += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-5*(image[row+3][col+dCol])
                elif hrow == 0:
                    Hup += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-3*(image[row][col+dCol])
                    Hbt += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-2.5*(image[row+3][col+dCol]+image[row+4][col+dCol])
                else:
                    Hup += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-2.5*(image[row][col+dCol]+image[row-1][col+dCol])
                    Hbt += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-2.5*(image[row+3][col+dCol]+image[row+4][col+dCol])
            HSU += [Hup/3.9]
            HSB += [Hbt/4.9]
            
            # compute Vertical convolutions
            Vr,Vl = 0,0
            for dRow in [0,1,2,3]:
                # ensure we dont check out of bounds
                if hcol == 12:
                    Vl += 1.2*(image[row+dRow][col+1]+image[row+dRow][col+2])-2.5*(image[row+dRow][col]+image[row+dRow][col-1])
                    Vr += 1.3*(image[row+dRow][col+1]+image[row+dRow][col+2])-5*(image[row+dRow][col+3])
                elif hcol == 0:
                    Vl += 1.2*(image[row+dRow][col+1]+image[row+dRow][col+2])-5*(image[row+dRow][col])
                    Vr += 1.3*(image[row+dRow][col+1]+image[row+dRow][col+2])-2.5*(image[row+dRow][col+3]+image[row+dRow][col+4])
                else:
                    Vl += 1.2*(image[row+dRow][col+1]+image[row+dRow][col+2])-2.5*(image[row+dRow][col]+image[row+dRow][col-1])
                    Vr += 1.3*(image[row+dRow][col+1]+image[row+dRow][col+2])-2.5*(image[row+dRow][col+3]+image[row+dRow][col+4])
            VSL += [Vl/4.78]
            VSR += [Vr/4.4]
    
    return [TLSU,TLSB,TRSU,TRSB,HSU,HSB,VSL,VSR]

def convoluteExtreme(image,V):# passed a 28x28 ndarray
    # different convolutions
    # "subtract"*amount means how exactly the pattern must match
    # to be treated as a non negative value after pooling and relu
    # pattern matching to diagonal, vertical, and horizontal lines
    TLSU = [] # diagonal starting top left and subtract 2 x upper
    TLSB = [] # subract 3 x bottom instead
    TRSU = []
    TRSB = []
    HSU = [] # horizontal subtract 2 x upper
    HSB = [] # above but subtract 2 x bottom
    VSL = [] # vertical subtract 2 x left
    VSR = []
    
    for hrow in range(0,13):
        for hcol in range(0,13):
            row = 2 * hrow
            col = 2 * hcol
            
            # compute convolutions
            # if these are both 0, theres really no point in checking this 4x4
            if image[row + 2][col + 1] == image[row + 2][col + 2] == 0:
                TLSU += [0]
                TLSB += [0]
                TRSU += [0]
                TRSB += [0]
                HSU += [0]
                HSB += [0]
                VSL += [0]
                VSR += [0]
                continue
                
            # # compute TL convolutions by doing positive parts then negatives separate
            # TLup = image[row][col] # top left subtract upper
            # for dRow in [1,2,3]:
            #     TLup += image[row+dRow][col+dRow-1] + image[row+dRow][col+dRow] # loop thru the 2 ones in the same row
            # TLbt = TLup # positive parts of both are equal.  TLbt is TLSB
            # # make it so that we have more to subtract so that we dont get too many values that register as diagonals
            # if hrow != 0 and hcol != 12:
            #     TLup -= 1.63 * (image[row][col+1]+image[row+1][col+2]+image[row+2][col+3]+(image[row+3][col+4]+image[row-1][col])/2)
            # else:
            #     TLup -= 2 * (image[row][col+1]+image[row+1][col+2]+image[row+2][col+3])
            # if hrow != 12 and hrow != 0:
            #     TLbt -= 2 * (image[row+2][col]+image[row+3][col+1]+(image[row+1][col-1]+image[row+4][col+2])/2)
            # else:
            #     TLbt -= 3 * (image[row+2][col]+image[row+3][col+1])
            # TLSU += [TLup/V[0]]
            # TLSB += [TLbt/V[1]]
            # 
            # # compute TR convolutions with pos parts first
            # TRup = image[row][col+3]
            # for dRow in [1,2,3]:
            #     for dCol in [-1,0]:
            #         TRup += image[row+dRow][col+3-dRow-dCol]
            # TRbt = TRup # positive parts of both are equal
            # # make it so that we have more to subtract so that we dont get too many values that register as diagonals
            # if hrow != 0 and hcol != 0:
            #     TRup -= 1.65 * (image[row][col+2]+image[row+1][col+1]+image[row+2][col]+(image[row+3][col-1]+image[row-1][col+3])/1.8)
            # else:
            #     TRup -= 2 * (image[row][col+2]+image[row+1][col+1]+image[row+2][col])
            # if hrow != 12 and hcol != 12:
            #     TRbt -= 1.9 * (image[row+2][col+3]+image[row+3][col+2]+(image[row+4][col+1]+image[row+1][col+4])/1.8)
            # else:
            #     TRbt -= 3 * (image[row+2][col+3]+image[row+3][col+2])
            # TRSU += [TRup/V[2]]
            # TRSB += [TRbt/V[3]]
            # 
            # # compute Horizontal convolutions
            # Hup,Hbt = 0,0
            # for dCol in [0,1,2,3]:
            #     # ensure we dont check out of bounds
            #     if hrow == 12:
            #         Hup += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-2.5*(image[row][col+dCol]+image[row-1][col+dCol])
            #         Hbt += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-5*(image[row+3][col+dCol])
            #     elif hrow == 0:
            #         Hup += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-3*(image[row][col+dCol])
            #         Hbt += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-2.5*(image[row+3][col+dCol]+image[row+4][col+dCol])
            #     else:
            #         Hup += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-2.5*(image[row][col+dCol]+image[row-1][col+dCol])
            #         Hbt += 1.2*(image[row+1][col+dCol]+image[row+2][col+dCol])-2.5*(image[row+3][col+dCol]+image[row+4][col+dCol])
            # HSU += [Hup/V[4]]
            # HSB += [Hbt/V[5]]
            # 
            # # compute Vertical convolutions
            # Vr,Vl = 0,0
            # for dRow in [0,1,2,3]:
            #     # ensure we dont check out of bounds
            #     if hcol == 12:
            #         Vl += 1.2*(image[row+dRow][col+1]+image[row+dRow][col+2])-2.5*(image[row+dRow][col]+image[row+dRow][col-1])
            #         Vr += 1.3*(image[row+dRow][col+1]+image[row+dRow][col+2])-5*(image[row+dRow][col+3])
            #     elif hcol == 0:
            #         Vl += 1.2*(image[row+dRow][col+1]+image[row+dRow][col+2])-5*(image[row+dRow][col])
            #         Vr += 1.3*(image[row+dRow][col+1]+image[row+dRow][col+2])-2.5*(image[row+dRow][col+3]+image[row+dRow][col+4])
            #     else:
            #         Vl += 1.2*(image[row+dRow][col+1]+image[row+dRow][col+2])-2.5*(image[row+dRow][col]+image[row+dRow][col-1])
            #         Vr += 1.3*(image[row+dRow][col+1]+image[row+dRow][col+2])-2.5*(image[row+dRow][col+3]+image[row+dRow][col+4])
            # VSL += [Vl/V[6]]
            # VSR += [Vr/V[7]]
            
            # the old version that didnt work
            # compute TL convolutions by doing positive parts then negatives separate
            TLup = image[row][col] # top left subtract upper
            for dRow in [1,2,3]:
                TLup += image[row+dRow][col+dRow-1] + image[row+dRow][col+dRow] # loop thru the 2 ones in the same row
            TLbt = TLup # positive parts of both are equal.  TLbt is TLSB
            # make it so that we have more to subtract so that we dont get too many values that register as diagonals
            if hrow != 0 and hcol != 12:
                TLup -= V[0] * (image[row][col+1]+image[row+1][col+2]+image[row+2][col+3]+(image[row+3][col+4]+image[row-1][col])/V[1])
            else:
                TLup -= V[27] * (image[row][col+1]+image[row+1][col+2]+image[row+2][col+3])
            if hrow != 12 and hrow != 0:
                TLbt -= V[2] * (image[row+2][col]+image[row+3][col+1]+(image[row+1][col-1]+image[row+4][col+2])/V[3])
            else:
                TLbt -= V[26] * (image[row+2][col]+image[row+3][col+1])
            TLSU += [TLup/V[4]]
            TLSB += [TLbt/V[5]]
            
            # compute TR convolutions with pos parts first
            TRup = image[row][col+3]
            for dRow in [1,2,3]:
                for dCol in [-1,0]:
                    TRup += image[row+dRow][col+3-dRow-dCol]
            TRbt = TRup # positive parts of both are equal
            # make it so that we have more to subtract so that we dont get too many values that register as diagonals
            if hrow != 0 and hcol != 0:
                TRup -= V[6] * (image[row][col+2]+image[row+1][col+1]+image[row+2][col]+(image[row+3][col-1]+image[row-1][col+3])/V[7])
            else:
                TRup -= V[25] * (image[row][col+2]+image[row+1][col+1]+image[row+2][col])
            if hrow != 12 and hcol != 12:
                TRbt -= V[8] * (image[row+2][col+3]+image[row+3][col+2]+(image[row+4][col+1]+image[row+1][col+4])/V[9])
            else:
                TRbt -= V[24] * (image[row+2][col+3]+image[row+3][col+2])
            TRSU += [TRup/V[10]]
            TRSB += [TRbt/V[11]]
            
            # compute Horizontal convolutions
            Hup,Hbt = 0,0
            for dCol in [0,1,2,3]:
                # ensure we dont check out of bounds
                if hrow == 12:
                    Hup += V[12]*(image[row+1][col+dCol]+image[row+2][col+dCol])-V[13]*(image[row][col+dCol]+image[row-1][col+dCol])
                    Hbt += V[14]*(image[row+1][col+dCol]+image[row+2][col+dCol])-2*V[15]*(image[row+3][col+dCol])
                elif hrow == 0:
                    Hup += V[12]*(image[row+1][col+dCol]+image[row+2][col+dCol])-2*V[13]*(image[row][col+dCol])
                    Hbt += V[14]*(image[row+1][col+dCol]+image[row+2][col+dCol])-V[15]*(image[row+3][col+dCol]+image[row+4][col+dCol])
                else:
                    Hup += V[12]*(image[row+1][col+dCol]+image[row+2][col+dCol])-V[13]*(image[row][col+dCol]+image[row-1][col+dCol])
                    Hbt += V[14]*(image[row+1][col+dCol]+image[row+2][col+dCol])-V[15]*(image[row+3][col+dCol]+image[row+4][col+dCol])
            HSU += [Hup/V[16]]
            HSB += [Hbt/V[17]]
            
            # compute Vertical convolutions
            Vr,Vl = 0,0
            for dRow in [0,1,2,3]:
                # ensure we dont check out of bounds
                if hcol == 12:
                    Vl += V[18]*(image[row+dRow][col+1]+image[row+dRow][col+2])-V[19]*(image[row+dRow][col]+image[row+dRow][col-1])
                    Vr += V[20]*(image[row+dRow][col+1]+image[row+dRow][col+2])-2*V[21]*(image[row+dRow][col+3])
                elif hcol == 0:
                    Vl += V[18]*(image[row+dRow][col+1]+image[row+dRow][col+2])-2*V[19]*(image[row+dRow][col])
                    Vr += V[20]*(image[row+dRow][col+1]+image[row+dRow][col+2])-V[21]*(image[row+dRow][col+3]+image[row+dRow][col+4])
                else:
                    Vl += V[18]*(image[row+dRow][col+1]+image[row+dRow][col+2])-V[19]*(image[row+dRow][col]+image[row+dRow][col-1])
                    Vr += V[20]*(image[row+dRow][col+1]+image[row+dRow][col+2])-V[21]*(image[row+dRow][col+3]+image[row+dRow][col+4])
            VSL += [Vl/V[22]]
            VSR += [Vr/V[23]]
    
    return [TLSU,TLSB,TRSU,TRSB,HSU,HSB,VSL,VSR]
    
    
def ReLU(arrays):
    # remove all elements that have values less than 1 and multiply others by 10 (temporary)
    for array in arrays:
        for j in range(len(array)):
            if array[j] <= 1:
                array[j] = 0
            elif array[j] > 1:
                array[j] *= 10
    return arrays    

def combineSimilar(arrays):
    # add similar convolution patterns together
    for i in range(int(len(arrays)/2)):
        for j in range(len(arrays[i])):
            arrays[2*i][j] += arrays[2*i+1][j]
    return arrays[0::2]

def addNearestNeighbors(arrays):        
    # add close elements.  if the center element is non-0, make end result 
    # larger than avg of it and near if some near it are non-zero
    NearNeighborArrays = []
    for array in arrays:
        size = int(round(len(array)**0.5))
        temp = np.ndarray.tolist(np.asarray(array).reshape(size,size)) # reshape arrays to 2D
        temp2 = [] # result of combining each element with neighbors
        for row in range(0,len(temp)): # loop thru combined array positions
            for col in range(0,len(temp[row])):
                # if the center one isnt 0, this process make the result larger
                # beause it adds the avg of the non-zero neighbors to the center
                nonZeroCount = 0
                current = 0
                for dy in [-1,0,1]:
                    # ensure list indicies arent out of range
                    if row+dy <= len(temp)-1 and row+dy >= 0:
                        for dx in [-1,0,1]:
                            # check list indicies and ensure we arent adding center
                            if col+dx <= len(temp[0])-1 and col+dx >= 0 and not (dx == dy == 0):
                                if temp[row+dy][col+dx] != 0:
                                    nonZeroCount+=1
                                    current += temp[row+dy][col+dx]
                # if center is 0, it must have two non zero neighbors or it will be 0
                if temp[row][col] == 0 and nonZeroCount < 2:
                    temp2 += [0] # add the center value
                elif nonZeroCount != 0:
                    temp2 += [temp[row][col] + current/(nonZeroCount+1)]
                else:
                    temp2 += [temp[row][col]]
        NearNeighborArrays += [temp2]
    return NearNeighborArrays


def maxPool(arrays): # these arrays are combinations of similar convolutions
    pools = [[] for i in range(len(arrays))]
    # reshape arrays and insert 0's at the end of each line and bottom to make pooling easier
    for i in range(len(arrays)):
        arrays[i] = np.ndarray.tolist(np.asarray(arrays[i]).reshape(13,13)) # rearrange
        for line in arrays[i]: # append 0's if odd number of elements in row
            if len(line)%2 == 1:
                line.append(0)
        if len(arrays[i])%2 == 1: # insert row of 0's if num rows is odd
            arrays[i].append([0]*len(arrays[i][0]))
        
        # for loops loop thru every other row and col and checks 2x2 squares for their max
        for row in range(int(len(arrays[i])/2)):
            for col in range(int(len(arrays[i][2*row])/2)):
                # create list of 4 elements and then find their max
                subPool = []
                for dRow in [0,1]:
                    subPool += [arrays[i][2*row+dRow][2*col],arrays[i][2*row+dRow][2*col+1]]
                # add the max of the pool to pools
                pools[i].append(max(subPool))
    
    
    return pools
    
    # print functions to check stuff
    
    # array = pools[0]
    # for k in range(28):
    #     for j in range(28):
    #         if k < 26 and j < 26 and array[int(k/2)*13+int(j/2)] > 0:
    #             print("\033[1m\033[4m%3i\033[0m" % array[int(k/2)*13+int(j/2)],end = "")
    #         else:
    #             print("%3i" % a[k][j],end = "")
    #     print()
    # print()
    
    # print("batch")
    # 
    # array = pools[0]
    # for k in range(7):
    #     for j in range(7):
    #         print("%3i" % array[k*7+j],end = "")
    #     print()
    # print()
    # 
    # array = arrays[0]
    # for k in range(14):
    #     for j in range(14):
    #         print("%3i" % array[k][j],end = "")
    #     print()
    # print()

# turns pools into a long vector for fully connected part of network
# returns unit vector
def linearize(arrays):
    vector = []
    for array in arrays:
        vector += array
    vector = np.matrix(vector).reshape(len(vector),1)
    if np.linalg.norm(vector) == 0:
        return vector
    else:
        return vector / np.linalg.norm(vector)


def preprocessing(image,values):
    # edge detect
    if values == None:
        image = edgeDetect(image)
    else:
        image = edgeDetectExtreme(image,values)
    
    # compute center of mass of image, then shift COM to middle of image
    x,y,nonZeroCount = 0,0,0 # sum of center of mass parts
    for w in range(len(image)):
        for j in range(len(image[w])):
            if image[w][j] != 0:
                x += j
                y += w
                nonZeroCount += 1
    # adjust x and y to avg placements instead of sum of placements
    x /= nonZeroCount
    y /= nonZeroCount
    # calculate necessary shifts to get COM as close as possible to 13.5,13.5
    xShift = int(round(13.5-x))
    yShift = int(round(13.5-y))
    # shift COM as necessary
    if xShift != 0 or yShift != 0:
        image = shiftCOM(image,xShift,yShift)
    
    # check for pattern placements in image
    if values == None: # use the corresponding convolution function
        arrays = convolute(image) # compute different convolutions.  RELU, pool
    else:
        arrays = convoluteExtreme(image,values)
    # bunch of different convolutions, we match a few patterns and each creates
    # a new array, then we combine those somehow, I completely forget how.
    arrays = ReLU(arrays) # take out low values
    arrays = combineSimilar(arrays)
    arrays = addNearestNeighbors(arrays) # combine similar convolutions
    arrays = maxPool(arrays) # combine high values that are near eachother
    a_L_1 = linearize(arrays) # creates 4*7*7 x 1 vector for fully connected part
    if np.linalg.norm(a_L_1) == 0:
        image = np.ndarray.tolist(np.asarray(image)*254)
        for k in range(28):
            for j in range(28):
                print("%3i"%image[k][j],end="")
            print()
        print()
    return a_L_1