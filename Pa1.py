from PIL import Image
from pylab import *
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2
import numpy as num, math
import numpy as np 
from math import pi, sqrt, exp
from copy import deepcopy
from numpy import matrix
import matplotlib.cm as cm
from matplotlib import pyplot



# Fetching ad plotting the original file
I = array(Image.open('/Users/dorymauretour/Homework1CP/testingImage.jpeg'))
fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.title('Original Image')
plt.imshow(I,cmap = cm.gray)
plt.show()

#Gaussian function taking two components, n and sigma
def Gaussian(n,sigma):
    size = range(-int(n/2),int(n/2)+1) 
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in size]

#Gaussian derivetive function
def Gaussian_Derrivative(n,sigma):
    size = range(-int(n/2),int(n/2)+1)
    return [-x / (sigma**3*sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in size]


#One dimensional Gaussian mask G to convolve with I
Ix=[]
G = Gaussian(7,1) #sigma > 0 
for i in range(len(I[:,0])):
    x=np.convolve(I[i,:], G) #Convolving Ix with Gx to give I'x
    Ix.append(x)
    
Ix =np.array(np.matrix(Ix))
#Plotting x component of the convolution
fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.title('X component of the convolution with a Gaussian')
plt.imshow(Ix,cmap = cm.gray)
plt.show()

#Y component of the convolution
Iy =[]
for i in range (len(I[0,:])):
    y = np.convolve(I[:,i], G) #Convolving Iy with Gy to give I'y 
    Iy.append(y) 
Iy = np.transpose(Iy)
#Plotting Y component 
fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.title('Y component of the convolution with a Gaussian')
plt.imshow(Iy,cmap = cm.gray)
plt.show()

#X component concolution with gaussian derivative
gd = Gaussian_Derrivative(7,1)
Gx = []
for i in range(len(Ix[:,0])):
    x = np.convolve(Ix[i,:],gd)
    Gx.append(x)
    
#Plotting X component   
fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')    
plt.title('X component of the image convolve with the derivative of a Gaussian')
plt.imshow(Gx,cmap = cm.gray)
plt.show()

#Y component of image convolve with the gaussian derivative
Gy=[]
for i in range (len(Iy[0,:])):
    y = np.convolve(Iy[:,i], gd)
    Gy.append(y) 
Iy_gd= np.transpose(Gy)
#Plotting Y component
fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.title('Y component of the image convolve with the derivative of a Gaussian')
plt.imshow(Iy_gd,cmap = cm.gray)
plt.show()

#Computing magnitude of the edge
Ix_gdsq= np.square(Gx)
Iy_gdsq= np.square(Iy_gd)

Magxy =[]
for i in  range (len(Ix_gdsq)):
    temp = []
    for j in range (len(Iy_gdsq[0,:])):
        temp.append(sqrt(Ix_gdsq[i,j] + Iy_gdsq[i,j]))
        if(j == len(Iy_gdsq[0,:])-1):
            Magxy.append(temp)
Magxy = np.array(np.matrix(Magxy))
#Plotting the magnitude
fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.title('Resulting magnitude image')
plt.imshow(Iy_gd,cmap = cm.gray)
plt.show()

#Implementing non-maximun supression algorithm
A= np.array(np.matrix(Gx))
B= np.array(np.matrix(Iy_gd))
AngleDeg =[]
for i in range(len(Gx)):
    temp=[]
    for j in range(len(Iy_gd[0,:])):
        temp.append((math.atan2(B[i,j],A[i,j]))*180/pi)
        if(j == len(Iy_gd[0,:])-1):
            AngleDeg.append(temp)
            
Angle= np.array(np.matrix(AngleDeg))
Magxy_Temp = Magxy 
NonMax  = deepcopy(Magxy)
for i in range(len(Angle[:,0])):
    for j in range(len(Angle[0,:])):
        try:
            #Horizontal Edge
            if ((-22.5< Angle[i,j] <= 22.5) | ( -157.5 < Angle[i,j] <= 157.5)):
                if((Magxy_Temp[i,j] < Magxy_Temp[i+1,j]) | (Magxy_Temp[i, j] < Magxy_Temp[i-1,j])):
                    NonMax[i,j] = 0
                    
                
            #Vertical Edge
            if ((-112.5 < Angle[i,j] <= -67.5) | ( 67.5 < Angle[i,j] <= 112.5)):
                if((Magxy_Temp[i,j] < Magxy_Temp[i,j+1]) | (Magxy_Temp[i, j] < Magxy_Temp[i,j-1])):
                    NonMax[i,j] = 0

                    
            #+45 Degree Edge
            
            if ((-67.5 < Angle[i,j] <= -22.5) | ( 112.5 < Angle[i,j] <= 157.5)):
                if((Magxy_Temp[i,j] < Magxy_Temp[i+1,j+1]) | (Magxy_Temp[i, j] < Magxy_Temp[i+1,j+1])):
                    NonMax[i,j] = 0
                    

            #-45 degree Edge
            
            if ((-157.5 < Angle[i,j] <= -112.5) | (22.5 < Angle[i,j] <= 67.5 )):
                if((Magxy_Temp[i,j] < Magxy_Temp[i-1,j-1]) | (Magxy_Temp[i, j] < Magxy_Temp[i+1,j+1])):
                    NonMax[i,j] = 0
                    
            

        except IndexError:
                pass
            
fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')            
plt.title('Canny edge image after non-maximun suppression')
plt.imshow(NonMax,cmap = cm.gray)
plt.show()           

#Applying Hysteresis thresholding to obtain final edge-map
NonMax =(matrix(NonMax))
Hysterisis = deepcopy(NonMax)
u = v = 0
highT = 4.5  
lowT = 1.5  
maxm = 255 

for i in range(len(Hysterisis[:,0])-1):
    
    for j in range(len(Hysterisis[0,:])-1):
        
        u = i
        v = j
        while((u!=0)&(v!=0)):
            
            if (Hysterisis[u,v] >=highT):
                
                Hysterisis[u,v] = maxm
                try:
                    
                    if (lowT<=Hysterisis[u+1,v] < highT):
                        
                        Hysterisis[u+1,v] = maxm
                        u = u+1
                        v = v
                    elif (lowT<=Hysterisis[u-1,v]<highT):
                        
                        Hysterisis[u-1,v] = maxm
                        u = u-1
                        v= v
                    elif (lowT<=Hysterisis[u+1,v+1]<highT):
                        
                                Hysterisis[u+1,y+1] = maxm
                                u = u+1
                                v = v+1
                    elif (lowT<=Hysterisis[u-1,v-1]<highT):
                                                
                        Hysterisis[u-1,v-1] = maxm
                        u = u-1
                        v = v-1
                    elif (lowT<=Hysterisis[u,v+1]<highT):
                                                                       
                        Hysterisis[u,v+1] = maxm
                        u = u
                        v = v+1

                    elif (lowT<=Hysterisis[u,v-1]<highT):
                        
                        Hysterisis[u,v-1] = maxm
                        u = u
                        v = v-1
                    elif (lowT<=Hysterisis[u-1,v+1]<highT):
                        
                        Hysterisis[u-1,v+1] = maxm
                        u = u-1
                        v = v+1
                    elif (lowT<=Hysterisis[u+1,v-1]<highT):
                        
                        Hysterisis[u+1,v-1] = maxm
                        u = u+1
                        v = v-1
                    else: 
                        
                        u = 0
                        v = 0


                except IndexError: 
                    
                    u = 0
                    v = 0

            elif (lowT<= Hysterisis[u,v]<highT):
                
                Hysterisis[u,v] = maxm

            else:
                Hysterisis[u,v] = 0
                u = 0
                v = 0 
                
fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.title('Hysteresis thresholding edge map')
plt.imshow(Hysterisis,cmap = cm.gray)
plt.show()

