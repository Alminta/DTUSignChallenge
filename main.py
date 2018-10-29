#%%
#%matplotlib inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import time
import cv2
import random


from os import listdir

#from IPython.display import clear_output
from skimage.transform import resize

#import data_utils

import colorsys
from matplotlib.pyplot import figure

def hlsThres(hls,thres):
    hls=np.abs(hls-np.ones(hls.shape)*90)
    hls[hls[...,0]<thres]=0
    #hls[np.logical_and(hls[...,0]>thres,hls[...,0]<180-thres)]=100
    return hls


a= listdir('Data')
print(a)


img = cv2.imread('Data/DTUSigns057.jpg')[...,::-1]



hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)

hls = hlsThres(hls,7)



figure(figsize=(16, 16), dpi=80)
plt.imshow(hls[...,0],cmap='Greys')
plt.show()

