import cv2
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.data import page
from skimage.filters import (threshold_otsu,threshold_niblack,threshold_sauvola)
import skimage.io
from skimage.color import rgb2gray
path = "/content/valid/images/*.*"
#for file in glob.glob(path):
for i in range(1):
  file = "try7.jpg"
  #print(os.path.split(file)[1])
  original = skimage.io.imread(fname=file)
  #print(original.shape)
  grayscale = rgb2gray(original)
  image=grayscale
  print(image.shape)
  window_size = 25
  thresh_sauvola = threshold_sauvola(image, window_size=window_size)
  # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  binary_sauvola = image > thresh_sauvola
  print(binary_sauvola)
  plt.imsave(os.path.split(file)[1],binary_sauvola,cmap=plt.cm.gray)
  plt.axis('off')
  #plt.savefig("/content/test/image/"+ os.path.split(file)[1])
  #plt.savefig("123.png")
  #plt.show()