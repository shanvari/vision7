import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.ndimage as ndi
from skimage.feature import corner_harris, corner_peaks
img = cv2.imread("Homeworks/Images/7/harris.JPG")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 
SOBEL_X = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int32")

SOBEL_Y = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int32")

GAUSS = np.array((
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]), dtype="float64")

import cv2 
import matplotlib.pyplot as plt
import numpy as np
#7.2.1
img1 = cv2.imread("Homeworks/Images/7/sl.jpg")  
img2 = cv2.imread("Homeworks/Images/7/sm.jpg")
img3 = cv2.imread("Homeworks/Images/7/sr.JPG")
def feature_matching(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    #sift
    sift = cv2.xfeatures2d.SIFT_create()
    
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    
    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    
    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
    return image

plt.figure(figsize=(20, 20))
image = feature_matching(img1, img2)
plt.imshow(image)
 
 
plt.figure(figsize=(20, 20))
image = feature_matching(img1, img3)
plt.imshow(image)

  


plt.figure(figsize=(20, 20))
image = feature_matching(img2, img3)
plt.imshow(image)

 

#7.2.2
img1 = cv2.imread("Homeworks/Images/7/my own camera1.jpg")  
img2 = cv2.imread("Homeworks/Images/7/my own camera2.jpg")
plt.figure(figsize=(20, 20))
image = feature_matching(img1, img2)
plt.imshow(image)
