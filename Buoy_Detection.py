
import numpy as np
from numpy import linalg as LA
import os, sys
from numpy import linalg as la
import math
from matplotlib import pyplot as plt
import glob
from scipy import stats

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2

def hist(image):
    img = cv2.imread(image)
    #img = cv2.normalize(img,None)
    histogram1 = cv2.calcHist([img],[0],None,[256],[0,256])     
    histogram2 = cv2.calcHist([img],[1],None,[256],[0,256])
    histogram3 = cv2.calcHist([img],[2],None,[256],[0,256]) 
    
    b,g,r = cv2.split(img)
    stdb = np.std(b)
    avgb = np.mean(b)
    
    stdg = np.std(g)
    avgg = np.mean(g)
    
    stdr= np.std(r)
    avgr= np.mean(r)
    
    return histogram1, histogram2, histogram3, stdb, avgb, stdg, avgg, stdr,avgr    
    
#%%    
red = glob.glob("CroppedBuoys/R*.jpg")
red.sort()
rhb = 0
rhg = 0
rhr = 0
rmb = 0
rmg = 0
rmr = 0
rvb = 0
rvg = 0
rvr = 0

for img in red:
    red_hist_b,red_hist_g,red_hist_r,red_var_b,red_mean_b,red_var_g, red_mean_g, red_var_r,red_mean_r = hist(img)
     
    rhb += red_hist_b
    rhg += red_hist_g 
    rhr += red_hist_r
 
    rmb += red_mean_b
    rmg += red_mean_g
    rmr += red_mean_r

    rvb += red_var_b
    rvg += red_var_g
    rvr += red_var_r
    
rhb = rhb/len(red)
rhg = rhg/len(red)
rhr = rhr/len(red)
rmb = rmb/len(red)
rmg = rmg/len(red)
rmr = rmr/len(red)
rvb = rvb/len(red)
rvg = rvg/len(red)
rvr = rvr/len(red)

#%%
green = glob.glob("CroppedBuoys/G*.jpg")
green.sort()
ghb = 0
ghg = 0
ghr = 0
gmb = 0
gmg = 0
gmr = 0
gvb = 0
gvg = 0
gvr = 0

for img in green:
    green_hist_b,green_hist_g,green_hist_r,green_var_b,green_mean_b,green_var_g, green_mean_g, green_var_r,green_mean_r = hist(img)
     
    ghb += green_hist_b
    ghg += green_hist_g 
    ghr += green_hist_r
 
    gmb += green_mean_b
    gmg += green_mean_g
    gmr += green_mean_r

    gvb += green_var_b
    gvg += green_var_g
    gvr += green_var_r
    
ghb = ghb/len(green)
ghg = ghg/len(green)
ghr = ghr/len(green)
gmb = gmb/len(green)
gmg = gmg/len(green)
gmr = gmr/len(green)
gvb = gvb/len(green)
gvg = gvg/len(green)
gvr = gvr/len(green)

#%%
yellow = glob.glob("CroppedBuoys/Y*.jpg")
yellow.sort()
yhb = 0
yhg = 0
yhr = 0
ymb = 0
ymg = 0
ymr = 0
yvb = 0
yvg = 0
yvr = 0

for img in yellow:
    yellow_hist_b,yellow_hist_g,yellow_hist_r,yellow_var_b,yellow_mean_b,yellow_var_g, yellow_mean_g, yellow_var_r,yellow_mean_r = hist(img)
     
    yhb += yellow_hist_b
    yhg += yellow_hist_g 
    yhr += yellow_hist_r
 
    ymb += yellow_mean_b
    ymg += yellow_mean_g
    ymr += yellow_mean_r

    yvb += yellow_var_b
    yvg += yellow_var_g
    yvr += yellow_var_r
    
yhb = yhb/len(yellow)
yhg = yhg/len(yellow)
yhr = yhr/len(yellow)
ymb = ymb/len(yellow)
ymg = ymg/len(yellow)
ymr = ymr/len(yellow)
yvb = yvb/len(yellow)
yvg = yvg/len(yellow)
yvr = yvr/len(yellow)


    

def calc_Gaussian(x,mean,sigma):

    part_A = 1/(sigma*np.sqrt(2*math.pi))
    part_B = part_A*np.exp(-(x-mean)**2/(2.*sigma**2))

    return part_B

def model_Parameters(samples):

    req_mean = np.mean(samples)
    req_sigma = np.std(samples)

    return req_mean,req_sigma

meanG = 221.2364824777658
standardG = 22.927225795881313


meanR = 247.91235431235
standardR = 9.905603947367796

meanY = 213.25615384615384
standardY = 17.8243214874831

cap = cv2.VideoCapture('detectbuoy.avi')

if (cap.isOpened()== False): 
   print("Error opening video stream or file")


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
   
   
    success, frame = cap.read()
    if success == True:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
        img_copy = frame.copy()
        img = cv2.medianBlur(img_copy,15)
        img = cv2.bilateralFilter(img,9,75,75)
        
        img = cv2.dilate(img,np.ones((5,5),np.uint8),iterations =1)
    
        img_green = ((img[:,:,1] - img[:,:,2]))
        
        img_red = img[:,:,0] 
        img_yellow = (img[:,:,1] + img[:,:,0] - img[:,:,2])
    
        prob_green = calc_Gaussian(img_green,meanG, standardG)
        prob_red = calc_Gaussian(img_red, meanR, standardR)
        prob_yellow = calc_Gaussian(img_yellow, meanY, standardY)
        
    
        
        prob_green = prob_green/np.amax(prob_green)
        prob_red = prob_red/np.amax(prob_red)
        prob_yellow = prob_yellow/np.amax(prob_yellow)
        
    
        
        green = np.zeros_like(img)
        green[prob_green>= 0.0008] = 255
        red = np.zeros_like(img)
        red[prob_red>=0.68] = 255
        final = np.zeros_like(img)
        final[((red==255) | (green == 255))] = 255
        gray1 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        
        kernel= np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(gray1, cv2.MORPH_CLOSE, kernel)
        img2 = frame.copy()
        _, contours, _  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if len(contours) == 3:
            (x,y),radius = cv2.minEnclosingCircle(contours[0])
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(img2,center,radius,(255,0,0),2)
            
            (x,y),radius = cv2.minEnclosingCircle(contours[1])
            center = (int(x),int(y))
            radius = int(radius+1)
            cv2.circle(img2,center,radius,(255,0,0),2)
            
            (x,y),radius = cv2.minEnclosingCircle(contours[2])
            center = (int(x),int(y))
            radius = int(radius + 2)
            cv2.circle(img2,center,radius,(255,0,0),2)
            
        elif len(contours) == 2:
            (x,y),radius = cv2.minEnclosingCircle(contours[0])
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(img2,center,radius,(255,0,0),2)
            
            (x,y),radius = cv2.minEnclosingCircle(contours[1])
            center = (int(x),int(y))
            radius = int(radius+1)
            cv2.circle(img2,center,radius,(255,0,0),2)
            
        else:
            (x,y),radius = cv2.minEnclosingCircle(contours[0])
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(img2,center,radius,(255,0,0),2)
            
          
        img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
        out.write(img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()    
# =============================================================================
#     cv2.imshow('frame',img2)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
# 
# =============================================================================
# =============================================================================
#     images.append(img2)
#     success, image = cap.read()
# 
# =============================================================================
#--------------------------------------------------------------
#video file
# =============================================================================
# def video(images,size):
#     video=cv2.VideoWriter('video2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
#     #print(np.shape(images))
#     for i in range(len(images)):
#         video.write(images[i])
#     video.release()
# #---------------------------------------------------------------
# # main
# if __name__ == '__main__':
# 
#     # Calling the function
#    
# 
#     #Image = images
#     video(images,size=(640,480))
#     
# =============================================================================
    
# =============================================================================
# cap.release()
# cv2.destroyAllWindows()
# =============================================================================




