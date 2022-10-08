from calendar import c
import imp
import cv2
import numpy as np
from scipy.signal import find_peaks
from utilities.preprocessing import butterworth_filter,als_baseline,peak_normalization

def crop_image(img):
    try:
        org_x,org_y,_ = img.shape
    except:
        org_x,org_y =img.shape
    l = []
    for i in range(0,2):
        jk = img.sum(axis=i)[:,0] - als_baseline(img.sum(axis=i)[:,0])
        jk =  butterworth_filter(peak_normalization(np.where(jk>0,jk,0)))
        peaks =  find_peaks(jk,prominence=0.2)[0]
        x,y = peaks[0]-400,peaks[-1]+400
        if x<0 or x > org_x: x = peaks[0]
        if y<0 or y> org_y: y=peaks[-1]
        l.append((x,y))
    return img[l[1][0]:l[1][1],l[0][0]:l[0][1]]

def get_count(image1,plot=True):
    image = image1.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)
    #plt.imshow(gray, cmap='gray')
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    #plt.imshow(blur, cmap='gray')
    canny = cv2.Canny(blur, 30, 40, 3)
    #plt.imshow(canny, cmap='gray')
    dilated = cv2.dilate(canny, (1, 1), iterations=0) 
    #plt.imshow(dilated, cmap='gray')
    (cnt, hierarchy) = cv2.findContours(
    dilated.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    count = 0
    areas = []
    cntd = []
    for cn  in cnt:
        area = cv2.contourArea(cn)
        areas.append(area)
        if area>80:
            count+=1
            cntd.append(cn)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cntd, -1, (0, 255, 0), 2)
   
    return count,areas

def get_blue(imageFrame1,plot=True):
    #imageFrame = cv2.subtract(imgy,dark)
    imageFrame = imageFrame1.copy()
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    kernal = np.ones((2, 2), "uint8")

        # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                   mask = blue_mask)

        # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    b=0
    for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            #print(area)
            if(area > 2000):
                x, y, w, h = cv2.boundingRect(contour)
                b+=1
                
    return b            

def white_count(img):
    lower_white = np.array([0,0,180])
    higher_white = np.array([255,255,255])
    white_range = cv2.inRange(img, lower_white, higher_white)
    (cnt, hierarchy) = cv2.findContours(white_range, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    kl = []
    for a in cnt:
        pr = cv2.contourArea(a)
        if pr>15: kl.append(pr)
    #qp=np.quantile(kl,0.9)
    count = len(kl)#len([i for i in kl if qp<i])
    return count
    
def white_count_2(img):
    lower_white = np.array([0,0,180])
    higher_white = np.array([255,255,255])
    white_range = cv2.inRange(img, lower_white, higher_white)
    (cnt, hierarchy) = cv2.findContours(white_range, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    kl = []
    for a in cnt:
        pr = cv2.contourArea(a)
        if pr>15: kl.append(pr)
    qp=np.quantile(kl,0.9)
    count = len([i for i in kl if qp<i])
    return count



def get_blue_1(imageFrame):
    #imageFrame = cv2.subtract(imgy,dark)
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)



    # Set range for blue color and
    # define mask
    blue_lower = np.array([0, 51, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    kernal = np.ones((2, 2), "uint8")

        # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                   mask = blue_mask)



        # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    b=0
    jk = []
    for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            #print(area)
            if(area > 0):
                x, y, w, h = cv2.boundingRect(contour)
                jk.append(area)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (255, 0, 0), 2)

               
    res = find_peaks(jk,prominence=max(jk)*0.25)[0].shape[0]
    return res   

def new_white_count(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31,31), 0)
    canny = cv2.Canny(blur, 10, 50, 20)
    dilated = cv2.dilate(canny, (21, 21), iterations=1)
    (cnt, hierarchy) = cv2.findContours(
        dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ckl = []
    for i in cnt:
        a = cv2.contourArea(i)
        if a>5: ckl.append(a)
    return find_peaks(ckl)[0].shape

def get_analysis(path):
    blue, white,tol_count,white_2,white_3=0,0,0,0,0
    try: img = cv2.imread(path)
    except Exception as e:
        print(e)
        return {"blue":blue,"white":white,"Total":tol_count}
    try:
        img_p = img#crop_image(img)
    except Exception as e:
        print(e)
    try:
        blue = get_blue(img_p)
        blue1 = get_blue_1(img_p)
    except Exception as e:
        print(e,"BLUE")
    try:
        white = white_count(img_p)
        white1,_ = get_count(img_p)
        white_2 = white_count_2(img_p)
        white_3 = new_white_count(img_p)
    except Exception as e:
        print(e,"WHITE")
    tol_count=blue+white
    return {"blue":blue,"blue1":blue1,"white":white,"Total":tol_count,"White1":white1,"white2":white_2,"Whiet3":white_3,"img_shape":img.shape,"Image_Mean":img.mean()}