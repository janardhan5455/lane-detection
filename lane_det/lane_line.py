import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

car_cascade = cv.CascadeClassifier('cars.xml')


def canny(img):
    grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(grey,(5,5),0)
    canny=cv.Canny(blur,50,150)
    return canny

def lane(img):
    height=img.shape[0]
    polygons=np.array([[(200,height),(1000,height),(500,280)]])
    mask=np.zeros_like(img)
    cv.fillPoly(mask,polygons,255)
    masked_img=cv.bitwise_and(img,mask)
    return masked_img

def dis_lines(img,lines):
    line_img=np.zeros_like(img)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv.line(line_img,(x1,y1),(x2,y2),(255,0,0),10)
    return line_img

def make_coord(img,line_para):
    try:
        slope,intercept=line_para
    except:
        slope,intercept=0.0001,0.0001
    y1=img.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def avg_lines(img,lines):
    left_fit=[]
    right_fit=[]

    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        # print(parameters)
        slope=parameters[0]
        intercept=parameters[1]
        if slope <0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg=np.average(left_fit,axis=0)
    right_fit_avg=np.average(right_fit,axis=0)
    left_line=make_coord(img,left_fit_avg)
    right_line=make_coord(img,right_fit_avg)

    return np.array([left_line,right_line])

# img=cv.imread('lane_1.jpg')
# lane_img=np.copy(img)
# canny=canny(lane_img)
# croped_img=lane(canny)
# cv.imshow('canny',canny)
# cv.imshow('croped',croped_img)

# lines=cv.HoughLinesP(croped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# avged_lines=avg_lines(lane_img,lines)
# line_img=dis_lines(lane_img,avged_lines)

# cv.imshow('line',line_img)

# combo_img=cv.addWeighted(lane_img,0.8,line_img,1,1)

# cv.imshow('mask',combo_img)

# cv.waitKey(0)


cap=cv.VideoCapture('solidWhiteRight.mp4')
while(True):
    # print(1)
    isTrue,frame=cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.05, 9)
    for (x,y,w,h) in cars:
        plate = frame[y:y + h, x:x + w]
        cv.rectangle(frame,(x,y),(x +w, y +h) ,(51 ,51,255),2)
        cv.rectangle(frame, (x, y - 40), (x + w, y), (51,51,255), -2)
        cv.putText(frame, 'Car', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    canny_img=canny(frame)
    croped_img=lane(canny_img)
    lines=cv.HoughLinesP(croped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    avged_lines=avg_lines(frame,lines)
    line_img=dis_lines(frame,avged_lines)
    combo_img=cv.addWeighted(frame,0.8,line_img,1,1)



        # cv.imshow('car',plate)

    cv.imshow('lane_det',combo_img)
    if cv.waitKey(1) &0xff==ord('a'):
        break
cap.release()
cv.destroyAllWindows