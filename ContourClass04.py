import cv2 as cv
import numpy as np
import csv
from scipy.spatial import distance
import time
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module

font = cv.FONT_HERSHEY_PLAIN

class GHFilter():
    def __init__(self,dt,g,h):
        self.dt = dt
        self.g = g
        self.h = h

    def predict(self, loc, dloc):
        pred_loc = np.array(loc) + np.array(dloc) * np.array(self.dt)
        return  pred_loc

    def update(self,z, pred_loc, dloc):
        residual = z - pred_loc
        dloc = dloc + self.h * residual/self.dt
        loc = pred_loc + self.g * residual
        return loc, dloc

class ContourObj():
    def __init__(self, loc, c):
        self.loc = loc
        self.color = c
        self.tag = False

class TrackObj:
    def __init__(self, id, loc, c):
        self.id = id
        self.loc = loc
        self.dloc = [0.,0.]
        self.color = c
        self.pred_loc = loc
        self.count = 0

def ProcessImage1(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.blur(gray, (5,5))
    outlines = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,9,2)
    cv.imshow("Outlines", outlines)
    return outlines

def ProcessImage2(src):
    kernel_size = 3
    kernel = np.ones((3,3), np.uint8)  
    low_threshold = 34 # see CCJ10_38.dat
    ratio = 3
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)   
    img_blur = cv.blur(gray, (4,4))    #is important to reduce noise. (4,4) is better than (3,3)
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    dst_gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)   
    dilated_image = cv.dilate(dst_gray, kernel, iterations=1)       
    return dilated_image
        
def GetContour(src):
    ContourList = []
    contours, hierarchy = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        contours_poly = cv.approxPolyDP(c, 3, True)
        boundRect = cv.boundingRect(contours_poly)
        #boundRect = cv.boundingRect(c)
        x = int(boundRect[0])
        w = int(boundRect[2])
        y = int(boundRect[1])
        h = int(boundRect[3])
        length = (w + h) * 2
        if (w > 20) and (h > 20) and (length < 1000):
            xMid = x + w / 2.
            yMid = y + h / 2.
            color = np.array(cv.mean(src[y:y+h,x:x+w])).astype(np.uint8)
            colorSum = int(color[0]) + int(color[1]) + int(color[2])
            relBlue = int((color[0] / colorSum)*10)
            relGreen = int((color[1] / colorSum)*10)
            relRed = int((color[2] / colorSum)*10)
            if (relBlue >= 4) and (relGreen <= 3) and (relRed <= 2):
                c_obj = ContourObj([xMid,yMid],0) # Blue
                ContourList.append(c_obj)
            elif (relBlue <= 2) and (relGreen <= 2) and (relRed >= 4):
                c_obj = ContourObj([xMid,yMid],1) # Red
                ContourList.append(c_obj)
    return  ContourList

def CleanList(List, label):
    c_vector = []
    for c in List:
        c_vector.append(c.loc)
    try:
        dist = distance.cdist(c_vector, c_vector, 'euclidean')
    except:
        print("missing contour data")
    counter = 0
    try:
        for i in range(len(dist)-1):
            for k in range(i+1,len(dist[i])): # only check upper right hand part of matrix as matrix is symetric
                if (dist[i][k] >= 0.) and (dist[i][k] < 30):
                    List.pop(k-counter)
                    counter += 1
    except:
        print("List cleaning failed")
    return List

def CompareObjects(ghf,ContourList, TrackList, use_ghf, id_counter):
    c_vector = []
    t_vector = []
    for c in ContourList:
        c_vector.append(c.loc)
    for t in TrackList:
        t_vector.append(t.loc)
    try:
        dist = distance.cdist(t_vector, c_vector, 'euclidean')
    except:
        print("missing contour or track data")
    try:
        for i in range(len(dist)):
            if np.min(dist[i]) < 30:
                index = np.argmin(dist[i])
                if TrackList[i].color == ContourList[index].color:
                    ContourList[index].tag = True
                    TrackList[i].count = 0      # Reset of detection counter
                    if use_ghf:
                        TrackList[i].loc, TrackList[i].dloc = ghf.update(ContourList[index].loc, TrackList[i].pred_loc, TrackList[i].dloc)
                    else:
                        TrackList[i].loc = ContourList[index].loc
            else:
                TrackList[i].count += 1
    except:
        print("Distance C-T not working")
    for c in ContourList:
        if c.tag == False:      # new object
            t_obj = TrackObj(id_counter,c.loc,c.color)
            TrackList.append(t_obj)
            id_counter += 1
    # Check persistence of tracked object, drop after 25 consecutive failed detections
    for i, t in enumerate(TrackList):
        if t.count >= 10:
            TrackList.pop(i)
    return id_counter

def propagate_tracks(ghf,TrackList):
    for t in TrackList:
        t.pred_loc = ghf.predict(t.loc, t.dloc)

def InitTrackList(ContourList,TrackList,id_counter):
    for c in ContourList:
        t_obj = TrackObj(id_counter,c.loc,c.color)
        TrackList.append(t_obj)
        id_counter += 1
    return TrackList,id_counter

def init_filter():
    g = 0.25
    h = 0.01
    dt = [1.,1.]
    return GHFilter(dt, g, h)

def init_camera():
    reso_x , reso_y = 640, 480
    camera = PiCamera()
    camera.resolution = (reso_x, reso_y)
    camera.framerate = 25
    camera.hflip = True
    camera.vflip = True
    raw_capture = PiRGBArray(camera, size=(reso_x, reso_y)) 
    return camera, raw_capture
    
def main():
    source_window = "Source"
    time.sleep(0.5)
    ghf = init_filter()
    TrackList = []
    id_counter = 0
    use_ghf = True
    camera, raw_capture = init_camera()   
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):  
        outlines = ProcessImage1(frame.array)
        ContourList = GetContour(outlines)
        ContourList = CleanList(ContourList,"C")
        if (len(TrackList)) > 0:
            propagate_tracks(ghf,TrackList)
        else:
            TrackList, id_counter = InitTrackList(ContourList,TrackList,id_counter)
        TrackList = CleanList(TrackList,"T")
        id_counter = CompareObjects(ghf,ContourList, TrackList,use_ghf, id_counter)
        for t in TrackList:
            coord = (int(t.loc[0]),int(t.loc[1]))
            cv.putText(frame.array,str(t.id)+"_"+str(t.color),coord, font, 1,(0,255-t.count*25,50),2,cv.LINE_AA)                
        #out_img = cv.resize(frame.array, (640,480), interpolation= cv.INTER_LINEAR) 
        cv.imshow(source_window, frame.array)
        frame.truncate(0)
        
        k = cv.waitKey(1) & 0xFF   
        if k == 27:         # Check for ESC
            break
        elif k == 48:         # Check for 0
            for o in ContourList:
                print(o.loc,o.color)
        elif k == 115:       # Check for s
            if use_ghf:
                print("Switch to simple")
                use_ghf = False
            else:
                print("Switch to GHF")
                use_ghf = True

    cv.destroyAllWindows()


if __name__ == ("__main__"):
    main()
