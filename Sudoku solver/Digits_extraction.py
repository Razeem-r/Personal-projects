import cv2
import numpy as np
import operator
import os
from os import listdir
from os.path import isfile,join

path = ('./sudoko/v1_training/images')
filename = [f for f in listdir(path) if isfile(join(path,f))]

def blurthresh(x,skipdilate=False):
    proc = x.copy()
    #proc = cv2.bilateralFilter(proc,9,75,75)
    #tweak parameters as necessary to find contours
    proc = cv2.cvtColor(proc,cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(proc, (9, 9), 0)
    #ret,proc = cv2.threshold(proc,120,255,cv2.THRESH_BINARY)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 1.1)
    proc = cv2.bitwise_not(proc, proc)  
    kernel = np.ones((4,4),np.uint8)
    proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel)


#     if skipdilate:
#         kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
#         proc = cv2.dilate(proc, kernel) 
   
    return proc


def conts(z):
    proc = cv2.Canny(z, 80, 180, apertureSize = 3)

    contours, h = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    y = z.copy()
    cv2.drawContours(y, contours, -1, (0,255,0), 3)
    cv2.imshow('cont',y)
    cv2.waitKey()
    polygon = contours[0]
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                      polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                      polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                         polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                       polygon]), key=operator.itemgetter(1))
    src,p = cropnwarp([polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]])
    return contours,src,p


def cropnwarp(crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    #print(src.shape)

    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    
    m = cv2.getPerspectiveTransform(src, dst)

    return src,cv2.warpPerspective(x, m, (int(side), int(side)))


def dilate(img):
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    kernel1 = np.array([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]],np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)  
    img_erosion = cv2.erode(img_dilation, kernel1, iterations=5) 
    return img_dilation


def removegrid2(img):    
    proc1 = blurthresh(img)
    #proc = cv2.GaussianBlur(proc1, (9, 9), 0)
    
    edges = cv2.Canny(proc1, 80, 180, apertureSize = 3)
#     proc1 = dilate(proc1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, None,20, 50)
    print(lines.shape)
    for line in lines:

        for x1, y1, x2, y2 in line:
            cv2.line(proc1, (x1, y1), (x2, y2),(0, 255, 0), 3)
    return proc1


def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def infer_grid(img):
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
            
    return squares


def getdigits(img,sq,i):
    j = img.copy()
    boxes=[]
    global r
    dir1 = os.path.join(path,filename[i]+str(i))
    if not os.path.exists(dir1):
        os.mkdir(dir1)

    for o in range(81):
        k = j[int(sq[o][0][1]):int(sq[o][1][1]),int(sq[o][0][0]):int(sq[o][1][0])]
        boxes.append(k)
        pic = path+'/'+filename[i]+str(i)+'/'+str(o)
        cv2.imwrite(pic+'.jpg',k)
        
        #cv2.imshow(str(i)+' '+ str(x),k)
        #cv2.waitKey()
        #retrievdig(k)
        #if cv2.waitKey() & 0xFF == ord('q'):
         #   break 
        #cv2.destroyWindow(str(i)+' '+ str(x))
    cv2.destroyWindow(str(i)+' '+ str(x))


for i in range(len(filename)):
    t=i
    x = cv2.imread(path+'/'+filename[t])
    
    cv2.imshow(str(i),x)

    proc = blurthresh(x,True)
    cv2.imshow('prepro '+str(i),proc)
    cv2.waitKey()
    cont,points,pers = conts(proc)
    z = cv2.drawContours(proc, cont, -1, (0,0,255), 2)
    
    gridless2 = removegrid2(pers)
    
            #gridless = removegrid(pers)

            #c = x.copy()
            #pts = np.array(points,dtype=np.int32)
            #pts = pts.reshape((-1,1,2))
            #cv2.polylines(c,[pts],isClosed=True,color=(255,0,0),thickness=5)

            #cv2.waitKey()
            #cv2.imshow(str(i)+' cont',z)
    
    
    #cv2.waitKey()
    #cv2.imshow(str(t)+' pers',pers)

            #cv2.waitKey()
            #cv2.imshow(str(i)+' gridless',gridless)

    cv2.waitKey()
    cv2.imshow(str(t)+' gridless1',gridless2)
        
    squares = infer_grid(pers)
    sq = np.asarray(squares)
        
            #getdigits(gridless,sq)
    getdigits(gridless2,sq,i)
    cv2.waitKey()
    
    if cv2.waitKey() & 0xFF == ord('q'):
        break
    
    cv2.destroyAllWindows()
cv2.destroyAllWindows()

