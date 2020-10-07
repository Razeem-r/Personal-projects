import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile,join,isdir
from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
%matplotlib inline

def cut_from_image(img, rect):
    #Cuts a rectangle from an image using the top left and bottom right points.
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def scale_centre(img, size, margin=0, background=0):
    
    h, w = img.shape[:2]

    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))

def fillgaps(impath,am = 0.05):
    image = img_as_bool(io.imread(impath))
    out = ndi.distance_transform_edt(~image)
    out = out < am * out.max()
    out = morphology.skeletonize(out)
    out = morphology.binary_dilation(out, morphology.selem.disk(1))
    out = segmentation.clear_border(out)
    out = out | image
    
    return out


def findfeature(q):
    #q = cv2.imread(path+filename[37])

    z = cv2.cvtColor(q,cv2.COLOR_BGR2GRAY)

    r=z.copy()
    height, width =(r.shape[:2])
    margin = 9

    max_area = 25
    seed_point = []
    count=0
    scan_tl = [margin, margin]

    scan_br = [height-margin, width-12]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            if r.item(y, x) > 200 and x < scan_br[1] and y < scan_br[0]:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(r, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    #max_area = area[0]
                    seed_point.append((x,y))
                    count+=1
    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image
    #cv2.imshow('filled',r)
    #cv2.waitKey()
    
    for x in range(width):
            for y in range(height):
                if r.item(y, x) > 100 and x < width and y < height:
                    cv2.floodFill(r, None, (x, y), 0)
    for i in range(count):

        if all([p is not None for p in seed_point[i]]):
                cv2.floodFill(r, mask, seed_point[i], 255)
    for x in range(width):
            for y in range(height):
                if r.item(y, x) < 100 and x < width and y < height:
                    cv2.floodFill(r, None, (x, y), 0)
    #print(count)
    #print(seed_point)
    if count>1:
        cv2.imwrite('C:/Users/Lenovo/Desktop/random/sudoku rand/'+filename[j]+'.jpg',r)
        r=fillgaps('C:/Users/Lenovo/Desktop/random/sudoku rand/'+filename[j]+'.jpg')
        r = r.astype(np.uint8)  #convert to an unsigned byte
        r*=255
    #cv2.imshow('connected',r)
    #cv2.waitKey()
    return r,height,width


def eroded(r):
    kernel = np.array(([0,1,0],[0,1,0],[0,1,0]),dtype=np.uint8)
    kernel1 = np.array(([0,0,0],[1,1,1],[0,0,0]),dtype=np.uint8)
    f = r.copy()
    f = cv2.erode(np.float32(f),kernel,iterations = 1)
    f = cv2.erode(f,kernel1,iterations = 1)
    #cv2.imshow('eroded',f)
    #cv2.waitKey()
    return f


def centre_img(f,height,width):
    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):

            if f.item(y, x) >0:
                #f[x][y]=255
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    bbox = np.array(bbox, dtype='float32')
    digit = cut_from_rect(f, bbox)
        #print(digit.shape)
    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        F = scale_and_centre(digit, 28, 4)
    else:
        F= np.zeros((28, 28), np.uint8)
    #cv2.imshow('centred',F)
    #cv2.waitKey()
    return F

def predictit(F):
    if F.max()==0:
        res = 0
    else:

        input_im = F
        input_im = F.reshape(1,28,28,1)
        #plt.imshow(imageL)
        ## Get Prediction - load classifier separately you have trained on
        res = str(classifier.predict_classes(input_im, 1, verbose = 0)[0])
    return res

s=[]
count = 0
#import files from path in which the digits extracted were saved in
#
for u in range(len(foldname)):
    dirpath = dpath+ foldname[u]+'/'
    filename = [f for f in listdir(dirpath) if isfile(join(dirpath,f))]
    for t in range(len(filename)):
        if filename[t][1]=='.':
            fOld =dirpath +filename[t] 
            fNew = dirpath +'0'+filename[t]
            os.rename(fOld,fNew)
    filename = [f for f in listdir(dirpath) if isfile(join(dirpath,f))]

    for j in range(len(filename)):
        im = cv2.imread(dirpath+filename[j])
        im,h,w = findfeature(im)
        #im = fillfeature(im,h,w)
        im = eroded(im)
        im = centre_img(im,h,w)
        cv2.imshow(str(j),im)

        x = predictit(im)
        print(x)
        
        if cv2.waitKey() & 0xFF == ord('q'):
        #In case of wrong prediction by classifier press 'q' to correct it
            x=int(input())
        s.append(x)
        cv2.waitKey()
        cv2.destroyAllWindows()
         
S = np.asarray(s,dtype=np.uint8)
S = S.reshape((9,9))
