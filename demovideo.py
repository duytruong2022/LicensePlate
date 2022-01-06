import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import numpy as np
import imutils
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from os import listdir
from os.path import isfile, join
import cv2
import math
from glob import glob
from tkinter import filedialog
import time
import random
class ALPR:
    def __init__(self, minAR=3, maxAR=5, dataPath='data/'):
        self.dataPath=dataPath
        self.minAR=minAR
        self.maxAR=maxAR
    def getAllImagePath(self):
        mypath='data'
        files = [f for f in listdir(self.dataPath) if isfile(join(self.dataPath, f))]
        return files
    def removeFolderContent(self,folderpath):
        files=glob(folderpath+'/*')
        ok=0
        if files==None:
            files=glob(folderpath+'*')
        for f in files:
            ok=os.remove(f)
        return ok
    def imgshow(self, image):
        plt.figure(figsize=(8,8))
        plt.imshow(image, cmap='gray')
        plt.show()
    def readImage(self, file_name):
        return cv2.imread(file_name, 0)
    def locate_license_plate_candidates(self, gray, keep=5):       
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        # print(cnts)
        return (blackhat, cnts)
    def countObjectArea(self, image):
        area=image[image==255].size
        return area
    def removeSmallComponents(self, image, original_plate):
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        # print(image.shape)

        sizes = stats[1:, -1]; nb_components = nb_components - 1
        threshold=sorted(sizes, reverse=True)[8]
        # threshold=13
        img2 = np.zeros((output.shape),dtype = np.uint8)
        componentBorder = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.removeFolderContent('imgs/')
        for i in range(0, len(sizes)):
            if sizes[i] >= threshold:
                
                x = stats[i+1, cv2.CC_STAT_LEFT]
                y = stats[i+1, cv2.CC_STAT_TOP]
                w = stats[i+1, cv2.CC_STAT_WIDTH]
                h = stats[i+1, cv2.CC_STAT_HEIGHT]
                if h/w >1.2 and h/w <7:
                    img2[output == i + 1] = 255
                    cv2.rectangle(componentBorder, (x-1, y-1), (x + w+1, y + h+1), (0, 255, 0), 1)
                    img_save=original_plate[y:y+h,x:x+w]
                    # self.imgshow(img_save)

                    cv2.imwrite("imgs/"+str(x)+'_'+str(y)+'.jpg',img_save)
        # self.imgshow(componentBorder)
        
        return img2

    def locate_license_plate(self, gray, candidates,blackhat , clearBorder=True):
        lpcnt=None
        roi=None
        for ar_i_min in [3,2.5,2,1]:
            for c in candidates:
                (x, y, w, h) = cv2.boundingRect(c)
                # tính tỷ lệ của chiều dài và chiều rộng
                ar = w / float(h)
                if ar >= ar_i_min and ar <= 5:
                    lpCnt = c
                    licensePlate =blackhat[y:y + h, x:x + w]
                    original_plate=gray[y:y+h, x:x+w]
                    roi=~cv2.threshold(licensePlate, 127, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    roi=self.removeSmallComponents(roi, licensePlate)
                    bitwise_and = cv2.bitwise_and(roi, licensePlate)
                    return ('lpText', bitwise_and, x, y, w, h)
    def normalizePredict(self, img):
        d=np.abs(img.shape[0]-img.shape[1])//2
        extend=np.zeros((img.shape[0], d))
        img=np.hstack((extend, img))
        img=np.hstack((img,extend))
        img=cv2.resize(img, (30,30))
        # self.imgshow(img)
        img=img.reshape((1,30,30,1)).astype('float32')/255.0
        return img
    def predict_image(self, filepath=None, img=None):
        model1 = load_model('model_number_2.h5')
        model=load_model('model_character1.h5')
        # label for number
        maplabel1={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
        inv_map1 = {v: k for k, v in maplabel1.items()}
        #label for character
        maplabel={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'P':14,'Q':15,'R':16,'S':17,'T':18,'U':19,'V':20,'W':21,'X':22,'Y':23,'Z':24}
        inv_map = {v: k for k, v in maplabel.items()}
        if img is not None:
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif filepath is not None:
            img=cv2.imread(filepath)
            gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # gray=cv2.resize(gray, (470,300))
        # self.imgshow(gray)
        blackhat, candidates=alpr.locate_license_plate_candidates(gray)
        data_result=alpr.locate_license_plate(gray, candidates, blackhat)
        if data_result is not None:
            data_result[1]
        imglistpaths=glob('imgs/*.jpg')
        imglistpaths=sorted(imglistpaths, key= lambda s : int(s[5:-4].split('_')[0]))
        lptext=''
        for i in range(len(imglistpaths)):
            img_character = self.normalizePredict(cv2.imread(imglistpaths[i],0))
            if i!=2:
                
                digit = model1.predict_classes(img_character)
                lptext=lptext+str(digit[0])
            else:
                
                digit = model.predict_classes(img_character)
                lptext=lptext+str(digit[0])
        try:
            x, y, w, h=data_result[2], data_result[3], data_result[4], data_result[5]
        except:
            return None
        cv2.rectangle(img, (x-1, y-1), (x + w+1, y + h+1), (0, 255, 0), 1)
        return (img, lptext, x, y)
    
    def videoPredict(self):
        vid = cv2.VideoCapture('Video3.mp4')
        while(True):
            ret, frame = vid.read()
            if random.randint(0,10)==0:
                try:
                    (img, lptext, x, y)=self.predict_image(img=frame)
                except:
                    continue
            else:
                continue
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (x,y)
            fontScale              = 1
            fontColor              = (0,255,0)
            lineType               = 2

            cv2.putText(img,lptext, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()
alpr=ALPR()
# filename=filedialog.askopenfilename()
# alpr.predict_image(filepath=filename)

alpr.videoPredict()