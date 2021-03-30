import cv2
import sys
import numpy as np
#from mail import sendEmail
from flask import Flask, render_template, Response, request, send_file
#from flask_basicauth import BasicAuth

import time
import threading
import ifm3dpy
import math
from jinja2 import *
import os
from xml.etree import ElementTree
#per guardar/llegir imatges
import scipy.misc
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Polygon
import seaborn as sns
import io


global maxArea, minArea, blobColor, minDistBetweenBlobs, minThreshold, maxThreshold, thresholdStep, minRepeatability, minCircularity, maxCircularity, minConvexity, maxConvexity, minInertiaRatio, maxInertiaRatio
global miarea, maarea, flaggflagg
flaggflagg = False
global frame_inici, frame_inici_amplitut, frame_inici2, frame_inici_amplitut2
global maxAreaValue, minAreaValue, colorBlobValue, minDistBetweenBlobsValue, minThresholdValue, maxThresholdValue, thresholdStepValue, minRepeatabilityValue, minCircularityValue, maxCircularityValue, minConvexityValue, maxConvexityValue, minInertiaRatioValue, maxInertiaRatioValue

global param_flagg
param_flagg = False

global llista, llista_bool
llista = []
llista_bool = False

global static_frame_flagg
static_frame_flagg = False

global frame_escollit, frame_escollit2
global im_xyz, im_rdis, im_amp, amplitut_color

#amplitut_color = 0
global frame_flagg, frame_flagg2

frame_flagg = 0
frame_flagg2 = 0

global mostra, mostra2

global frame_valid

global calib_flagg
calib_flagg = 0

global headings, headings2
global Punt0, Punt1, Punt2

global im_xyz
im_xyz = 0

headings = ("ID","Angle","POS X","POS Y","POS Z")
headings2 = ("ID", "RIGHT","LEFT", "BOTTOM", "TOP", "HEIGHT")

#email_update_interval = 600 # sends an email only once in this time interval
#video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically
#object_classifier = cv2.CascadeClassifier("models/fullbody_recognition_model.xml") # an opencv classifier

class myThread(threading.Thread):
    global amplitut_color

    def __init__ (self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        
        
        global llista, llista_bool
        global static_frame_flagg, flaggflagg
        global frame_inici, frame_inici_amplitut, frame_inici2, frame_inici_amplitut2
        global miarea, maarea
        global maxAreaValue, minAreaValue, colorBlobValue, minDistBetweenBlobsValue, minThresholdValue, maxThresholdValue, thresholdStepValue, minRepeatabilityValue, minCircularityValue, maxCircularityValue, minConvexityValue, maxConvexityValue, minInertiaRatioValue, maxInertiaRatioValue
        global frame_valid, detector
        global llistaCentreMases,resta_color, img_grey, img_contorns, binary, keyp
        global im_xyz, im_rdis, im_amp
        global amplitut_color_lineas, amplitut_color
        
        calib_flagg = 0

        
        grados_flagg = False
        done = False
        reload = False
        contador = 0
        llista = []
        
        ##-- Agafem la informació del background que tenim --##
        ##------- actualment(ho indica frame_flagg) ---------##

        if frame_flagg == 0:
            image = cv2.imread('background1.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            image_ok = cv2.resize(image, (224,172))
            matriu = np.load('background1.dat', allow_pickle=True)
            
            frame_escollit = 'background1.png'
            frame_inici = matriu
            frame_inici_amplitut = image_ok

            ret, jpeg = cv2.imencode('.jpg',image_ok)

        if frame_flagg2 == 0:
            image = cv2.imread('background2.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            image_ok = cv2.resize(image, (224,172))
            matriu2 = np.load('background2.dat', allow_pickle=True)
            
            frame_escollit2 = 'background2.png'

            frame_inici2 = matriu2
            frame_inici_amplitut2 = image_ok

            ret, jpeg = cv2.imencode('.jpg',image_ok)


        ##-----Funcio que llegeix els parametres del XML-----##
        
        carreguemValorsXML()
        
        ##---------Assignem el valors llegits al XML---------##        
        
        fg = ifm3dpy.FrameGrabber(ifm3dpy.Camera(), ifm3dpy.IMG_AMP | ifm3dpy.IMG_RDIS | ifm3dpy.IMG_CART)
        im = ifm3dpy.ImageBuffer()

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True    # default: True
        params.minArea = int(minAreaValue)          # default: 50
        params.maxArea = int(maxAreaValue)        # default: 5000

        params.filterByCircularity = False      # default: False
        params.minCircularity = float(minCircularityValue)  # default: 0.8
        params.maxCircularity = float(maxCircularityValue)  # default: inf   

        params.filterByConvexity = False       # default: True
        params.minConvexity = float(minConvexityValue)               # default: 0.95
        params.maxConvexity = float(maxConvexityValue)                 # default: Inf

        params.filterByInertia = False        # default: True
        params.minInertiaRatio = float(minInertiaRatioValue)              # default: 0.1
        params.maxInertiaRatio = float(maxInertiaRatioValue)           # default: Inf

        params.minRepeatability = 1           # default: 2

        params.minDistBetweenBlobs = int(minDistBetweenBlobsValue)       # default: 10

        params.thresholdStep = int(thresholdStepValue)            # default: 10
        params.minThreshold = int(minThresholdValue)              # default: 50
        params.maxThreshold = int(maxThresholdValue)              # default: 220


        params.filterByColor = False            # default: True
        params.blobColor = int(colorBlobValue)                    # default: 0

        
        ##-------------Creem blob detector amb------------ ##
        ##----------- els valors dels parametres-----------## 

        detector = cv2.SimpleBlobDetector_create(parameters=params)

        img_blanca = 255 * np.ones((172,224,3), dtype=np.uint8)
        
        llista = []

        while not done:

        ##-------- Ens mantenim constantment observant -------##
        ##------- si canviem el valor d'algun parametre ------##

            if param_flagg == True:
                
                if minArea != False:
                    params.minArea = int(minArea)
                    #print ("minArea:",minArea)
                    reload = True
                if maxArea != False:
                    params.maxArea = int(maxArea)
                    #print ("maxArea:",maxArea)
                    reload = True
                if blobColor != False:
                    params.blobColor = int(blobColor)
                    #print ("blobColor:",blobColor)
                    reload = True
                if minDistBetweenBlobs != False:
                    params.minDistBetweenBlobs = int(minDistBetweenBlobs)
                    #print ("minDistBetweenBlobs:", minDistBetweenBlobs)
                    reload = True
                if minThreshold != False:
                    params.minThreshold = int(minThreshold)
                    #print ("minThreshold:", minThreshold)
                    reload = True
                if minRepeatability != False:
                    params.minRepeatability = int(minRepeatability)
                    #print ("maxThreshold:", maxThreshold)
                    reload = True
                if minCircularity != False:
                    params.minCircularity = float(minCircularity)
                    #print ("thresholdStep:", thresholdStep)
                    reload = True
                if maxCircularity != False:
                    params.maxCircularity = float(maxCircularity)
                    #print ("thresholdStep:", thresholdStep)
                    reload = True
                if minConvexity != False:
                    params.minConvexity = float(minConvexity)
                    #print ("thresholdStep:", thresholdStep)
                    reload = True
                if maxConvexity != False:
                    params.maxConvexity = float(maxConvexity)
                    #print ("thresholdStep:", thresholdStep)
                    reload = True
                if minInertiaRatio != False:
                    params.minInertiaRatio = float(minInertiaRatio)
                    #print ("thresholdStep:", thresholdStep)
                    reload = True
                if maxInertiaRatio != False:
                    params.maxInertiaRatio = float(maxInertiaRatio)
                    #print ("thresholdStep:", thresholdStep)
                    reload = True
                

            key = cv2.waitKey(1)

        ##----- Si hem canviat el valor d'algun parametre ----##
        ##------ actualitzem el codi amb els nous valors -----##

            if reload == True: # reload parameters
                del detector
                detector = cv2.SimpleBlobDetector_create(parameters=params)
                #print ("-SimpleBlobDetector_Updated ")
                reload = False
            
            
            ##-------- Capturem els frames i guardem --------##
            ##-----els tipus d'iformacio que necessitem -----##
            
            fg.wait_for_frame(im)

            ##-------- Frame Pixel/Distancia --------##
            im_rdis = im.distance_image()
            ##--------- Frame Pixel/Amplitut --------##
            im_amp = im.amplitude_image()
            ##--------- Frame Pixel/Posicio XYZ --------##
            im_xyz = im.xyz_image()

            
            amplitut_color = cv2.applyColorMap(im_amp.astype(np.uint8), cv2.COLORMAP_BONE)

            resta = frame_inici - im_rdis
            resta2 = frame_inici2 - im_rdis

            resta_altura =  resta


            acum1 = 0
            acum2 = 0
            
            for i in range(171):
                for j in range(223):
                    
                    acum1 += resta[i][j]
                    acum2 += resta2[i][j]

                    if resta[i][j] > 0.01:
                        resta[i][j]= 0    #blanc
                    else:
                        resta[i][j]= 100  #negre

                    if resta_altura[i][j] > 0.01:
                        resta_altura[i][j]= 0   #blanc
                    else:
                        resta_altura[i][j]= 100
            
##-----Escollim dinàmicament en cada iteracio-------##
##--Si ens quedem amb el background1 o background2--##
            
        ##--Background2 valid--##
            
            if abs(acum1) > abs(acum2):
                frame_valid = 1
                resta = resta2
            
            ##--Background1 valid--##

            else:
                frame_valid = 2
            

            calcularInfoResta(resta, amplitut_color, detector)
            calcularInfoRestaAltura(resta_altura, amplitut_color, detector, llistaCentreMases, im_xyz)

            amplitut_color[86, 112] = [0, 0, 255]

            amplitut_color_lineas = amplitut_color


            
            
            
                
        
        




# App Globals (do not edit)
app = Flask(__name__)
#run_with_ngrok(app)
#app.config['BASIC_AUTH_USERNAME'] = 'amida4'
#app.config['BASIC_AUTH_PASSWORD'] = 'amida4'
#app.config['BASIC_AUTH_FORCE'] = True

#basic_auth = BasicAuth(app)
last_epoch = 0
@app.route('/')
#@basic_auth.required

def index():
    return render_template('videos_openCV.html')



def carreguemValorsXML():

    global maxAreaValue, minAreaValue, colorBlobValue, minDistBetweenBlobsValue, minThresholdValue, maxThresholdValue, thresholdStepValue, minRepeatabilityValue, minCircularityValue, maxCircularityValue, minConvexityValue, maxConvexityValue, minInertiaRatioValue, maxInertiaRatioValue

    file_name = 'templates/arxiu.xml'
    dom = ElementTree.parse(file_name)
    parametres = dom.findall('Parametres')
    root = dom.getroot()
    
    for p in parametres:
        
        minAreaValue = p.find('minArea').text
        maxAreaValue = p.find('maxArea').text
        colorBlobValue = p.find('colorBlob').text
        minDistBetweenBlobsValue = p.find('minDistBetweenBlobs').text
        minThresholdValue = p.find('minThreshold').text
        maxThresholdValue = p.find('maxThreshold').text
        thresholdStepValue = p.find('thresholdStep').text
        minRepeatabilityValue = p.find('minRepeatability').text
        minCircularityValue = p.find('minCircularity').text
        maxCircularityValue = p.find('maxCircularity').text
        minConvexityValue = p.find('minConvexity').text
        maxConvexityValue = p.find('maxConvexity').text
        minInertiaRatioValue = p.find('minInertiaRatio').text
        maxInertiaRatioValue = p.find('maxInertiaRatio').text



def calcularInfoResta(resta, amplitut_color, detector):
    
    global llistaCentreMases,resta_color, img_grey, img_contorns, binary, keyp
    global llista_bool, llista_definitiva, llista_definitiva2
    matriu_blanc = np.zeros((172,224), np.uint8)
    resta_color = cv2.applyColorMap(resta.astype(np.uint8), cv2.COLORMAP_TWILIGHT)
    img_grey = cv2.cvtColor(resta_color, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(img_grey)
    _, binary = cv2.threshold(img_grey, 100,255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_contorns = cv2.drawContours(matriu_blanc, contours, -1 ,(255,0,0), 1)

    llistaCentreMases = []
    llista_definitiva = [0,0,0,0,0,0]
    llista_definitiva2 = [0,0,0,0,0,0]
    
    

    if len(keypoints) > 0:
        
        for k in range(len(keypoints)):
            #print(len(keypoints))
            llista = []
            llista2 = []
            sumatori = 0
            
            x = keypoints[k].pt[0]
            y = keypoints[k].pt[1]
            
            cv2.putText(amplitut_color,str(k),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            for cnt in contours:
                
                area = cv2.contourArea(cnt)
                
                if area > 100 and area < 1200:   
                    
                    M = cv2.moments(cnt)
                    
                    if M['m00'] != 0:
                        
                        
                        #centroid
                        X = M['m10'] / M['m00']
                        Y = M['m01'] / M['m00']
                        
                        if [X,Y] not in llistaCentreMases:
                            
                            llistaCentreMases.append([X,Y])
                             
                    if M['mu20']-M['mu02'] != 0:
                        
                        grados_flagg = True
                        
                        theta = 0.5 * np.arctan (2 * M['mu11']/((M['mu20']-M['mu02'])))
                        grados = (theta / math.pi) * 180
                        grados = round(grados, 2)
                        
                        if grados < 0:
                            grados = 45 + (45 - abs(grados))
                        
                        
                        llista_definitiva[sumatori] = [sumatori, grados]
                        llista_definitiva2[sumatori] = [sumatori] 

                        llista_bool = True
                        sumatori += 1
  
    

    #print(llistaCentreMases)      
    keyp = cv2.drawKeypoints(amplitut_color,keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    


def calcularInfoRestaAltura(resta_altura, amplitut_color, detector, llistaCentreMases, im_xyz):
    
    global llista_definitiva ,llista_definitiva2, mitjanaReal, ite
    
    matriu_blanc = np.zeros((172,224), np.uint8)
    resta_altura = cv2.applyColorMap(resta_altura.astype(np.uint8), cv2.COLORMAP_TWILIGHT)
    img_grey_rest = cv2.cvtColor(resta_altura, cv2.COLOR_BGR2GRAY)
    keypoints_rest = detector.detect(img_grey_rest)
    _, binary_rest = cv2.threshold(img_grey_rest, 100,255, cv2.THRESH_BINARY)
    contours_rest, _ = cv2.findContours(binary_rest, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_contorns_rest = cv2.drawContours(matriu_blanc, contours_rest, -1 ,(255,0,0), 1)
  
    llista_cnt = []
    iteracions = 0
    ite = len(llistaCentreMases)
    
    if ite > 0:    
        #for k in range(len(llistaCentreMases)):
        mitjanaReal = [0 for i in range (ite)]
        #mitjanaReal =[0,0,0,0,0]   
        for cnt in contours_rest:
            x_sum = 0
            y_sum = 0
            z_sum = 0

            contador = 0
            area_rest = cv2.contourArea(cnt)
            
            if area_rest > 200 and area_rest < 1200:
            
                max_left = -10
                max_right = 10
                max_top = -10
                max_bottom = 10
                max_height = 10

                for y in range(0,171):
                    for x in range(0,223):
                        
                        dist = cv2.pointPolygonTest(cnt, (x,y), True)
                        if dist >= 0:
                            if im_xyz[y,x,1] > max_left:
                                max_left = round(im_xyz[y,x,1],4)
                            if im_xyz[y,x,1] < max_right:
                                max_right = round(im_xyz[y,x,1],4)
                            
                            if im_xyz[y,x,2] > max_top:
                                max_top = round(im_xyz[y,x,2],4)
                            if im_xyz[y,x,2] < max_bottom:
                                max_bottom = round(im_xyz[y,x,2],4)
                            
                            if im_xyz[y,x,0] < max_height:    
                                max_height = round(im_xyz[y,x,0],4)
                            
                            x_sum += im_xyz[y,x, 1] 
                            y_sum += im_xyz[y,x, 2] 
                            z_sum += im_xyz[y,x, 0] 

                            contador += 1


                #for cont in cnt:
                #    x_sum += im_xyz[cnt[contador][0][1],cnt[contador][0][0], 1] 
                #    y_sum += im_xyz[cnt[contador][0][1],cnt[contador][0][0], 2] 
                #    z_sum += im_xyz[cnt[contador][0][1],cnt[contador][0][0], 0] 
                    
                #    contador += 1
                
                if contador != 0:
                    
                    RealPointX = round(x_sum/contador, 4)
                    RealPointY = round(y_sum/contador, 4)
                    RealPointZ = round(z_sum/contador, 4)

                    #print(RealPointX, RealPointY, RealPointZ)

                    mitjanaReal[iteracions] = RealPointX, RealPointY, RealPointZ         
                   
                    llista_definitiva[iteracions].append(RealPointX)
                    llista_definitiva[iteracions].append(RealPointY)
                    llista_definitiva[iteracions].append(RealPointZ)

                    llista_definitiva2[iteracions].append(max_left)
                    llista_definitiva2[iteracions].append(max_right)
                    llista_definitiva2[iteracions].append(max_top)
                    llista_definitiva2[iteracions].append(max_bottom)
                    llista_definitiva2[iteracions].append(max_height)

                    iteracions += 1

@app.route('/2')
def video2():
    global amplitut_color
    print(amplitut_color)
    
    return Response(function(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/function')
def function():
    while 1==1:
    
        ret, jpeg = cv2.imencode('.jpg',amplitut_color)

        if jpeg is not None:

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            print("frame is none")



def main():
    global im_xyz
    th = myThread("th")
    th.start()
    
    
    #while True:
    #    print(amplitut_color)
    
if __name__ == '__main__':
    main()
    app.run(host='192.168.0.80', debug=True, threaded=True)

