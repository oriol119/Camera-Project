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
from redis import Redis

ampl = 0
uri = 0

cli = Redis('localhost')
resta_calib = 0

llista_definitiva = [0,0,0,0,0,0]
llista_definitiva2 = [0,0,0,0,0,0]



class myTimer(threading.Thread):

    def __init__ (self, name):
        threading.Thread.__init__(self)
        self.name = name

    def timeout(self):
        print("HHELLLOO")
    
    def run(self):
        #global fg, im
     
        time.sleep(5)
        while True:
            #fg.wait_for_frame(im)

            #start = time.time()
            #captura1 = im.amplitude_image()
            #time.sleep(1)
            
            #duration = time.time() - start 
            #captura2 = im.amplitude_image()

            e =3



class myThread(threading.Thread):

    def __init__ (self, name):
        
        threading.Thread.__init__(self)
        self.name = name
    
    def run(self):

        global llista, llista_bool
        global static_frame_flagg, flaggflagg
        global frame_inici, frame_inici_amplitut, frame_inici2, frame_inici_amplitut2
        global miarea, maarea
        global maxAreaValue, minAreaValue, blobColorValue, minDistBetweenBlobsValue, minThresholdValue, maxThresholdValue, thresholdStepValue, minRepeatabilityValue, minCircularityValue, maxCircularityValue, minConvexityValue, maxConvexityValue, minInertiaRatioValue, maxInertiaRatioValue
        global frame_valid, detector
        global llistaCentreMases,resta_color, img_grey, img_contorns, binary
        global im_xyz, im_rdis
        global im_amp
        global amplitut_color_lineas
        global resta
        global ampl
        global amplitut_color
        global fg, im
        global minArea, maxArea, blobcolor, minDistBetweenBlobs, minThreshold, maxThreshold, thresholdStep, minRepeatability, minCircularity, maxCircularity, minConvexity, maxConvexity, minInertiaRatio, maxInertiaRatio
        global frame_flagg, frame_flagg2

        frame_flagg = 0
        frame_flagg2 = 0
                
        grados_flagg = False
        done = False
        reload = False
        contador = 0
        llista = []
        
        ##-- Agafem la informació del background que tenim --##
        ##------- actualment(ho indica frame_flagg) ---------##

        if frame_flagg == 0:
            background_static()

        if frame_flagg2 == 0:
            background_static2()


        ##-----Funcio que llegeix els parametres del XML-----##
        
        minAreaValue, maxAreaValue, blobColorValue, minDistBetweenBlobsValue,\
        minThresholdValue, maxThresholdValue, thresholdStepValue, \
        minRepeatabilityValue, minCircularityValue, maxCircularityValue,\
        minConvexityValue, maxConvexityValue, minInertiaRatioValue, \
        maxInertiaRatioValue = carreguemParametresXML() 
    

        ##---------Assignem el valors llegits del XML---------##        
        
        params = assignarValorsXML()
        
        
        ##-------------Creem blob detector amb------------ ##
        ##----------- els valors dels parametres-----------## 

        detector = cv2.SimpleBlobDetector_create(parameters=params)
        
        img_blanca = 255 * np.ones((172,224,3), dtype=np.uint8)
        
        llista = []

        shared_var = 0

        fg = ifm3dpy.FrameGrabber(ifm3dpy.Camera(), ifm3dpy.IMG_AMP | ifm3dpy.IMG_RDIS | ifm3dpy.IMG_CART)
        im = ifm3dpy.ImageBuffer()

        
        while not done:
        
        ##-------- Ens mantenim constantment observant -------##
        ##------- si canviem el valor d'algun parametre ------##
            param_flagg = str(cli.get('param_flagg'))
            param_flagg = param_flagg[2:-1]
            
            if param_flagg == 'True':
                
                minArea, maxArea, blobColor, minDistBetweenBlobs, minThreshold, \
                maxThreshold, thresholdStep, minRepeatability, minCircularity, \
                maxCircularity, minConvexity, maxConvexity, minInertiaRatio, \
                maxInertiaRatio = getChangedParams()

                param_flagg = False
                param_flagg = cli.set('param_flagg', str(param_flagg))

                if minArea != False:
                    params.minArea = int(minArea)
                    print ("minArea:",minArea)
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
                
            minAreaValue, maxAreaValue, blobColorValue, minDistBetweenBlobsValue,\
            minThresholdValue, maxThresholdValue, thresholdStepValue, \
            minRepeatabilityValue, minCircularityValue, maxCircularityValue,\
            minConvexityValue, maxConvexityValue, minInertiaRatioValue, \
            maxInertiaRatioValue = carreguemParametresXML()
            
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

            p = cv2.applyColorMap(im_amp.astype(np.uint8), cv2.COLORMAP_BONE)
            ret, jpeg = cv2.imencode('.jpg',p)
            img_bytes = jpeg.tobytes()
            cli.set('amplitut', img_bytes)
            
            
            ##--------- Frame Pixel/Posicio XYZ --------##
            im_xyz = im.xyz_image()
           
        
            ##------------ CAPTUREM FRAME -------------##
            
            treureFrame = str(cli.get('treureFrame'))
            treureFrame = treureFrame[2:-1]

            if (treureFrame == 'True'):

                FuncTreureFrame()           

            ##------------ CAPTUREM FRAME2 -------------##

            treureFrame2 = str(cli.get('treureFrame2'))
            treureFrame2 = treureFrame2[2:-1]

            if (treureFrame2 == 'True'):
                
                FuncTreureFrame2()

            ##------------ GUARDEM FRAME -------------##

            guardarFrame = str(cli.get('guardarFrame'))
            guardarFrame = guardarFrame[2:-1]  

            if (guardarFrame == 'True'):
                
                FuncGuardarFrame()
                
            ##------------ GUARDEM FRAME2 -------------##

            guardarFrame2 = str(cli.get('guardarFrame2'))
            guardarFrame2 = guardarFrame2[2:-1]  

            if (guardarFrame2 == 'True'):
                
                FuncGuardarFrame2()

            
            amplitut_color = cv2.applyColorMap(im_amp.astype(np.uint8), cv2.COLORMAP_BONE)
            
            resta, resta_altura = calcularBackgroundDinamic(frame_inici, frame_inici2, im_rdis,)
            
            calcularInfoResta(resta, amplitut_color, detector)

            calcularInfoRestaAltura(resta_altura, amplitut_color, detector, llistaCentreMases, im_xyz)

            amplitut_color[86, 112] = [0, 0, 255]

            amplitut_color_lineas = amplitut_color
            
            ampl = amplitut_color
 
            calibflag2 = str(cli.get('calibflag2'))
            calibflag2 = calibflag2[2:-1]


            ##---------- USUARI PULSA BOTO CALIBRAR -----------##
            
            if calibflag2 == 'True':
                
                matriu_blanc = np.zeros((172,224), np.uint8)
                resta_color = cv2.applyColorMap(resta.astype(np.uint8), cv2.COLORMAP_TWILIGHT)
                img_grey = cv2.cvtColor(resta_color, cv2.COLOR_BGR2GRAY)
                keypoints_C = detector.detect(img_grey)
                _, binary = cv2.threshold(img_grey, 100,255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                img_contorns = cv2.drawContours(matriu_blanc, contours, -1 ,(255,0,0), 1)
                
                calib_flagg = 0
                llistaCentreMases_C = []
            
                num_elements = 0
                if len(llistaCentreMases) > 0:

                    for k in range(len(llistaCentreMases)):
                        
                        llista = []
                        llista2 = []
            
                        num_elements += 1
                        
                        for cnt in contours:
                            
                            area = cv2.contourArea(cnt)
                            
                            if area > 100 and area < 1200:   
                                
                                M = cv2.moments(cnt)
                                
                                if M['m00'] != 0:
                                    
                                    #centroid
                                    X = M['m10'] / M['m00']
                                    Y = M['m01'] / M['m00']
                                    
                                    if [X,Y] not in llistaCentreMases_C:
                                        
                                        llistaCentreMases_C.append([X,Y])

                                    llista_bool = 'True'
                                
                     
                keyp = cv2.drawKeypoints(amplitut_color,keypoints_C, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                if num_elements == 3:
                
                    # Unim el centre dels 3 elements de calibracio
                    cv2.line(amplitut_color_lineas,(int(llistaCentreMases[0][0]),int(llistaCentreMases[0][1])),(int(llistaCentreMases[1][0]),int(llistaCentreMases[1][1])),(0,255,0), 1)
                    cv2.line(amplitut_color_lineas,(int(llistaCentreMases[0][0]),int(llistaCentreMases[0][1])),(int(llistaCentreMases[2][0]),int(llistaCentreMases[2][1])),(0,255,0), 1)
                    cv2.line(amplitut_color_lineas,(int(llistaCentreMases[2][0]),int(llistaCentreMases[2][1])),(int(llistaCentreMases[1][0]),int(llistaCentreMases[1][1])),(0,255,0), 1)

                    # Guardem l'imatge  
                    scipy.misc.imsave('calibracio.jpg', amplitut_color_lineas)


                    calcularInfoRestaAltura(resta, amplitut_color, detector, llistaCentreMases, im_xyz)
                    
                    P0_X, P0_Y, P0_Z, P1_X, P1_Y, P1_Z, P2_X, \
                    P2_Y, P2_Z = agafarPunts(llistaCentreMases)
            
                    distancia1, distancia2, distancia3 = calcularDistancies(P0_X, \
                    P0_Y, P0_Z, P1_X, P1_Y, P1_Z, P2_X, P2_Y, P2_Z)                  
                
                    Punt0, Punt1, Punt2 = calcularPuntsWorld(distancia1,distancia2, \
                    distancia3, P0_X, P0_Y, P0_Z, P1_X, P1_Y, P1_Z, P2_X, P2_Y, P2_Z)
                
                

                    if abs(P0_Z - P1_Z) > 0.01:
                        print("Error: L'alçada dels objectes no es vàlida")
                        calib_flagg = 4
                        
                    if abs(P1_Z - P2_Z) > 0.01:
                        print("Error: L'alçada dels objectes no es vàlida")
                        calib_flagg = 4
                    
                    if abs(P2_Z - P0_Z) > 0.01:
                        print("Error: L'alçada dels objectes no es vàlida")
                        calib_flagg = 4
                

                    angle_rad, angle_rad2, angle_graus, angle_graus2 = \
                    calcularAngles(Punt0,Punt1,Punt2)
                    
                    desplaçament = [Punt0[0],Punt0[1]]

                    Punt0_wp, Punt1_wp, Punt2_wp = calcularPuntsWorldPrima(desplaçament, Punt0, Punt1, Punt2)
                    
                    Punt0_usr, Punt1_usr, Punt2_usr = calcularPuntsUser(angle_rad, Punt0_wp, Punt1_wp, Punt2_wp)
           
                
                elif num_elements == 2:
                    calib_flagg = 1
                    print("Error: Falta detectar 1 element per poder calibrar")  
                
                elif num_elements == 1:
                    calib_flagg = 2
                    print("Error: Falten detectar 2 elements per poder calibrar")

                elif num_elements == 0:
                    calib_flagg = 3
                    print("Error: Falten detectar 3 elements per poder calibrar")


                cli.set('calib_flagg', str(calib_flagg))
                calibflag2 = False
                cli.set('calibflag2', str(calibflag2))



############################## FUNCIONS ################################### 
############################## FUNCIONS ################################### 
############################## FUNCIONS ################################### 



######################## carreguemParametresXML ###########################
#                                                                         #
# -PRE-CONDITION:                                                         #  
# --> (file_name): Ha d'existir parametres.xml dins de la carpeta         #
#     templates del projecte. Totes les etiquetes dels paràmetres que     #
#     volem llegir han d'estar dins de l'etiqueta:                        #
#                                                                         #
#                       <Parametres> </Parametres>                        #
#                                                                         #      
######################## carreguemParametresXML ###########################


def carreguemParametresXML():

    global param_flagg 
    file_name = 'templates/parametres.xml'
    dom = ElementTree.parse(file_name)
    parametres = dom.findall('Parametres')
    root = dom.getroot()
    
    for p in parametres:
        
        minAreaValue = p.find('minArea').text
        maxAreaValue = p.find('maxArea').text
        blobColorValue = p.find('colorBlob').text
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
        
      
    cli.set('minAreaValue', minAreaValue)
    cli.set('maxAreaValue', maxAreaValue)
    cli.set('blobColorValue', blobColorValue)
    cli.set('minDistBetweenBlobsValue', minDistBetweenBlobsValue)
    cli.set('minThresholdValue', minThresholdValue)
    cli.set('maxThresholdValue', maxThresholdValue)
    cli.set('thresholdStepValue', thresholdStepValue)
    cli.set('minRepeatabilityValue', minRepeatabilityValue)
    cli.set('minCircularityValue', minCircularityValue)
    cli.set('maxCircularityValue', maxCircularityValue)
    cli.set('minConvexityValue', minConvexityValue)
    cli.set('maxConvexityValue', maxConvexityValue)
    cli.set('minInertiaRatioValue', minInertiaRatioValue)
    cli.set('maxInertiaRatioValue', maxInertiaRatioValue)


    return minAreaValue, maxAreaValue, blobColorValue, minDistBetweenBlobsValue, minThresholdValue, maxThresholdValue, thresholdStepValue, minRepeatabilityValue, minCircularityValue, maxCircularityValue, minConvexityValue, maxConvexityValue, minInertiaRatioValue, maxInertiaRatioValue


########################### assignarValorsXML #############################
#                                                                         #
# -PRE-CONDITION:                                                         #  
# --> La funció carreguemParametresXML ha de retornar tots els valors     # 
#     correctament(son els parametres d'aquesta funcio). Tots els valors  #
#     han de ser números.                                                 #
#                                                                         # 
########################### assignarValorsXML #############################


def assignarValorsXML():

    params = cv2.SimpleBlobDetector_Params()
        
    params.filterByArea = True              # default: True
    params.minArea = int(minAreaValue)      # default: 50
    params.maxArea = int(maxAreaValue)      # default: 5000

    params.filterByCircularity = False                    # default: False
    params.minCircularity = float(minCircularityValue)    # default: 0.8
    params.maxCircularity = float(maxCircularityValue)    # default: inf   

    params.filterByConvexity = False                   # default: True
    params.minConvexity = float(minConvexityValue)     # default: 0.95
    params.maxConvexity = float(maxConvexityValue)     # default: Inf

    params.filterByInertia = False                          # default: True
    params.minInertiaRatio = float(minInertiaRatioValue)    # default: 0.1
    params.maxInertiaRatio = float(maxInertiaRatioValue)    # default: Inf

    params.minRepeatability = 1                         # default: 2

    params.minDistBetweenBlobs = int(minDistBetweenBlobsValue) # default: 10

    params.thresholdStep = int(thresholdStepValue)        # default: 10
    params.minThreshold = int(minThresholdValue)          # default: 50
    params.maxThreshold = int(maxThresholdValue)          # default: 220


    params.filterByColor = False                    # default: True
    params.blobColor = int(blobColorValue)          # default: 0

    return params


####################### calcularBackgroundDinamic #########################
#                                                                         #
# -INVARIANT:                                                             #
# --> frame_inici: Array 2d (224x172) on cada element es la distancia en  #
#     metres des del centre de la camera fins al element.                 #
# --> frame_inici2: Array 2d (224x172), elem = dist(camera, elem)         #  
# --> im_rdis: Array 2d (224x172), elem = dist(camera, elem)              #
#                                                                         #
# -PRE-CONDITION:                                                         #
# --> frame_inici: Frame Background1 que tenim guardat                    #
# --> frame_inici2: Frame Background2 que tenim guardat                   #
# --> im_rdis: Frame actual (per poder fer la resta)                      #
#                                                                         #  
# -POST-CONDITION:                                                        #  
# --> resta: Array 2d (224x172) on cada element es 0 o 100.               #
# ------>   0 (blanc): Al fer la resta l'element es mes gran que el valor #
#                      indicat. Es part del objecte que busquem.          #  
# ------>   1 (negre): Al restar, l'element es mes petit que el valor     #
#                      indicat, No es part del objecte que busquem.       #
# --> resta_altura: Es igual que resta, canviant l'alçada a partir de la  #
#     que calculem la resta.                                              #
#                                                                         #
####################### calcularBackgroundDinamic #########################


def calcularBackgroundDinamic(frame_inici, frame_inici2, im_rdis):
       
    resta = frame_inici - im_rdis
    resta2 = frame_inici2 - im_rdis

    resta_altura =  resta

    cli.set('resta', str(resta))

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
    
    cli.set('frame_valid', frame_valid)
    
    return [resta, resta_altura]


def getChangedParams():

    minArea = str(cli.get('minArea'))
    maxArea = str(cli.get('maxArea'))
    blobColor = str(cli.get('blobColor'))
    minDistBetweenBlobs = str(cli.get('minDistBetweenBlobs'))
    minThreshold = str(cli.get('minThreshold'))
    maxThreshold = str(cli.get('maxThreshold'))
    thresholdStep = str(cli.get('thresholdStep'))
    minRepeatability = str(cli.get('minRepeatability'))
    minCircularity = str(cli.get('minCircularity'))
    maxCircularity = str(cli.get('maxCircularity'))
    minConvexity = str(cli.get('minConvexity'))
    maxConvexity = str(cli.get('maxConvexity'))
    minInertiaRatio = str(cli.get('minInertiaRatio'))
    maxInertiaRatio = str(cli.get('maxInertiaRatio'))

    minArea = minArea[2:-1]
    maxArea = maxArea[2:-1]
    blobColor = blobColor[2:-1]
    minDistBetweenBlobs = minDistBetweenBlobs[2:-1]
    minThreshold = minThreshold[2:-1]
    maxThreshold = maxThreshold[2:-1]
    thresholdStep = thresholdStep[2:-1]
    minRepeatability = minRepeatability[2:-1]
    minCircularity = minCircularity[2:-1]
    maxCircularity = maxCircularity[2:-1]
    minConvexity = minConvexity[2:-1]
    maxConvexity = maxConvexity[2:-1]
    minInertiaRatio = minInertiaRatio[2:-1]
    maxInertiaRatio = maxInertiaRatio[2:-1]

    return [minArea, maxArea, blobColor, minDistBetweenBlobs, minThreshold, maxThreshold, thresholdStep, minRepeatability, minCircularity, maxCircularity, minConvexity, maxConvexity, minInertiaRatio, maxInertiaRatio]


def FuncTreureFrame():

    treureFrame = False
    cli.set('treureFrame', str(treureFrame))
    
    frame_inici_amplitut = im.amplitude_image()

    frame_inici = im.distance_image()

    frame_inici.dump('AMIDA4.dat')

    scipy.misc.imsave('AMIDA4.jpg', frame_inici_amplitut)

    frame_escollit = 'AMIDA4.jpg'
    cli.set('frame_escollit', frame_escollit)


def FuncTreureFrame2():

    treureFrame2 = False
    cli.set('treureFrame2', str(treureFrame2))
    
    frame_inici_amplitut2 = im.amplitude_image()

    frame_inici2 = im.distance_image()

    frame_inici2.dump('AMIDA4_2.dat')

    scipy.misc.imsave('AMIDA4_2.jpg', frame_inici_amplitut2)

    frame_escollit2 = 'AMIDA4_2.jpg'
    cli.set('frame_escollit2', frame_escollit2)


def FuncGuardarFrame():

    guardarFrame = False
    cli.set('guardarFrame', str(guardarFrame))
    frame_inici.dump('background1.dat')
    scipy.misc.imsave('background1.png', frame_inici_amplitut)


def FuncGuardarFrame2():

    guardarFrame2 = False
    cli.set('guardarFrame2', str(guardarFrame2))
    frame_inici2.dump('background2.dat')
    scipy.misc.imsave('background2.png', frame_inici_amplitut2)


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
  

    if len(keypoints) > 0:
        
        for k in range(len(keypoints)):
      
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
                    else:
                        llista_bool = False


                    cli.set('llista_bool', str(llista_bool))

    keyp = cv2.drawKeypoints(amplitut_color,keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ret, jpeg = cv2.imencode('.jpg',keyp)
    keyp_bytes = jpeg.tobytes()
     
    cli.set('keypoints', keyp_bytes)


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
            
            if area_rest > 100 and area_rest < 1200:
            
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

                
                if contador != 0:

                    if len(llistaCentreMases) > 0:  
                          
                        RealPointX = round(x_sum/contador, 4)
                        RealPointY = round(y_sum/contador, 4)
                        RealPointZ = round(z_sum/contador, 4)

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

    cli.set('llista_definitiva', str(llista_definitiva))
    cli.set('llista_definitiva2', str(llista_definitiva2))


def agafarPunts(llistaCentreMases):

    P0_X = im_xyz[int(llistaCentreMases[0][1]),int(llistaCentreMases[0][0]),1]
    P0_Y = im_xyz[int(llistaCentreMases[0][1]),int(llistaCentreMases[0][0]),2]
    P0_Z = im_xyz[int(llistaCentreMases[0][1]),int(llistaCentreMases[0][0]),0]
    
    P1_X = im_xyz[int(llistaCentreMases[1][1]),int(llistaCentreMases[1][0]),1]
    P1_Y = im_xyz[int(llistaCentreMases[1][1]),int(llistaCentreMases[1][0]),2]
    P1_Z = im_xyz[int(llistaCentreMases[1][1]),int(llistaCentreMases[1][0]),0]
    
    P2_X = im_xyz[int(llistaCentreMases[2][1]),int(llistaCentreMases[2][0]),1]
    P2_Y = im_xyz[int(llistaCentreMases[2][1]),int(llistaCentreMases[2][0]),2]
    P2_Z = im_xyz[int(llistaCentreMases[2][1]),int(llistaCentreMases[2][0]),0]

    return P0_X, P0_Y, P0_Z, P1_X, P1_Y, P1_Z, P2_X, P2_Y, P2_Z


def calcularPuntsWorld(distancia1,distancia2,distancia3, P0_X, P0_Y, \
                        P0_Z, P1_X, P1_Y, P1_Z, P2_X, P2_Y, P2_Z):
    
    if distancia1 > distancia2 and distancia1 > distancia3:
        Punt0 = [P2_X, P2_Y, P2_Z]
        if distancia2 > distancia3:
            Punt1 = [P1_X, P1_Y, P1_Z]
            Punt2 = [P0_X, P0_Y, P0_Z]
        else:
            Punt1 = [P0_X, P0_Y, P0_Z]
            Punt2 = [P1_X, P1_Y, P1_Z]

    elif distancia2 > distancia1 and distancia2 > distancia3:
        Punt0 = [P0_X, P0_Y, P0_Z]
        if distancia1 > distancia3:
            Punt1 = [P1_X, P1_Y, P1_Z]
            Punt2 = [P0_X, P0_Y, P0_Z]
        else:
            Punt1 = [P0_X, P0_Y, P0_Z]
            Punt2 = [P1_X, P1_Y, P1_Z]
        
    elif distancia3 > distancia1 and distancia3 > distancia2:
        Punt0 = [P1_X, P1_Y, P1_Z]
        if distancia1 > distancia2:
            Punt1 = [P0_X, P0_Y, P0_Z]
            Punt2 = [P2_X, P2_Y, P2_Z]
        else:
            Punt1 = [P2_X, P2_Y, P2_Z]
            Punt2 = [P0_X, P0_Y, P0_Z]
    
    cli.set('Punt0', str(Punt0))
    cli.set('Punt1', str(Punt1))
    cli.set('Punt2', str(Punt2))

    return Punt0, Punt1, Punt2


def calcularDistancies(P0_X, P0_Y, P0_Z, P1_X, P1_Y, P1_Z, P2_X, P2_Y, P2_Z):
    distancia1 = math.sqrt(((P1_X-P0_X)**2)+((P1_Y-P0_Y)**2))
    distancia2 = math.sqrt(((P1_X-P2_X)**2)+((P1_Y-P2_Y)**2))
    distancia3 = math.sqrt(((P0_X-P2_X)**2)+((P0_Y-P2_Y)**2))

    return distancia1, distancia2, distancia3


def calcularAngles():
    
    angle_rad = math.atan2((Punt0[0]-Punt1[0]),(Punt0[1]-Punt1[1]))
    angle_rad2 = math.atan2((Punt1[0]-Punt0[0]),(Punt1[1]-Punt0[1]))

    angle_graus = (angle_rad * 180)/ math.pi
    angle_graus2 = (angle_rad2 * 180)/ math.pi

    return angle_rad, angle_rad2, angle_graus, angle_graus2


def calcularPuntsWorldPrima(desplaçament, Punt0, Punt1, Punt2):
    
    Punt0_wp = [Punt0[0] - desplaçament[0], Punt0[1] - desplaçament[1], Punt0[2]]
    Punt1_wp = [Punt1[0] - desplaçament[0], Punt1[1] - desplaçament[1], Punt1[2]]
    Punt2_wp = [Punt2[0] - desplaçament[0], Punt2[1] - desplaçament[1], Punt2[2]]

    cli.set('Punt0_wp', str(Punt0_wp))
    cli.set('Punt1_wp', str(Punt1_wp))
    cli.set('Punt2_wp', str(Punt2_wp))

    return Punt0_wp, Punt1_wp, Punt2_wp


def calcularPuntsUser(angle_rad, Punt0_wp, Punt1_wp, Punt2_wp):
    
    Punt0_usr = [(Punt0_wp[0] * math.cos(angle_rad))-(Punt0_wp[1] * math.sin(angle_rad)), 
    (Punt0_wp[0] * math.sin(angle_rad))-(Punt0_wp[1] * math.cos(angle_rad))]

    Punt1_usr = [(Punt1_wp[0] * math.cos(angle_rad))-(Punt1_wp[1] * math.sin(angle_rad)), 
    (Punt1_wp[0] * math.sin(angle_rad))-(Punt1_wp[1] * math.cos(angle_rad))]

    Punt2_usr = [(Punt2_wp[0] * math.cos(angle_rad))-(Punt2_wp[1] * math.sin(angle_rad)), 
    (Punt2_wp[0] * math.sin(angle_rad))-(Punt2_wp[1] * math.cos(angle_rad))]
    
    cli.set('Punt0_usr', str(Punt0_usr))
    cli.set('Punt1_usr', str(Punt1_usr))
    cli.set('Punt2_usr', str(Punt2_usr))


def background_static():
    global frame_inici, frame_inici_amplitut,frame_escollit
    
    image = cv2.imread('background1.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    image_ok = cv2.resize(image, (224,172))
    matriu = np.load('background1.dat', allow_pickle=True)
    
    #imatge que visualitzem
    frame_escollit = 'background1.png'
    
    #enviem el nom de la imatge
    cli.set('frame_escollit', frame_escollit)

    #matriu amb la informacio de la imatge
    frame_inici = matriu
    frame_inici_amplitut = image_ok

    ret, jpeg = cv2.imencode('.jpg',image_ok)
    if jpeg is not None:
        
        return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    

def background_static2():

    global frame_inici2, frame_inici_amplitut2, frame_escollit2, mostra2

    image = cv2.imread('background2.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    image_ok = cv2.resize(image, (224,172))
    matriu2 = np.load('background2.dat', allow_pickle=True)
    
    frame_escollit2 = 'background2.png'

    #enviem el nom de la imatge
    cli.set('frame_escollit2', frame_escollit2)

    frame_inici2 = matriu2
    frame_inici_amplitut2 = image_ok

    ret, jpeg = cv2.imencode('.jpg',image_ok)
    if jpeg is not None:
        
        return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

