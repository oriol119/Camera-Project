from importlib import import_module

from flask import Flask, render_template, Response, request, send_file
from xml.etree import ElementTree
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Polygon
from redis import Redis

named_libs = [('numpy', 'np'), ('matplotlib.pyplot', 'plt'), ('seaborn','sns')]

for (name,short) in named_libs:
    try:
        lib= import_module(name)
    except ImportError as err:
        print ('Error:', err)
    else:
        globals()[short] = lib

libnames = ['cv2','sys','time','threading', 'ifm3dpy', 'math', 'os', 'scipy.misc', 'PIL', 'io', 'SingleThread' ]
for libname in libnames:
    try:
        lib= import_module(libname)
    except ImportError as err:
        print('Error:', err)
        
    else:
        globals()[libname] = lib



global miarea, maarea, flaggflagg
flaggflagg = False
global frame_inici, frame_inici_amplitut, frame_inici2, frame_inici_amplitut2
global maxAreaValue, minAreaValue, blobColorValue, minDistBetweenBlobsValue, minThresholdValue, maxThresholdValue, thresholdStepValue, minRepeatabilityValue, minCircularityValue, maxCircularityValue, minConvexityValue, maxConvexityValue, minInertiaRatioValue, maxInertiaRatioValue


param_flagg = False

global llista, llista_bool
llista = []
llista_bool = False

global static_frame_flagg
static_frame_flagg = False

global frame_escollit, frame_escollit2
global im_xyz, im_rdis


global frame_flagg, frame_flagg2

frame_flagg = 0
frame_flagg2 = 0

global mostra, mostra2

global calib_flagg
calib_flagg = 0

global headings, headings2


amplitut_color = 0

cli = Redis('localhost')

calibflag2 = False
cli.set('calibflag2', str(calibflag2))

param_flagg = False
cli.set('param_flagg', str(param_flagg))


headings = ("ID","Angle","POS X","POS Y","POS Z")
headings2 = ("ID", "RIGHT","LEFT", "BOTTOM", "TOP", "HEIGHT")

#email_update_interval = 600 # sends an email only once in this time interval
#video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically
#object_classifier = cv2.CascadeClassifier("models/fullbody_recognition_model.xml") # an opencv classifier

# App Globals (do not edit)
app = Flask(__name__)

#run_with_ngrok(app)
#app.config['BASIC_AUTH_USERNAME'] = 'amida4'
#app.config['BASIC_AUTH_PASSWORD'] = 'amida4'
#app.config['BASIC_AUTH_FORCE'] = True
fig,ax = plt.subplots()
ax = sns.set_style(style="darkgrid")

#basic_auth = BasicAuth(app)
last_epoch = 0
@app.route('/')
#@basic_auth.required
def index():
    return render_template('videos_openCV.html')


@app.route("/angles", methods=["POST", "GET"])
def angles():
    global llista_bool
    llista_bool = cli.get('llista_bool')

    ##------ FROM STRING TO ARRAY ------##
    lst = str(cli.get('llista_definitiva'))
    lst2 = str(cli.get('llista_definitiva2'))
    
    lst = lst.replace("b",'',1)
    lst = lst.replace("'",'',2)

    lst2 = lst2.replace("b",'',1)
    lst2 = lst2.replace("'",'',2)

 
    llista_definitiva = eval(lst)
    llista_definitiva2 = eval(lst2)

    #print(llista_definitiva, llista_definitiva2)
  
    return render_template("angles.html", llista_bool = llista_bool, llista_definitiva = llista_definitiva, headings = headings, headings2 = headings2, llista_definitiva2 = llista_definitiva2)


@app.route("/configuracio", methods=["POST", "GET"])
def staticFrame():
    global frame_valid, treureFrame, calibflag2
    treureFrame = False
    cli.set('treureFrame', str(treureFrame))
    calibflag2 = False
    cli.set('calibflag2', str(calibflag2))

    frame_valid = str(cli.get('frame_valid'))
    frame_valid = frame_valid[2:-1]
 
   


    return render_template("configuracio.html", frame_valid = frame_valid)


@app.route('/video_amplitut')
def video_amplitut():
    return Response(amplitut(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/amplitut')
def amplitut():
  
    global video_bool  
    video_bool = 2

    while video_bool == 2:
        
        #b = SingleThread.amplitut_color

        a = cli.get('amplitut')

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + a + b'\r\n\r\n')
        

@app.route('/video_keypoints')
def video_keypoints():
    return Response(keypoints(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/keypoints')
def keypoints():

    
    global video_bool
    video_bool = 1

    while video_bool == 1:
    
        b = cli.get('keypoints')

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + b + b'\r\n\r\n')
        

@app.route("/parametres", methods=["POST", "GET"])
def params():
    global minArea, maxArea, blobcolor, minDistBetweenBlobs, minThreshold, maxThreshold, thresholdStep, minRepeatability, minCircularity, maxCircularity, minConvexity, maxConvexity, minInertiaRatio, maxInertiaRatio
    global param_flagg
    calibflag2 = False
    cli.set('calibflag2', str(calibflag2))

    video_keypoints()
    
    
    minAreaValue = str(cli.get('minAreaValue')) 
    
    maxAreaValue = str(cli.get('maxAreaValue'))
    blobColorValue = str(cli.get('blobColorValue'))
    minDistBetweenBlobsValue = str(cli.get('minDistBetweenBlobsValue'))
    minThresholdValue = str(cli.get('minThresholdValue'))
    maxThresholdValue = str(cli.get('maxThresholdValue'))
    thresholdStepValue = str(cli.get('thresholdStepValue'))
    minRepeatabilityValue = str(cli.get('minRepeatabilityValue'))
    minCircularityValue = str(cli.get('minCircularityValue'))
    maxCircularityValue = str(cli.get('maxCircularityValue'))
    minConvexityValue = str(cli.get('minConvexityValue'))
    maxConvexityValue = str(cli.get('maxConvexityValue'))
    minInertiaRatioValue = str(cli.get('minInertiaRatioValue'))
    maxInertiaRatioValue = str(cli.get('maxInertiaRatioValue'))

    minAreaValue = minAreaValue.replace("b",'')
    minAreaValue = minAreaValue.replace("'",'')
    maxAreaValue = maxAreaValue.replace("b",'')
    maxAreaValue = maxAreaValue.replace("'",'')
    blobColorValue = blobColorValue.replace("b",'')
    blobColorValue = blobColorValue.replace("'",'')
    minDistBetweenBlobsValue = minDistBetweenBlobsValue.replace("b",'')
    minDistBetweenBlobsValue = minDistBetweenBlobsValue.replace("'",'')
    minThresholdValue = minThresholdValue.replace("b",'')
    minThresholdValue = minThresholdValue.replace("'",'')
    maxThresholdValue = maxThresholdValue.replace("b",'')
    maxThresholdValue = maxThresholdValue.replace("'",'')
    thresholdStepValue = thresholdStepValue.replace("b",'')
    thresholdStepValue = thresholdStepValue.replace("'",'')
    minRepeatabilityValue = minRepeatabilityValue.replace("b",'')
    minRepeatabilityValue = minRepeatabilityValue.replace("'",'')
    minCircularityValue = minCircularityValue.replace("b",'')
    minCircularityValue = minCircularityValue.replace("'",'')
    maxCircularityValue = maxCircularityValue.replace("b",'')
    maxCircularityValue = maxCircularityValue.replace("'",'')
    minConvexityValue = minConvexityValue.replace("b",'')
    minConvexityValue = minConvexityValue.replace("'",'')
    maxConvexityValue = maxConvexityValue.replace("b",'')
    maxConvexityValue = maxConvexityValue.replace("'",'')
    minInertiaRatioValue = minInertiaRatioValue.replace("b",'')
    minInertiaRatioValue = minInertiaRatioValue.replace("'",'')
    maxInertiaRatioValue = maxInertiaRatioValue.replace("b",'')
    maxInertiaRatioValue = maxInertiaRatioValue.replace("'",'')
  
    
    if request.method == "POST":
        
        param_flagg = True
        cli.set('param_flagg', str(param_flagg))
  
        #flag_lectura = False
        flaggflagg = True
        minArea = request.form["minArea"]
        maxArea = request.form["maxArea"]
        blobColor = request.form["blobColor"]
        minDistBetweenBlobs = request.form["minDistBetweenBlobs"]
        minThreshold = request.form["minThreshold"]
        maxThreshold = request.form["maxThreshold"]
        thresholdStep = request.form["thresholdStep"]
        minRepeatability = request.form["minRepeatability"]
        minCircularity = request.form["minCircularity"]
        maxCircularity = request.form["maxCircularity"]
        minConvexity = request.form["minConvexity"]
        maxConvexity = request.form["maxConvexity"]
        minInertiaRatio = request.form["minInertiaRatio"]
        maxInertiaRatio = request.form["maxInertiaRatio"]

        cli.set('minArea', str(minArea))
        cli.set('maxArea', str(maxArea))
        cli.set('blobColor', str(blobColor))
        cli.set('minDistBetweenBlobs', str(minDistBetweenBlobs))
        cli.set('minThreshold', str(minThreshold))
        cli.set('maxThreshold', str(maxThreshold))
        cli.set('thresholdStep', str(thresholdStep))
        cli.set('minRepeatability', str(minRepeatability))
        cli.set('minCircularity', str(minCircularity))
        cli.set('maxCircularity', str(maxCircularity))
        cli.set('minConvexity', str(minConvexity))
        cli.set('maxConvexity', str(maxConvexity))
        cli.set('minInertiaRatio', str(minInertiaRatio))
        cli.set('maxInertiaRatio', str(maxInertiaRatio))


        file_name = 'templates/parametres.xml'
        dom = ElementTree.parse(file_name)
        parametres = dom.findall('Parametres')
        root = dom.getroot()
    
   
        for minarea in root.iter("minArea"):
            minarea.text= minArea
        for maxarea in root.iter("maxArea"):
            maxarea.text= maxArea
        for blobcolor in root.iter("blobColor"):
            blobcolor.text= blobColor
        for mindist in root.iter("minDistBetweenBlobs"):
            mindist.text= minDistBetweenBlobs
        for minthres in root.iter("minThreshold"):
            minthres.text= minThreshold
        for maxthres in root.iter("maxThreshold"):
            maxthres.text= maxThreshold
        for thresstep in root.iter("thresholdStep"):
            thresstep.text= thresholdStep
        for mirep in root.iter("minRepeatability"):
            mirep.text= minRepeatability
        for micirc in root.iter("minCircularity"):
            micirc.text= minCircularity
        for macirc in root.iter("maxCircularity"):
            macirc.text= maxCircularity
        for miconv in root.iter("minConvexity"):
            miconv.text= minConvexity
        for maconv in root.iter("maxConvexity"):
            maconv.text= maxConvexity
        for miinert in root.iter("minInertiaRatio"):
            miinert.text= minInertiaRatio
        for mainert in root.iter("maxInertiaRatio"):
            mainert.text= maxInertiaRatio

        with open(file_name,"wb") as fileupdate:
            dom.write(fileupdate)

        return render_template("parametres.html", param_flagg = param_flagg, minAreaResp = minArea, maxAreaResp = maxArea, blobColorResp = blobColor, minDistResp = minDistBetweenBlobs, minThresholdResp = minThreshold, maxThresholdResp = maxThreshold, thresholdStepResp = thresholdStep, minRepeatabilityResp = minRepeatability , minCircularityResp = minCircularity, maxCircularityResp = maxCircularity, minConvexityResp = minConvexity, maxConvexityResp = maxConvexity, minInertiaRatioResp = minInertiaRatio, maxInertiaRatioResp = maxInertiaRatio)

    else:

        param_flagg = False
        #cli.set('param_flagg', str(param_flagg))
        
        return render_template("parametres.html",param_flagg = param_flagg,  minAreaR = minAreaValue, maxAreaR = maxAreaValue, blobColorR = blobColorValue, minDistR = minDistBetweenBlobsValue, minThresholdR = minThresholdValue, maxThresholdR = maxThresholdValue, thresholdStepR = thresholdStepValue, minRepeatabilityR = minRepeatabilityValue, minCircularityR = minCircularityValue, maxCircularityR = maxCircularityValue, minConvexityR = minConvexityValue, maxConvexityR = maxConvexityValue, minInertiaRatioR = minInertiaRatioValue, maxInertiaRatioR = maxInertiaRatioValue)


@app.route('/20')
def background1():
    
    return Response(background_static(1,0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/21')
def background1_choosed():

    return Response(background_static(2,0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/22')
def background2():
    return Response(background_static2(1,0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/23')
def background2_choosed():
    global mostra2
    mostra2 = True
    return Response(background_static2(2,0), mimetype='multipart/x-mixed-replace; boundary=frame' )


@app.route('/calibrate')
def calibrate():
    global calib_flagg
    global im_xyz, im_rdis, im_amp, frame_inici_amplitut, detector, img_contorns, amplitut_color_li
    calibflag2 = True
    cli.set('calibflag2', str(calibflag2))

    time.sleep(3)

    calib_flagg = str(cli.get('calib_flagg'))
    calib_flagg = calib_flagg[2:-1]
   
    print('2',calib_flagg)
    
    if calib_flagg != '0':
        
        calibflag2 = False
        cli.set('calibflag2', str(calibflag2))
        return render_template("error_calibracio.html", calib_flagg = calib_flagg)
    else:
       
        return render_template("resultats_calibracio.html", calib_flagg = calib_flagg)


@app.route('/world')
def world():

    fig.clear()

    Punt0 = str(cli.get('Punt0'))
    Punt1 = str(cli.get('Punt1'))
    Punt2 = str(cli.get('Punt2'))

    print(Punt0, Punt1, Punt2)
    
    Punt0 = Punt0.replace("b",'')
    Punt0 = Punt0.replace("'",'')
    Punt0 = eval(Punt0)
    Punt1 = Punt1.replace("b",'')
    Punt1 = Punt1.replace("'",'')
    Punt1 = eval(Punt1)
    Punt2 = Punt2.replace("b",'')
    Punt2 = Punt2.replace("'",'')
    Punt2 = eval(Punt2)

    Punt0_wp = str(cli.get('Punt0_wp'))
    Punt1_wp = str(cli.get('Punt1_wp'))
    Punt2_wp = str(cli.get('Punt2_wp'))

    Punt0_wp = Punt0_wp.replace("b",'')
    Punt0_wp = Punt0_wp.replace("'",'')
    Punt0_wp = eval(Punt0_wp)
    Punt1_wp = Punt1_wp.replace("b",'')
    Punt1_wp = Punt1_wp.replace("'",'')
    Punt1_wp = eval(Punt1_wp)
    Punt2_wp = Punt2_wp.replace("b",'')
    Punt2_wp = Punt2_wp.replace("'",'')
    Punt2_wp = eval(Punt2_wp)

    Punt0_usr = str(cli.get('Punt0_usr'))
    Punt1_usr = str(cli.get('Punt1_usr'))
    Punt2_usr = str(cli.get('Punt2_usr'))

    Punt0_usr = Punt0_usr.replace("b",'')
    Punt0_usr = Punt0_usr.replace("'",'')
    Punt0_usr = eval(Punt0_usr)
    Punt1_usr = Punt1_usr.replace("b",'')
    Punt1_usr = Punt1_usr.replace("'",'')
    Punt1_usr = eval(Punt1_usr)
    Punt2_usr = Punt2_usr.replace("b",'')
    Punt2_usr = Punt2_usr.replace("'",'')
    Punt2_usr = eval(Punt2_usr)
  
    
    x = np.array([Punt0[0], Punt1[0], Punt2[0]])
    y = np.array([Punt0[1], Punt1[1], Punt2[1]])
    x_wp = np.array([Punt0_wp[0], Punt1_wp[0], Punt2_wp[0]])
    y_wp = np.array([Punt0_wp[1], Punt1_wp[1], Punt2_wp[1]])
    x_usr = np.array([Punt0_usr[0], Punt1_usr[0], Punt2_usr[0]])
    y_usr = np.array([Punt0_usr[1], Punt1_usr[1], Punt2_usr[1]])

 
    plt.scatter(x,y)
    plt.scatter(x_wp,y_wp)
    plt.scatter(x_usr,y_usr)


    plt.plot(x,y)
    plt.plot(x_wp,y_wp)
    plt.plot(x_usr,y_usr)


    #sns.lineplot(x,y)
    #sns.lineplot(x_wp,y_wp)
    #sns.lineplot(x_usr,y_usr)
    
    
    canvas=FigureCanvas(fig)
    img=io.BytesIO()
    fig.savefig(img)
    img.seek(0)

    calibflag2 = False
    cli.set('calibflag2', str(calibflag2))
   

    return send_file(img, mimetype='img/png')

  
def background_static(v, cam):
    global frame_inici, frame_inici_amplitut,frame_escollit
    if v == 2:
        #agafem el nom de la imatge
        frame_escollit = cli.get('frame_escollit')
        #pasem  a string
        frame_escollit = str(frame_escollit)
        #rectifiquem el string per deixar nomes el titol
        frame_escollit = frame_escollit[2:len(frame_escollit)-1]
  
        image = cv2.imread(frame_escollit, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image_ok = cv2.resize(image, (224,172))
        
        ret, jpeg = cv2.imencode('.jpg',image_ok)
        if jpeg is not None:
            
            return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    
   
    elif v == 3:

        fg = ifm3dpy.FrameGrabber(cam, ifm3dpy.IMG_AMP | ifm3dpy.IMG_RDIS | ifm3dpy.IMG_CART)
        im = ifm3dpy.ImageBuffer()
        fg.wait_for_frame(im)

        frame_inici_amplitut = im.amplitude_image()

        frame_inici = im.distance_image()
        frame_inici.dump('AMIDA4.dat')

        #frame_inici.dump('frameEscollit1.dat')
        #guardamos imagen con nombre
        #im = Image.fromarray(frame_inici_amplitut)
        scipy.misc.imsave('AMIDA4.jpg', frame_inici_amplitut)
        frame_escollit = 'AMIDA4.jpg'
        

        #ret, jpeg = cv2.imencode('.jpg',frame_escollit)
        #if jpeg is not None:
            
        #    return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
    else:
        pass


def background_static2(v, cam):

    global frame_inici2, frame_inici_amplitut2, frame_escollit2, mostra2

    if v == 2:
        
        #agafem el nom de la imatge
        frame_escollit2 = cli.get('frame_escollit2')
        #pasem  a string
        frame_escollit2 = str(frame_escollit2)
        #rectifiquem el string per deixar nomes el titol
        frame_escollit2 = frame_escollit2[2:len(frame_escollit2)-1]
        #print(frame_escollit2)
        image = cv2.imread(frame_escollit2, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image_ok = cv2.resize(image, (224,172))
        
        ret, jpeg = cv2.imencode('.jpg',image_ok)
        if jpeg is not None:
            
            return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    
    
    elif v == 3:
        
        fg = ifm3dpy.FrameGrabber(cam, ifm3dpy.IMG_AMP | ifm3dpy.IMG_RDIS | ifm3dpy.IMG_CART)
        im = ifm3dpy.ImageBuffer()
        fg.wait_for_frame(im)

        frame_inici_amplitut2 = im.amplitude_image()

        frame_inici2 = im.distance_image()
        frame_inici2.dump('AMIDA4_2.dat')

        scipy.misc.imsave('AMIDA4_2.jpg', frame_inici_amplitut2)
        frame_escollit2 = 'AMIDA4_2.jpg'
    else:
        pass



@app.route("/enviarFrame", methods=["POST"])
def enviarFrame():
    global frame_inici, frame_inici_amplitut, frame_flagg, frame_flagg2, frame_escollit
    
    frame_flagg += 1 

    if request.method == 'POST':
        
        frame_escollit = request.form["nom_frame"]
        
        if frame_escollit.endswith('.png') or frame_escollit.endswith('.jpg'):
            frame_escollit_retallat = frame_escollit[:-4]
            frame_dat = frame_escollit_retallat + '.dat'
        
        
        frame_inici = np.load(frame_dat, allow_pickle=True)
        
        background_static(2,0)
        return render_template("frame.html", frame_flagg = frame_flagg, frame_flagg2 = frame_flagg2)
    
    return render_template("frame.html", frame_flagg = frame_flagg, frame_flagg2 = frame_flagg2)
    

@app.route("/enviarFrame2", methods=["POST"])
def enviarFrame2():
    global frame_inici2, frame_inici_amplitut2, frame_flagg2, frame_escollit2
    
    frame_flagg2 += 1 
    
    if request.method == 'POST':
        
        frame_escollit2 = request.form["nom_frame2"]
        if frame_escollit2.endswith('.png') or frame_escollit2.endswith('.jpg'):
            frame_escollit2_retallat = frame_escollit2[:-4]
            frame_dat2 = frame_escollit2_retallat + '.dat'

        frame_inici2 = np.load(frame_dat2, allow_pickle=True)
        
        background_static2(2,0)

        return render_template("frame.html", frame_flagg2 = frame_flagg2)
    
    return render_template("frame.html", frame_flagg2 = frame_flagg2)


@app.route("/treureFrame1")
def treureFrame1():  
    global frame_flagg, frame_flagg2, treureFrame, frame_escollit
    frame_flagg = 1
    frame_flagg2 = 0
    treureFrame = True
    
    cli.set('treureFrame', str(treureFrame))
    frame_escollit = str(cli.get('frame_escollit'))
    frame_escollit = frame_escollit[2:-1]
    
    return render_template("/configuracio.html", frame_flagg = frame_flagg, frame_flagg2 = frame_flagg2)
     

@app.route("/treureFrame2")
def treureFrame2(): 
    global frame_flagg, frame_flagg2, frame_escollit
    
    frame_flagg = 0
    frame_flagg2 = 1
    treureFrame2 = True
    
    cli.set('treureFrame2', str(treureFrame2))
    frame_escollit2 = str(cli.get('frame_escollit2'))
    frame_escollit2 = frame_escollit2[2:-1]
    
    return render_template("/configuracio.html", frame_flagg = frame_flagg, frame_flagg2 = frame_flagg2)
     

@app.route("/guardarFrame")
def guardarFrame():  

    guardarFrame = True
    cli.set('guardarFrame', str(guardarFrame))
    
    #frame_inici.dump('background1.dat')
    #scipy.misc.imsave('background1.png', frame_inici_amplitut)
    #background_static(1,0)
    return render_template("/configuracio.html")
    

@app.route("/guardarFrame2")
def guardarFrame2(): 

    guardarFrame2 = True
    cli.set('guardarFrame2', str(guardarFrame2))
    
    
    #frame_inici2.dump('background2.dat')
    #scipy.misc.imsave('background2.png', frame_inici_amplitut2)
    #background_static2(1,0)
    return render_template("/configuracio.html")


@app.route("/imgCalibracio")
def imgCalibracio(): 
    
    return Response(imatge(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/imatge')
def imatge():

    image = cv2.imread('calibracio.jpg', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    ret, jpeg = cv2.imencode('.jpg',image)
    return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        

if __name__ == '__main__':

    SingleThread.myThread("th").start()   
    #myTimer("tm").start()
    app.run(host='192.168.0.80', debug=True, threaded=True)
    