import queue
import sys
import os
import vosk
import json
import sounddevice as sd
from easydict import EasyDict
import torch
import PIL

from pdf2image import convert_from_path
from PySide2.QtGui import *
from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtCore import *
from PySide2.QtCore import Slot, Signal
from utils.temp_folder import TempFolder
from utils.text import *
from detection.detect import detect
from utils.question_answer import answer_question


class MyEmitter(QObject):
    # setting up custom signal
    doneHandle = Signal(str, int)
    doneRead = Signal()
    doneListen = Signal(str)
    doneDetect = Signal(list)
    doneLabel = Signal(list)


class TextReader(QRunnable):
    def __init__(self, image, lang):
        super(TextReader, self).__init__()
        self.image = image
        self.lang = lang
        self.emitter = MyEmitter()

    def run(self):
        """
        Extract text from image
        Read text
        :return:
        """
        text = extract_text(self.image, self.lang)
        generate_speech(text, lang_code[self.lang])
        self.all_work_done()

    def all_work_done(self):
        """
        Notify StartWindow
        :return:
        """
        self.emitter.doneRead.emit()


class PdfHandler(QRunnable):
    def __init__(self, src, dst):
        super(PdfHandler, self).__init__()
        self.src = src
        self.dst = dst
        self.pageNum = 0
        self.lang = ""
        self.emitter = MyEmitter()

    def run(self):
        """
        Split pdf to a list of images
        Save the images to temp folder
        :return:
        """
        pages = convert_from_path(self.src.lstrip("file:///"))
        self.pageNum = len(pages)

        for idx, page in enumerate(pages):
            page.save(r"{}/{}.jpg".format(self.dst, idx), "JPEG")
        self.initialise_reader()

    def initialise_reader(self):
        """
        Detect pdf language
        :return:
        """
        if self.pageNum == 1:
            image = self.dst + '/0.jpg'
            l, oc, sc = detect_language(image, ["chi_sim", "eng", "fra", "jpn"])
        else:
            for idx in range(1, self.pageNum):
                image = self.dst + '/{}.jpg'.format(idx)
                print(image)
                l, oc, sc = detect_language(image, ["chi_sim", "eng", "fra", "jpn"])

                if sc.count(max(sc)) != 1:
                    continue
                else:
                    self.lang = l
                    break

        self.all_work_done(l)

    def all_work_done(self, lang):
        """
        Emit language detected and pdf page count to StartWindow
        :param lang: language detected
        :return:
        """
        self.emitter.doneHandle.emit(lang, self.pageNum)


class Listener(QRunnable):
    def __init__(self):
        super(Listener, self).__init__()
        self.q = queue.Queue()
        self.samplerate = 48000
        self.blocksize = 8000
        self.model = vosk.Model("../vosk-en")
        self.emitter = MyEmitter()

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def run(self):
        """
        Recognise User voice input, save to string, finished with "thank you"
        :return:
        """
        with sd.RawInputStream(samplerate=self.samplerate, blocksize=self.blocksize, device=1, dtype='int16',
                               channels=1, callback=self.callback):

            rec = vosk.KaldiRecognizer(self.model, self.samplerate)
            speech = ""

            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.FinalResult())
                    speech = speech + res['text'] + ", "
                    print(res['text'])
                    if res['text'] == "thank you":
                        break

            self.all_work_done(speech)

    def all_work_done(self, s):
        """
        Emit user voice input to StartWindow
        :param s:
        :return:
        """
        self.emitter.doneListen.emit(s)


class ImageObjectDetector(QRunnable):
    def __init__(self, src, dst):
        super(ImageObjectDetector, self).__init__()
        # YOLO V5 Configuration
        opt = EasyDict()
        opt.source = src
        opt.dst = dst
        opt.weights = '../detection/best.pt'
        opt.augment = True
        opt.device = '0'  # GPU
        opt.img_size = 640
        opt.line_thickness = 6
        opt.conf_thres = 0.25
        opt.iou_thres = 0.45
        opt.agnostic_nms = True
        self.opt = opt
        self.emitter = MyEmitter()

    def run(self):
        """ Run detection """
        with torch.no_grad():
            labels = detect(opt=self.opt)
        # print(type(labels))
        self.all_work_done(labels)

    def all_work_done(self, obj):
        # print(obj)
        self.emitter.doneDetect.emit(obj)


class Labeler(QRunnable):
    def __init__(self, filename, labelList, x_s, y_s, w_s, h_s):
        super(Labeler, self).__init__()
        self.labelList = labelList
        self.filename = filename
        self.x_s = x_s
        self.y_s = y_s
        self.w_s = w_s
        self.h_s = h_s
        self.emitter = MyEmitter()

        self.q = queue.Queue()
        self.samplerate = 48000
        self.blocksize = 8000
        self.model = vosk.Model("../vosk-en")

    def run(self):
        """
        After user release the mouse button, ask for the object class
        :return:
        """
        generate_speech("Could you please name the object?")
        self.getcha()

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def getcha(self):
        """
        Load Vosk Model to
        :return:
        """
        with sd.RawInputStream(samplerate=self.samplerate, blocksize=self.blocksize, device=1, dtype='int16',
                               channels=1, callback=self.callback):

            rec = vosk.KaldiRecognizer(self.model, self.samplerate)
            speech = ""

            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.FinalResult())
                    speech = speech + res['text'] + ", "
                    print(res['text'])
                    if res['text'] != "":
                        break
        self.nameTheObject(speech)

    def nameTheObject(self, Context):
        Question = "What is the object?"
        obj = answer_question(Question, Context)
        self.write_to_txt(obj)

    def write_to_txt(self, obj_class):
        """
        Write the annotation file
        Give the same name as corresponding image
        Emit the updated LabelList
        :return:
        """
        if obj_class not in self.labelList:
            self.labelList.append(obj_class)

        label_index = self.labelList.index(obj_class)

        with open(self.filename, 'a') as f:
            f.write(f'{label_index} {self.x_s} {self.y_s} {self.w_s} {self.h_s}\n')
        self.emitter.doneLabel.emit(self.labelList)


class StartWindow(QObject):
    def __init__(self):
        QObject.__init__(self)

        # Create a temporal folder to store pdf file
        self.tfObj = TempFolder()
        self.pool = QThreadPool()
        self.tfPath = ""
        self.lang = ""
        self.speech = ""
        self.ObjDetect = []

        self.languages = ["chi_sim", "eng", "fra", "jpn"]
        self.currentPage = 0

        self.enableOperate = True

        # Label anchor point
        self.labelXInit = 0
        self.labelYInit = 0
        self.labelXEnd = 0
        self.labelYEnd = 0

        # Window Size
        self.contentPageHeight = 0
        self.contentPageWidth = 0

        self.labelList = []

    # Signals To Send Data
    signalFile = Signal(str, str, int)
    signalFinished = Signal()
    signalLabelling = Signal(bool)

    @Slot(str)
    def get_file(self, file_path):
        """
        Get user input pdf file
        Save the pdf file to temp folder
        Launch the pdf handler thread
        :param file_path: original file path
        :return:
        """
        # Image or pdf file only
        # if file_path.endswith('.pdf'):
        if file_path.endswith((".pdf", ".png")):
            # Change slash to back slash, otherwise cannot reload image
            self.tfPath = self.tfObj.get_directory().replace(os.sep, "/")
            print(self.tfPath)
            worker = PdfHandler(file_path, self.tfPath)
            worker.emitter.doneHandle.connect(self.on_handler_done)
            self.pool.start(worker)

    @Slot(int)
    def get_pageIndex(self, i):
        """
        Slot function - Update the page index of image displayed on the screen
        :param i:
        :return:
        """
        self.currentPage = i

    @Slot()
    def read(self):
        """
        Slot function - Launch the text reader thread to read text
        :return:
        """
        if self.enableOperate:
            self.enableOperate = False
            image = self.tfPath + f"/{self.currentPage}.jpg"
            worker = TextReader(image, self.lang)
            worker.emitter.doneRead.connect(self.on_reader_done)
            self.pool.start(worker)

    @Slot()
    def listen(self):
        """
        Slot function - Launch the listener thread to recognise user voice input
        :return:
        """
        if self.enableOperate:
            self.enableOperate = False
            worker = Listener()
            worker.emitter.doneListen.connect(self.on_listen_done)
            self.pool.start(worker)

    @Slot()
    def detect(self):
        """
        Slot function - Launch the detector worker thread to detect image objects
        :return:
        """
        if self.enableOperate:
            self.enableOperate = False
            worker = ImageObjectDetector(self.tfPath + f"/{self.currentPage}.jpg", self.tfPath)
            worker.emitter.doneDetect.connect(self.on_detect_done)
            self.pool.start(worker)

    @Slot()
    def label(self):
        """
        Slot function
        Firstly, visit the temp folder to get the actual image fize
        Secondly, use anchor point to reverse the image scale-down mathematically to calculate yolo format annotation
        Lastly, launch the work thread with annotations
        :return:
        """
        filename = self.tfPath + f"/{self.currentPage}.txt"
        print(filename)

        # Retrieve image size
        image = PIL.Image.open(self.tfPath + f"/{self.currentPage}.jpg")
        imgWidth, imgHeight = image.size
        print(f'Image size: {imgWidth} {imgHeight}')

        # Assume image height < width && imgSize > windowSize
        # ContentPageHeight is the adapted image height
        ratio = imgHeight / self.contentPageHeight
        print(f'Ratio: {ratio}')

        # blankSpace only exists in x-axis as image is adapted by y-axis
        blankSpace = (self.contentPageWidth - imgWidth / ratio) / 2
        print(f'Blank space in x-axis: {blankSpace}')

        offset_x = self.labelXInit
        offset_y = self.labelYInit

        x_centroid = (self.labelXEnd - self.labelXInit)/2 + offset_x - blankSpace
        y_centroid = (self.labelYEnd - self.labelYInit)/2 + offset_y
        print(f'x_centroid: {x_centroid}, y_centroid: {y_centroid}')

        x_scale = round(x_centroid / (imgWidth/ratio), 2)
        y_scale = round(y_centroid / self.contentPageHeight, 2)
        # Width scale = label rectangle width / adapted image width
        width_scale = round((self.labelXEnd - self.labelXInit)/(imgWidth/ratio), 2)
        height_scale = round((self.labelYEnd - self.labelYInit)/self.contentPageHeight, 2)
        print(x_scale, y_scale, width_scale, height_scale)

        worker = Labeler(filename, self.labelList, x_scale, y_scale, width_scale, height_scale)
        worker.emitter.doneLabel.connect(self.on_label_done)
        self.pool.start(worker)

    @Slot(str, int)
    def on_handler_done(self, lang, c):
        """
        Internal slot function
        Save variable in the backend
        Emit signals to the UI
        :param lang: text language emitted from worker thread
        :param c:
        :return:
        """
        self.lang = lang
        p = os.path.join("file:///" + self.tfPath + "/")
        self.signalFile.emit(p, lang, c)

    @Slot()
    def on_reader_done(self):
        self.enableOperate = True

    @Slot(str)
    def on_listen_done(self, s):
        """
        Internal slot function
        Save the user voice input for question answering
        :param s: recognised speech
        :return:
        """
        self.speech = s
        print(self.speech)
        self.enableOperate = True

    @Slot(list)
    def on_detect_done(self, labels):
        """
        Internal slot function
        :param labels: detected object classes
        :return:
        """
        self.ObjDetect = labels
        self.signalFinished.emit()
        self.ObjDetect.reverse()

        s = "Object detected from image: "
        for c in self.ObjDetect:
            s += f"{c} "
            if c not in self.labelList:
                self.labelList.append(c)
        generate_speech(s)
        self.enableOperate = True

    @Slot(bool)
    def enable_labelling(self, b):
        """ Label one object at a time """
        if not b:
            self.signalLabelling.emit(True)

    @Slot(int, int, int, int)
    def get_pos(self, x1, y1, x2, y2):
        """
        anchor point position
        :param x1: Start point x axis
        :param y1: Start point y axis
        :param x2: End point x axis
        :param y2: End point y axis
        :return:
        """
        self.labelXInit = x1
        self.labelYInit = y1
        self.labelXEnd = x2
        self.labelYEnd = y2
        print(x1, y1, x2, y2)

    @Slot(int, int)
    def get_contentPage_size(self, hei, wid):
        """
        Get UI Content page width and height
        :param hei: UI height
        :param wid: UI width
        :return:
        """
        self.contentPageHeight = hei
        self.contentPageWidth = wid

    @Slot(list)
    def on_label_done(self, ll):
        """
        Save object classes to the backend
        :param ll:
        :return:
        """
        self.labelList = ll


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Get Context
    start = StartWindow()
    engine.rootContext().setContextProperty("backend", start)

    engine.load(os.path.join(os.path.dirname(__file__), "QML/main.qml"))

    # Response for closing the application after clicking the close button
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())
