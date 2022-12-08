from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2
from cv2 import merge
from cv2 import medianBlur
import numpy
import matplotlib.pyplot as plt

from subWindow import Ui_SubWindow

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(680, 380)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_img1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_img1.setGeometry(QtCore.QRect(30, 80, 113, 32))
        self.btn_img1.setObjectName("btn_img1")
        self.btn_img2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_img2.setGeometry(QtCore.QRect(30, 200, 113, 32))
        self.btn_img2.setObjectName("btn_img2")
        self.groupBox1 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox1.setGeometry(QtCore.QRect(200, 30, 211, 291))
        self.groupBox1.setObjectName("groupBox1")
        self.btn11 = QtWidgets.QPushButton(self.groupBox1)
        self.btn11.setGeometry(QtCore.QRect(10, 50, 181, 32))
        self.btn11.setObjectName("btn11")
        self.btn12 = QtWidgets.QPushButton(self.groupBox1)
        self.btn12.setGeometry(QtCore.QRect(10, 110, 181, 32))
        self.btn12.setObjectName("btn12")
        self.btn13 = QtWidgets.QPushButton(self.groupBox1)
        self.btn13.setGeometry(QtCore.QRect(10, 170, 181, 32))
        self.btn13.setObjectName("btn13")
        self.btn14 = QtWidgets.QPushButton(self.groupBox1)
        self.btn14.setGeometry(QtCore.QRect(12, 230, 181, 32))
        self.btn14.setObjectName("btn14")
        self.groupBox2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox2.setGeometry(QtCore.QRect(430, 30, 201, 291))
        self.groupBox2.setObjectName("groupBox2")
        self.btn21 = QtWidgets.QPushButton(self.groupBox2)
        self.btn21.setGeometry(QtCore.QRect(22, 70, 151, 32))
        self.btn21.setObjectName("btn21")
        self.btn22 = QtWidgets.QPushButton(self.groupBox2)
        self.btn22.setGeometry(QtCore.QRect(22, 140, 151, 32))
        self.btn22.setObjectName("btn22")
        self.btn23 = QtWidgets.QPushButton(self.groupBox2)
        self.btn23.setGeometry(QtCore.QRect(22, 210, 151, 32))
        self.btn23.setObjectName("btn23")
        self.l_img1 = QtWidgets.QLabel(self.centralwidget)
        self.l_img1.setGeometry(QtCore.QRect(40, 115, 121, 31))
        self.l_img1.setObjectName("l_img1")
        self.l_img2 = QtWidgets.QLabel(self.centralwidget)
        self.l_img2.setGeometry(QtCore.QRect(40, 235, 121, 31))
        self.l_img2.setObjectName("l_img2")

        ## test image label here
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(680, 40, 251, 281))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        ## test image label here

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 983, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # ********************* connect button ***************************
        self.btn_img1.clicked.connect(self.btn1_clicked)
        self.btn_img2.clicked.connect(self.btn2_clicked)
        self.btn11.clicked.connect(self.split_channel)
        self.btn12.clicked.connect(self.colour_transformation)
        self.btn13.clicked.connect(self.colour_detection)
        self.btn14.clicked.connect(self.blending_img)
        self.btn21.clicked.connect(self.guassianBlur)
        self.btn22.clicked.connect(self.bilateralFilter)
        self.btn23.clicked.connect(self.medianFilter)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2022 Opencvdl Hw1"))
        self.btn_img1.setText(_translate("MainWindow", "Load Image 1"))
        self.btn_img2.setText(_translate("MainWindow", "Load Image 2"))
        self.groupBox1.setTitle(_translate("MainWindow", "1. Image Processing"))
        self.btn11.setText(_translate("MainWindow", "1.1 Color Saperation"))
        self.btn12.setText(_translate("MainWindow", "1.2 Color Transformation"))
        self.btn13.setText(_translate("MainWindow", "1.3 Color Detection"))
        self.btn14.setText(_translate("MainWindow", "1.4 Blending"))
        self.groupBox2.setTitle(_translate("MainWindow", "2. Image Smoothing"))
        self.btn21.setText(_translate("MainWindow", "2.1 Guassian Blur"))
        self.btn22.setText(_translate("MainWindow", "2.2 Bilateral Filter"))
        self.btn23.setText(_translate("MainWindow", "2.3 Median Filter"))
        self.l_img1.setText(_translate("MainWindow", "No Image Loaded"))
        self.l_img2.setText(_translate("MainWindow", "No Image Loaded"))
        self.photo.setText(_translate("MainWindow", "TextLabel"))

    # ************************* Write my functions here *************************
    def btn1_clicked(self):
        # load image1
        self.loadImage_subWin()
        self.l_img1.setText(self.filename.split('/')[-1])

    def btn2_clicked(self):
        self.loadImage_subWin2()
        self.l_img2.setText(self.filename2.split('/')[-1])

    def split_channel(self):
        img = self.image
        b, g, r = cv2.split(img)
        # make all zeros channel
        zeros = numpy.zeros(img.shape[:2], dtype = numpy.uint8)
        merge_r = cv2.merge([r, zeros, zeros])
        merge_g = cv2.merge([zeros, g, zeros])
        merge_b = cv2.merge([zeros, zeros, b])

        # create figure
        rows = 1
        columns = 3
        fig = plt.figure(figsize=(8, 2)) # create figure (inches)

        fig.add_subplot(rows, columns, 1) # Adds a subplot at the 1st position
        plt.imshow(merge_b)
        plt.axis('off')
        plt.title('B channel')
        fig.add_subplot(rows, columns, 2) 
        plt.imshow(merge_g)
        plt.axis('off')
        plt.title('G channel')
        fig.add_subplot(rows, columns, 3) # Adds a subplot at the 1st position
        plt.imshow(merge_r)
        plt.axis('off')
        plt.title('R channel')
        plt.show()

    def colour_transformation(self):
        img = self.image
        b, g, r = cv2.split(img)
        img = cv2.merge([b, g, r])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = r/3 + g/3 + b/3 # (r+g+b)/3 will overflow

        rows = 1
        columns = 2
        fig = plt.figure(figsize=(8, 2)) # create figure (inches)
        fig.add_subplot(rows, columns, 1) # Adds a subplot at the 1st position
        plt.imshow(gray, cmap='gray', vmin = 0, vmax = 255) # show the correct gray scale
        plt.axis('off')
        plt.title('OpenCV function')
        fig.add_subplot(rows, columns, 2) 
        plt.imshow(gray2, cmap='gray')
        plt.axis('off')
        plt.title('Average weighted')
        plt.show()

    def colour_detection(self):
        # transform from rgb to hsv
        img = self.image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # create the mask
        lower_green = numpy.array([40, 50, 20])
        upper_green = numpy.array([80, 255, 255])
        lower_white = numpy.array([0, 0, 200])
        upper_white = numpy.array([180, 20, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        # bit wise AND the image with mask
        img_green = cv2.bitwise_and(img, img, mask=mask_green)
        img_white = cv2.bitwise_and(img, img, mask=mask_white)
        # show image
        fig = plt.figure(figsize=(8, 2)) 
        fig.add_subplot(1, 2, 1) 
        plt.imshow(img_green)
        plt.title('Green')
        plt.axis('off')
        fig.add_subplot(1, 2, 2) 
        plt.imshow(img_white)
        plt.title('White')
        plt.axis('off')
        plt.show()

    #******************** how to show image in a sub window ****************************
    # 1-3 blending images
    def blending_img(self):
        # open the image window 
        self.setUp_SubWindow()
        # set the bar
        bar = self.ui.blend_slider
        bar.setMaximum(255)
        bar.setMinimum(0)
        bar.valueChanged['int'].connect(self.get_blend_value)
        # set the image (img1 on top)
        image1 = self.image
        height, width, channel = image1.shape
        pages = 3 * width
        qimage = QtGui.QImage(image1, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label

    def get_blend_value(self, value):
        self.ui.bl_label.setText('Blend: ' + str(value))
        img = self.image
        height, width, channel = img.shape
        pages = 3 * width
        img2 = self.image2

        # blend two images together
        alpha = value / 256
        beta = (1.0 - alpha)
        images = cv2.addWeighted(img2, alpha, img, beta, 0.0)

        # show image
        # convert the form of "OpenCV (numpy)" to "QImage", and insert the image's height, width, pages
        qimage = QtGui.QImage(images, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label
  

    # 2-1: Guassian Blur
    def guassianBlur(self):
        self.set_blur_window()
        self.ui.blend_slider.valueChanged['int'].connect(self.GBlur_change)

    def GBlur_change(self, value):
        self.ui.bl_label.setText('magnitude: ' + str(value))
        img = self.image
        height, width, channel = img.shape
        pages = 3 * width
        k = 2 * value +1

        blur = cv2.GaussianBlur(img, (k, k), 0)
        qimage = QtGui.QImage(blur, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label

    # 2-2: Bilateral Filter
    def bilateralFilter(self):
        self.set_blur_window()
        self.ui.blend_slider.valueChanged['int'].connect(self.BiFilter_change)

    def BiFilter_change(self, value):
        self.ui.bl_label.setText('magnitude: ' + str(value))
        img = self.image
        height, width, channel = img.shape
        pages = 3 * width
        k = 2 * value +1

        blur = cv2.bilateralFilter(img, k, 90, 90) # sigmaColor = 90, sigmaSpace = 90
        qimage = QtGui.QImage(blur, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label

    # 2-3: Median Filter
    def medianFilter(self):
        self.set_blur_window()
        self.ui.blend_slider.valueChanged['int'].connect(self.MeFilter_change)

    def MeFilter_change(self, value):
        self.ui.bl_label.setText('magnitude: ' + str(value))
        img = self.image
        height, width, channel = img.shape
        pages = 3 * width
        k = 2 * value +1

        blur = medianBlur(img, k)
        qimage = QtGui.QImage(blur, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label

    def set_blur_window(self):
        self.setUp_SubWindow()
        img = self.image
        height, width, channel = img.shape
        pages = 3 * width
        qimage = QtGui.QImage(img, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label
        bar = self.ui.blend_slider
        bar.setMaximum(10)
        bar.setMinimum(0)
        self.ui.bl_label.setText('magnitude: 0')
        
    

    # ************************* USE SUB WINDOW TO SHOW... *************************

    def setUp_SubWindow(self):
        # open the second window: image window
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_SubWindow()
        self.ui.setupUi(self.window)
        self.window.show()
        # there will be a bar and a photo label

    def loadImage_subWin(self):
        # load my image chosen to subwindow
        self.filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
        self.image = cv2.imread(self.filename)
        

    def loadImage_subWin2(self):
        # load my image chosen to subwindow
        self.filename2, filetype2 = QtWidgets.QFileDialog.getOpenFileName()
        self.image2 = cv2.imread(self.filename2)
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
