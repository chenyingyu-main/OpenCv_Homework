from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(190, 20, 231, 331))
        self.groupBox.setObjectName("groupBox")
        self.btn31 = QtWidgets.QPushButton(self.groupBox)
        self.btn31.setGeometry(QtCore.QRect(20, 50, 181, 32))
        self.btn31.setObjectName("btn31")
        self.btn32 = QtWidgets.QPushButton(self.groupBox)
        self.btn32.setGeometry(QtCore.QRect(20, 120, 181, 32))
        self.btn32.setObjectName("btn32")
        self.btn33 = QtWidgets.QPushButton(self.groupBox)
        self.btn33.setGeometry(QtCore.QRect(20, 190, 181, 32))
        self.btn33.setObjectName("btn33")
        self.btn34 = QtWidgets.QPushButton(self.groupBox)
        self.btn34.setGeometry(QtCore.QRect(20, 260, 181, 32))
        self.btn34.setObjectName("btn34")
        self.load_btn = QtWidgets.QPushButton(self.centralwidget)
        self.load_btn.setGeometry(QtCore.QRect(20, 160, 141, 32))
        self.load_btn.setObjectName("load_btn")
        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(30, 190, 131, 21))
        self.image_label.setObjectName("image_label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(450, 20, 231, 331))
        self.groupBox_2.setObjectName("groupBox_2")
        self.btn41 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn41.setGeometry(QtCore.QRect(20, 50, 181, 32))
        self.btn41.setObjectName("btn41")
        self.btn42 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn42.setGeometry(QtCore.QRect(20, 120, 181, 32))
        self.btn42.setObjectName("btn42")
        self.btn43 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn43.setGeometry(QtCore.QRect(20, 190, 181, 32))
        self.btn43.setObjectName("btn43")
        self.btn44 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn44.setGeometry(QtCore.QRect(20, 260, 181, 32))
        self.btn44.setObjectName("btn44")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 700, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # ********************* connect button ***************************
        self.load_btn.clicked.connect(self.load_image)
        self.btn31.clicked.connect(self.gaussian_blur)
        self.btn32.clicked.connect(self.sobel_x)
        self.btn33.clicked.connect(self.sobel_y)
        self.btn34.clicked.connect(self.magnitude)
        self.btn41.clicked.connect(self.resize)
        self.btn42.clicked.connect(self.translation_overlay)
        self.btn43.clicked.connect(self.rotate_scale)
        self.btn44.clicked.connect(self.shearing)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "hw1-2 MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "3. Edge Detection"))
        self.btn31.setText(_translate("MainWindow", "3.1 Gaussian Blur"))
        self.btn32.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.btn33.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.btn34.setText(_translate("MainWindow", "3.4 Magnitude"))
        self.load_btn.setText(_translate("MainWindow", "Load Image"))
        self.image_label.setText(_translate("MainWindow", "image loaded"))
        self.groupBox_2.setTitle(_translate("MainWindow", "4. Transformation"))
        self.btn41.setText(_translate("MainWindow", "4.1 Resize"))
        self.btn42.setText(_translate("MainWindow", "4.2 Translate"))
        self.btn43.setText(_translate("MainWindow", "4.3 Rotate, Scaling"))
        self.btn44.setText(_translate("MainWindow", "4.4 Shearing"))

    # ************************* Write my functions here *************************
    def load_image(self):
        # load image
        self.filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
        self.image = cv2.imread(self.filename)
        self.image_label.setText(self.filename.split('/')[-1])

    # **************** HW1-3 *************** # 
    def gaussian_blur(self):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # turn to grayscale first
        kernel = np.array([[0.045, 0.122, 0.045], [0.122, 0.332, 0.122], [0.045, 0.122, 0.045]])
        output = convolve2D(image, kernel)
        plt.imshow(output, cmap='gray')
        plt.title('Gaussian Blur')
        plt.show()

    def sobel_x(self):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.so_x = convolve2D(image, kernel)

        output = np.sqrt(np.square(self.so_x)) # or it will be gray scale
        plt.imshow(output, cmap='gray')
        plt.title('Sobel X')
        plt.show()

    def sobel_y(self):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.so_y = convolve2D(image, kernel)

        output = np.sqrt(np.square(self.so_y)) # or it will be gray scale
        plt.imshow(output, cmap='gray')
        plt.title('Sobel Y')
        plt.show()

    def magnitude(self):
        gradient_magnitude = np.sqrt(np.square(self.so_x) + np.square(self.so_y))
        gradient_magnitude *= 255.0 / gradient_magnitude.max()
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title('Magnitude')
        plt.show()

     # **************** HW1-4 *************** # 
    def resize(self):
        image = self.image
        self.HEIGHT, self.WIDTH, dimen = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2-BGR, plt-RGB
        
        M = np.float32([[215/self.HEIGHT , 0, 0],[0, 215/ self.WIDTH, 0]])
        self.resize_img = cv2.warpAffine(image, M, (self.HEIGHT, self.WIDTH))
        plt.imshow(self.resize_img)
        plt.show()

    def translation_overlay(self):
        image = self.resize_img
        M = np.float32([[1 , 0, 215],[0, 1, 215]])
        transla = cv2.warpAffine(image, M, (self.HEIGHT, self.WIDTH))

        self.overlay = cv2.addWeighted(image, 1, transla, 1, 0.0)
        plt.imshow(self.overlay)
        plt.show()

    def rotate_scale(self):
        image = self.overlay
        center_x, center_y = self.HEIGHT//2, self.WIDTH//2

        M1 = np.float32([[0.5 , 0, center_x//2], [0, 0.5, center_y//2]])
        M2 = cv2.getRotationMatrix2D((center_x, center_y), 45, 1)

        self.rot_scal = cv2.warpAffine(image, M1, (self.HEIGHT, self.WIDTH))
        self.rot_scal = cv2.warpAffine(self.rot_scal, M2, (self.HEIGHT, self.WIDTH))
        plt.imshow(self.rot_scal)
        plt.show()

    def shearing(self):
        image = self.rot_scal
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[100,50],[100,250]])

        M = cv2.getAffineTransform(pts1, pts2)
        shear = cv2.warpAffine(image, M, (self.HEIGHT, self.WIDTH))
        plt.imshow(shear)
        plt.show()
        

# 2D convolution OUTSIDE THE CLASS
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
