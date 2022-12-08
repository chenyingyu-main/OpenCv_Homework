from operator import mod
from turtle import width
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2
from cv2 import transform
from matplotlib import image, transforms
import numpy as np
import matplotlib.pyplot as plt

# Keras, dataset, and VGG19 imports
import keras
from keras.datasets import cifar10
from keras.applications import VGG19
# Loading VGG19 with imagenet weights

import PIL.Image as Image
from torchvision import transforms
import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 550)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 261, 381))
        self.groupBox.setObjectName("groupBox")
        self.load_btn = QtWidgets.QPushButton(self.groupBox)
        self.load_btn.setGeometry(QtCore.QRect(30, 30, 191, 32))
        self.load_btn.setObjectName("load_btn")
        self.btn1 = QtWidgets.QPushButton(self.groupBox)
        self.btn1.setGeometry(QtCore.QRect(30, 110, 200, 32))
        self.btn1.setObjectName("btn1")
        self.btn2 = QtWidgets.QPushButton(self.groupBox)
        self.btn2.setGeometry(QtCore.QRect(30, 160, 200, 32))
        self.btn2.setObjectName("btn2")
        self.btn3 = QtWidgets.QPushButton(self.groupBox)
        self.btn3.setGeometry(QtCore.QRect(30, 210, 200, 32))
        self.btn3.setObjectName("btn3")
        self.btn4 = QtWidgets.QPushButton(self.groupBox)
        self.btn4.setGeometry(QtCore.QRect(30, 260, 200, 32))
        self.btn4.setObjectName("btn4")
        self.btn5 = QtWidgets.QPushButton(self.groupBox)
        self.btn5.setGeometry(QtCore.QRect(30, 310, 200, 32))
        self.btn5.setObjectName("btn5")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(350, 10, 350, 500))
        self.photo.setObjectName("photo")
        self.text1 = QtWidgets.QLabel(self.centralwidget)
        self.text1.setGeometry(QtCore.QRect(350, 30, 200, 32))
        self.text1.setObjectName("text1")
        self.text2 = QtWidgets.QLabel(self.centralwidget)
        self.text2.setGeometry(QtCore.QRect(350, 50, 200, 32))
        self.text2.setObjectName("text2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 750, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # ********************* connect button ***************************
        self.btn1.clicked.connect(self.load_data_AND_show_train_image)
        self.btn2.clicked.connect(self.show_model_structure)
        self.load_btn.clicked.connect(self.load_image)
        self.btn3.clicked.connect(self.show_augmentation)
        self.btn4.clicked.connect(self.show_accuracy_loss)
        self.btn5.clicked.connect(self.inference)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "hw1-5 MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "5. Resnet101 Test"))
        self.load_btn.setText(_translate("MainWindow", "Load Image"))
        self.btn1.setText(_translate("MainWindow", "1. Show Train Images"))
        self.btn2.setText(_translate("MainWindow", "2. Show Model Structure"))
        self.btn3.setText(_translate("MainWindow", "3. Show Data Augmentation"))
        self.btn4.setText(_translate("MainWindow", "4. Show Accuracy and Loss"))
        self.btn5.setText(_translate("MainWindow", "5. Inference"))
        self.photo.setText(_translate("MainWindow", "Image Label"))
        self.text1.setText(_translate("MainWindow", "confidence = 0"))
        self.text2.setText(_translate("MainWindow", "prdiction label = :)"))
        self.text1.setVisible(False)
        self.text2.setVisible(False)


    def load_data_AND_show_train_image(self):
        (self.x_train, self.y_train) , (self.x_val, self.y_val) = cifar10.load_data()

        self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        width_grid = length_grid = 3
        fig, axes = plt.subplots(length_grid, width_grid, figsize = (6,6))
        axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

        n_train = len(self.x_train) # get the length of the train dataset

        for i in np.arange(0, width_grid * length_grid): # create evenly spaces variables 

            # Select a random number
            index = np.random.randint(0, n_train)
            # read and display an image with the selected index    
            axes[i].imshow(self.x_train[index,1:])
            label_index = int(self.y_train[index])
            axes[i].set_title(self.labels[label_index], fontsize = 8)
            axes[i].axis('off')

        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def show_model_structure(self):
        self.my_model = tf.keras.models.load_model('cifar_ten_vgg19.h5')
        self.my_model.summary()

    def load_image(self):
        # load image
        self.filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
        self.image = cv2.imread(self.filename)
        self.image = cv2.resize(self.image, (224, 224), interpolation=cv2.INTER_CUBIC)
        height, width, channel = self.image.shape
        pages = 3 * width
        # show image
        # convert the form of "OpenCV (numpy)" to "QImage", and insert the image's height, width, pages
        qimage = QtGui.QImage(self.image, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label

    def show_augmentation(self):
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # we will use plt to show later BGR -> RGB
        transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(360), transforms.ToTensor()])
        rot_img = transform(img)
        
        transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomResizedCrop(size=(300,350)), transforms.ToTensor()]) 
        size_img = transform(img)

        transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.9), transforms.ToTensor()])
        flip_img = transform(img)

        npimg_rot = rot_img.numpy()
        npimg_size = size_img.numpy()
        npimg_flip = flip_img.numpy()
        fig = plt.figure(figsize=(8, 2)) 
        fig.add_subplot(1, 3, 1) 
        # in pytorch img[Channels, H, W], in plt img[H, W, Channels]
        plt.imshow(np.transpose(npimg_rot, (1, 2, 0)))
        plt.title('random rotation')
        plt.axis('off')
        fig.add_subplot(1, 3, 2)
        plt.imshow(np.transpose(npimg_size, (1, 2, 0)))
        plt.title('random resize crops')
        plt.axis('off')
        fig.add_subplot(1, 3, 3)
        plt.imshow(np.transpose(npimg_flip, (1, 2, 0)))
        plt.title('random horizontal flip')
        plt.axis('off')
        plt.show()

    def show_accuracy_loss(self):
        self.text1.setVisible(False)
        self.text2.setVisible(False)
        img = cv2.imread('accu_loss.png')
        # resize my image
        img = cv2.resize(img, (int(img.shape[1]*0.95), int(img.shape[0]*0.95)), interpolation=cv2.INTER_AREA)
        height, width, channel = img.shape
        pages = 3 * width
        # show image
        # convert the form of "OpenCV (numpy)" to "QImage", and insert the image's height, width, pages
        qimage = QtGui.QImage(img, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label

    def inference(self):
        self.text1.setVisible(True)
        self.text2.setVisible(True)
        img = self.image
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        x_test = np.reshape(img,[1,224,224,3])
        x_test = np.array(x_test)

        y_pred = self.my_model.predict(x_test)
        confi = y_pred[0][np.argmax(y_pred)]
        self.text1.setText("confidence: " + str(confi))
        
        y_pred = np.argmax(y_pred, axis=1)
        label_index = int(y_pred)
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if confi < 0.7 :
            self.text2.setText('i dont know what that is :(')
        else:
            self.text2.setText('prediction label: ' + labels[label_index])

        # show image (if we forget to load the image again)
        height, width, channel = self.image.shape
        pages = 3 * width
        # show image
        # convert the form of "OpenCV (numpy)" to "QImage", and insert the image's height, width, pages
        qimage = QtGui.QImage(self.image, width, height, pages, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.photo.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label


        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
