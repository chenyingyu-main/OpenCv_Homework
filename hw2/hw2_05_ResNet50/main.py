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
from tensorflow.python.keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.layers import Dense, Flatten
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# import tensorflow_addons as tfa
# Loading VGG19 with imagenet weights

import PIL
import PIL.Image
import random
import os
from torchvision import transforms
import tensorflow as tf
import ssl


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(770, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(370, 40, 250, 250))
        self.image_label.setObjectName("image_label")
        self.predict_label = QtWidgets.QLabel(self.centralwidget)
        self.predict_label.setGeometry(QtCore.QRect(370, 320, 250, 40))
        self.predict_label.setObjectName("predict_label")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 20, 241, 401))
        self.groupBox.setObjectName("groupBox")
        self.load_img_btn = QtWidgets.QPushButton(self.groupBox)
        self.load_img_btn.setGeometry(QtCore.QRect(60, 60, 113, 32))
        self.load_img_btn.setObjectName("load_img_btn")
        self.btn51 = QtWidgets.QPushButton(self.groupBox)
        self.btn51.setGeometry(QtCore.QRect(20, 130, 191, 32))
        self.btn51.setObjectName("btn51")
        self.btn52 = QtWidgets.QPushButton(self.groupBox)
        self.btn52.setGeometry(QtCore.QRect(22, 180, 191, 32))
        self.btn52.setObjectName("btn52")
        self.btn53 = QtWidgets.QPushButton(self.groupBox)
        self.btn53.setGeometry(QtCore.QRect(22, 230, 191, 32))
        self.btn53.setObjectName("btn53")
        self.btn54 = QtWidgets.QPushButton(self.groupBox)
        self.btn54.setGeometry(QtCore.QRect(22, 280, 191, 32))
        self.btn54.setObjectName("btn54")
        self.btn55 = QtWidgets.QPushButton(self.groupBox)
        self.btn55.setGeometry(QtCore.QRect(22, 330, 191, 32))
        self.btn55.setObjectName("btn55")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 770, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # ********************* connect button ***************************
        self.load_img_btn.clicked.connect(self.load_image)
        self.btn51.clicked.connect(self.show_imgs)
        self.btn52.clicked.connect(self.show_distribution)
        self.btn53.clicked.connect(self.model_structure)
        self.btn54.clicked.connect(self.comparison)
        self.btn55.clicked.connect(self.inference)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "hw2_05"))
        self.image_label.setText(_translate("MainWindow", "Image label"))
        self.predict_label.setText(_translate("MainWindow", "Prediction: no prdiction so far :)"))
        self.groupBox.setTitle(_translate("MainWindow", "5. ResNet50"))
        self.load_img_btn.setText(_translate("MainWindow", "Load Image"))
        self.btn51.setText(_translate("MainWindow", "1. Show Images"))
        self.btn52.setText(_translate("MainWindow", "2. Show Distribution"))
        self.btn53.setText(_translate("MainWindow", "3. Show Model Structure"))
        self.btn54.setText(_translate("MainWindow", "4. Show Comparison"))
        self.btn55.setText(_translate("MainWindow", "5. Inference"))

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
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimage))  # show the image in the label

    def show_imgs(self):
        # at home part:
        # path = '/Users/chenyingyu/VSProject/opencvdl_hw2/hw2_05/Dataset_OpenCvDl_Hw2_Q5/training_dataset'
        # train_dataset = tf.keras.preprocessing.image_dataset_from_directory(path, image_size=(224, 224))
        # class_names = train_dataset.class_names

        # ## cats: 5,412, dogs: 10,788

        # # i wonder why i cannot use it on inference :(
        # # get images from each classes
        # for i in range(len(class_names)):
        #     filtered_ds = train_dataset.filter(lambda x, l: l[0]==i)
        #     for image, label in filtered_ds.take(1):
        #         ax = plt.subplot(1, len(class_names), i+1)
        #         plt.imshow(image[0].numpy().astype('uint8'))
        #         plt.title(class_names[label.numpy()[0]])
        #         plt.axis('off')
        # plt.show()

        # demo part:
        FOLDERNAME = 'inference_dataset'
        filename = random.choice(os.listdir(FOLDERNAME + "/Cat"))
        path = '%s/Cat/%s' % (FOLDERNAME , filename)
        print(path)
        img1 = cv2.imread(path)
        img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_CUBIC)

        filename = random.choice(os.listdir(FOLDERNAME + "/Dog"))
        path = '%s/Dog/%s' % (FOLDERNAME , filename)
        print(path)
        img2 = cv2.imread(path)
        img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_CUBIC)

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('cat')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('dog')
        plt.axis('off')
        plt.show()
        
    def show_distribution(self):
        img = cv2.imread('others/class_distribution.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def model_structure(self):
        self.my_model_FL = tf.keras.models.load_model('model/model-resnet50-catdog_FocalLoss.h5', compile=False)
        self.my_model_BCE = tf.keras.models.load_model('model/model-resnet50-catdog_BinaryCross.h5', compile=False)
        self.my_model_BCE.summary()

    def comparison(self):
        img = cv2.imread('others/accuracy_comparison.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def inference(self):
        img = self.image

        net = self.my_model_BCE
        class_list = ['cats', 'dogs']

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        x_batch = np.reshape(img,[1,224,224,3])
        x_batch = np.array(x_batch)
        x_batch = x_batch
        pred = net.predict(x_batch[0:1])
        self.predict_label.setText("Prediction: " +  class_list[np.argmax(pred)] + "\nConfidence: " + str(pred[0][np.argmax(pred)]))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
