from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2
import os
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1030, 450)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 190, 360))
        self.groupBox.setObjectName("groupBox")
        self.load_folder = QtWidgets.QPushButton(self.groupBox)
        self.load_folder.setGeometry(QtCore.QRect(20, 80, 150, 32))
        self.load_folder.setObjectName("load_folder")
        self.load_image_l = QtWidgets.QPushButton(self.groupBox)
        self.load_image_l.setGeometry(QtCore.QRect(20, 180, 150, 32))
        self.load_image_l.setObjectName("load_image_l")
        self.load_image_r = QtWidgets.QPushButton(self.groupBox)
        self.load_image_r.setGeometry(QtCore.QRect(20, 260, 150, 32))
        self.load_image_r.setObjectName("load_image_r")
        self.folder_label = QtWidgets.QLabel(self.groupBox)
        self.folder_label.setGeometry(QtCore.QRect(30, 110, 131, 16))
        self.folder_label.setObjectName("folder_label")
        self.L_image_label = QtWidgets.QLabel(self.groupBox)
        self.L_image_label.setGeometry(QtCore.QRect(30, 210, 131, 16))
        self.L_image_label.setObjectName("L_image_label")
        self.R_image_label = QtWidgets.QLabel(self.groupBox)
        self.R_image_label.setGeometry(QtCore.QRect(30, 290, 131, 16))
        self.R_image_label.setObjectName("R_image_label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(220, 20, 201, 360))
        self.groupBox_2.setObjectName("groupBox_2")
        self.btn11 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn11.setGeometry(QtCore.QRect(20, 90, 150, 32))
        self.btn11.setObjectName("btn11")
        self.btn12 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn12.setGeometry(QtCore.QRect(20, 170, 150, 32))
        self.btn12.setObjectName("btn12")
        self.ring1_label = QtWidgets.QLabel(self.groupBox_2)
        self.ring1_label.setGeometry(QtCore.QRect(10, 220, 181, 41))
        self.ring1_label.setObjectName("ring1_label")
        self.ring2_label = QtWidgets.QLabel(self.groupBox_2)
        self.ring2_label.setGeometry(QtCore.QRect(10, 260, 181, 41))
        self.ring2_label.setTextFormat(QtCore.Qt.AutoText)
        self.ring2_label.setScaledContents(False)
        self.ring2_label.setObjectName("ring2_label")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(430, 20, 190, 360))
        self.groupBox_3.setObjectName("groupBox_3")
        self.btn21 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn21.setGeometry(QtCore.QRect(20, 40, 150, 32))
        self.btn21.setObjectName("btn21")
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_6.setGeometry(QtCore.QRect(10, 130, 171, 111))
        self.groupBox_6.setObjectName("groupBox_6")
        self.btn23 = QtWidgets.QPushButton(self.groupBox_6)
        self.btn23.setGeometry(QtCore.QRect(10, 70, 150, 32))
        self.btn23.setObjectName("btn23")
        self.spinBox_23 = QtWidgets.QSpinBox(self.groupBox_6)
        self.spinBox_23.setGeometry(QtCore.QRect(30, 30, 101, 24))
        self.spinBox_23.setMinimum(1)
        self.spinBox_23.setMaximum(20)
        self.spinBox_23.setObjectName("spinBox_23")
        self.btn22 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn22.setGeometry(QtCore.QRect(20, 90, 150, 32))
        self.btn22.setObjectName("btn22")
        self.btn24 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn24.setGeometry(QtCore.QRect(20, 260, 150, 32))
        self.btn24.setObjectName("btn24")
        self.btn25 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn25.setGeometry(QtCore.QRect(20, 310, 150, 32))
        self.btn25.setObjectName("btn25")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(630, 20, 190, 360))
        self.groupBox_4.setObjectName("groupBox_4")
        self.textEdit_3 = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit_3.setGeometry(QtCore.QRect(20, 60, 151, 41))
        self.textEdit_3.setObjectName("textEdit_3")
        self.label_6_char = QtWidgets.QLabel(self.groupBox_4)
        self.label_6_char.setGeometry(QtCore.QRect(20, 110, 151, 20))
        self.label_6_char.setObjectName("label_6_char")
        self.btn31 = QtWidgets.QPushButton(self.groupBox_4)
        self.btn31.setGeometry(QtCore.QRect(-1, 170, 191, 32))
        self.btn31.setObjectName("btn31")
        self.btn32 = QtWidgets.QPushButton(self.groupBox_4)
        self.btn32.setGeometry(QtCore.QRect(0, 240, 191, 32))
        self.btn32.setObjectName("btn32")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(830, 20, 190, 360))
        self.groupBox_5.setObjectName("groupBox_5")
        self.btn41 = QtWidgets.QPushButton(self.groupBox_5)
        self.btn41.setGeometry(QtCore.QRect(0, 130, 191, 32))
        self.btn41.setObjectName("btn41")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1030, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # ********************* connect button ***************************
        self.load_folder.clicked.connect(self.load_folder_image)
        self.load_image_l.clicked.connect(self.load_image_L)
        self.load_image_r.clicked.connect(self.load_image_R)
        self.btn11.clicked.connect(self.draw_contour)
        self.btn12.clicked.connect(self.count_rings)
        self.btn21.clicked.connect(self.find_corners)
        self.btn22.clicked.connect(self.intrinsic)
        self.btn23.clicked.connect(self.extrinsic)
        self.btn24.clicked.connect(self.distortion)
        self.btn25.clicked.connect(self.show_udist_result)
        self.btn31.clicked.connect(self.show_on_board)
        self.btn32.clicked.connect(self.show_vertically)
        self.btn41.clicked.connect(self.stereo_disparity_map)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2022 opencv hw2"))
        self.groupBox.setTitle(_translate("MainWindow", "Load Data"))
        self.load_folder.setText(_translate("MainWindow", "Load Folder"))
        self.load_image_l.setText(_translate("MainWindow", "Load Image L"))
        self.load_image_r.setText(_translate("MainWindow", "Load Image R"))
        self.folder_label.setText(_translate("MainWindow", "No folder loaded"))
        self.L_image_label.setText(_translate("MainWindow", "No image loaded"))
        self.R_image_label.setText(_translate("MainWindow", "No image loaded"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1. Find Contour"))
        self.btn11.setText(_translate("MainWindow", "1.1 Draw Contour"))
        self.btn12.setText(_translate("MainWindow", "1.2 Count Rings"))
        self.ring1_label.setText(_translate("MainWindow", "There are __ rings in img1.jpg"))
        self.ring2_label.setText(_translate("MainWindow", "There are __ rings in img2.jpg"))
        self.groupBox_3.setTitle(_translate("MainWindow", "2. Calibration"))
        self.btn21.setText(_translate("MainWindow", "2.1 Find Corners"))
        self.groupBox_6.setTitle(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.btn23.setText(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.btn22.setText(_translate("MainWindow", "2.2 Find Intrinsic"))
        self.btn24.setText(_translate("MainWindow", "2.4 Find Distortion"))
        self.btn25.setText(_translate("MainWindow", "2.5 Show Result"))
        self.groupBox_4.setTitle(_translate("MainWindow", "3. Augmented Reality"))
        self.label_6_char.setText(_translate("MainWindow", "Word less than 6 char"))
        self.btn31.setText(_translate("MainWindow", "3.1 Show Words on Board"))
        self.btn32.setText(_translate("MainWindow", "3.2 Show Words Vertically"))
        self.groupBox_5.setTitle(_translate("MainWindow", "4. Stereo Disparity Map"))
        self.btn41.setText(_translate("MainWindow", "4.1 Stereo Disparity Map"))

    # ************************* Write my functions here *************************
    def load_folder_image(self):
        # load image
        foldername = QFileDialog.getExistingDirectory(None, "Select Directory")

        # load all images in the folder
        self.images = []
        for filename in os.listdir(foldername):
            img = cv2.imread(os.path.join(foldername, filename))
            if img is not None:
                self.images.append(img)
            
        # print out folder size (images), and adjust the max of spinBox
        print('images size  = ' + str(len(self.images)))
        self.spinBox_23.setMaximum(len(self.images))
        # set label
        self.folder_label.setText(foldername.split('/')[-1])
        print(foldername)

    def load_image_L(self):
        # load image
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
        self.imageL = cv2.imread(filename)
        self.L_image_label.setText(filename.split('/')[-1])

    def load_image_R(self):
        # load image
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
        self.imageR = cv2.imread(filename)
        self.R_image_label.setText(filename.split('/')[-1])

    #************* Q1 ***************#
    def draw_contour(self):
        if len(self.images) == 2 :
            img1 = self.images[1]
            img2 = self.images[0]

        # resize to 1/2
        img1 = cv2.resize(img1, (int(img1.shape[1] / 2), int(img1.shape[0] / 2)), interpolation= cv2.INTER_AREA)
        img2 = cv2.resize(img2, (int(img2.shape[1] / 2), int(img2.shape[0] / 2)), interpolation= cv2.INTER_AREA)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # remove noise (Guassian Blur)
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        edge1 = cv2.Canny(gray1, 50, 150) # low threshhold : high threshhold = 1:3
        edge2 = cv2.Canny(gray2, 50, 150)
        contours1, hierarchy1 = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours2, hierarchy2 = cv2.findContours(edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours(img1, contours1, -1, (0, 0, 255), thickness=1)
        cv2.drawContours(img2, contours2, -1, (0, 0, 255), thickness=1)

        self.ring1 = int(len(contours1) / 4)
        self.ring2 = int(len(contours2) / 4)

        # show image
        fig = plt.figure(figsize=(8, 3)) 
        fig.add_subplot(1, 2, 1) 
        plt.imshow(img1)
        plt.title('img1.jpg')
        plt.axis('off')
        fig.add_subplot(1, 2, 2) 
        plt.imshow(img2)
        plt.title('img2.jpg')
        plt.axis('off')
        plt.show()

    def count_rings(self):
        self.ring1_label.setText("There are " + str(self.ring1) + " rings in img1.jpg")
        self.ring2_label.setText("There are " + str(self.ring2) + " rings in img2.jpg")

    #************* Q2 ***************#
    def find_corners(self):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        h = 8
        w = 11

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((w*h, 3), np.float32)
        objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        for org_img in self.images:
            img = org_img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (w, w), (-1, -1), criteria)
                self.objpoints.append(objp)
                self.imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (w, h), corners, ret)

                # show image and puase for 0.5 second
                plt.imshow(img)
                plt.axis('off')
                plt.show()
                plt.pause(0.5)
                plt.clf()
        
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
 
    def intrinsic(self):
        print("\nIntrinsic : ")
        print(self.mtx)
        print('-------------')

    def extrinsic(self):
        num = self.spinBox_23.value()
        np.set_printoptions(formatter={'float_kind':'{:f}'.format})

        R = cv2.Rodrigues(self.rvecs[num-1])
        ext = np.hstack((R[0], self.tvecs[num-1]))
        print("\nExtrinsic for image " + str(num) + " : ")
        print(ext)
        print('-------------')

    def distortion(self):
        print("\nDistortion : ")
        print(self.dist)
        print('-------------')

    def show_udist_result(self):

        fig = plt.figure(figsize=(8, 3)) 
        
        for org_img in self.images:
            img = org_img.copy()
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))

            # undistort
            dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            # show image and puase for 0.5 second
            fig.add_subplot(1, 2, 1) 
            plt.imshow(img)
            plt.title('Distorted Image')
            plt.axis('off')
            fig.add_subplot(1, 2, 2) 
            plt.imshow(dst)
            plt.title('Undistorted Image')
            plt.axis('off')

            plt.show()
            plt.pause(0.5)
            plt.clf()
            
    #************* Q3 ***************#
    def show_on_board(self):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        h = 8
        w = 11

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((w*h, 3), np.float32)
        objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.


        string = self.textEdit_3.toPlainText()
        fs = cv2.FileStorage('Q3_library/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        points_zero = [[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]

        for i in range(int(len(self.images))):
            img = self.images[i].copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (w, w), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                for j in range(int(len(string))):
                    ch = fs.getNode(string[j]).mat()
                    axis = np.float32(ch).reshape(-1, 3)
                    for point in axis:
                        point += points_zero[j]

                    imgpts, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], mtx, dist)
                    for k in range(int(len(imgpts)/2)):
                        img = cv2.line( img, tuple(imgpts[k*2].ravel()), tuple(imgpts[k*2+1].ravel()), (255, 0, 0), 5)
                        # img, start, end, color, width of line

                # show image and puase for 0.5 second
                plt.imshow(img)
                plt.axis('off')
                plt.show()
                plt.pause(0.5)
                plt.clf()
    
    def show_vertically(self):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        h = 8
        w = 11

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((w*h, 3), np.float32)
        objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.


        string = self.textEdit_3.toPlainText()
        fs = cv2.FileStorage('Q3_library/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        points_zero = [[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]]

        for i in range(int(len(self.images))):
            img = self.images[i].copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (w, w), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                for j in range(int(len(string))):
                    ch = fs.getNode(string[j]).mat()
                    axis = np.float32(ch).reshape(-1, 3)
                    for point in axis:
                        point += points_zero[j]

                    imgpts, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], mtx, dist)
                    for k in range(int(len(imgpts)/2)):
                        img = cv2.line( img, tuple(imgpts[k*2].ravel()), tuple(imgpts[k*2+1].ravel()), (255, 0, 0), 5)
                        # img, start, end, color, width of line

                # show image and puase for 0.5 second
                plt.imshow(img)
                plt.axis('off')
                plt.show()
                plt.pause(0.5)
                plt.clf()

    #************* Q4 ***************#

    def stereo_disparity_map(self):
        imgl = self.imageL
        imgr = self.imageR
        grayl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
        grayr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = stereo.compute(grayl, grayr)
        self.disp_norm = cv2.normalize(
            self.disparity,
            self.disparity,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        fig0 = plt.figure()
        fig0.add_subplot(1, 1, 1)
        plt.imshow(self.disp_norm, cmap='gray')
        plt.axis('off')

        # 4.2 Checking the Disparity Value #
        Cx = 279.184

        def draw_match(event):

            x = int(event.xdata)
            y = int(event.ydata)
            if x > self.disparity.shape[1]:
                return

            dist = int((self.disparity[y][x])/16) 
            if dist <= 0:
                return

            print(x, y, dist)
            point = (x - dist, y)
            imgR = self.imageR.copy()
            imgR = cv2.circle(imgR, point, 10, (0, 255, 0), -1)

            fig2 = plt.figure()
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(800,100,640, 545)
            fig2.add_subplot(1, 1, 1)
            plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        fig1 = plt.figure() 
        cid = fig1.canvas.mpl_connect('button_press_event', draw_match)
        fig1.add_subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
