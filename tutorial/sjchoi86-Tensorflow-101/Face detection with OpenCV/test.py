# -*- coding:utf-8 -*-
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# %matplotlib inline
print ("Packages loaded.")

# 脸部探测器
#下载地址 https://raw.githubusercontent.com/shantnu/Webcam-Face-Detect/master/haarcascade_frontalface_default.xml
# Load Face Detector
cwd = os.getcwd()
clsf_path = cwd + "/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(clsf_path)
print ("face_cascade is %s" % (face_cascade))

# DETECT FACES IN THE IMAGES IN A FOLDER
path = cwd + "/images"
flist = os.listdir(path)
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
for f in flist:
    if os.path.splitext(f)[1].lower() not in valid_exts:
        continue
    fullpath = os.path.join(path, f)
    img_bgr = cv2.imread(fullpath)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img_gray)
    # PLOT
    plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
    ca = plt.gca()
    for face in faces:
        ca.add_patch(Rectangle((face[0], face[1]), face[2], face[3]
                               , fill=None, alpha=1, edgecolor='red'))
    plt.title("Face detection with Viola-Jones")
    plt.show()