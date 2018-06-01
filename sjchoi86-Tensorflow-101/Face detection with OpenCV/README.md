# Face detection with OpenCV

参考：

- https://github.com/sjchoi86/Tensorflow-101/blob/master/notebooks/basic_opencv2.ipynb
- https://blog.csdn.net/wc781708249/article/details/78587604

---

# 单个图片
```python
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

# LOAD IMAGE WITH FACES
# THIS IS BGR
imgpath = cwd + "/images/celebs2.jpg"
# imgpath = cwd + "/../../img_dataset/celebs/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0001.jpg"
img_bgr = cv2.imread(imgpath)

# CONVERT TO RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# DETECT FACE
faces = face_cascade.detectMultiScale(img_gray)
print ("%d faces deteced. " % (len(faces)))

# PLOT DETECTED FACES
# PLOT
plt.figure(0)
plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
ca = plt.gca()
for face in faces:
    ca.add_patch(Rectangle((face[0], face[1]), face[2], face[3]
                           , fill=None, alpha=1, edgecolor='red'))
plt.title("Face detection with Viola-Jones")
# plt.draw()
plt.show()
```

![这里写图片描述](https://img-blog.csdn.net/20180601200546918?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

---

# 按文件夹

```python
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
```