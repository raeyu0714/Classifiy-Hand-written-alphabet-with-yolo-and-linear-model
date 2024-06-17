from ultralytics import YOLO
import cv2 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd 
# Load the YOLOv8 model
model = YOLO('/content/runs/detect/train5/weights/best.pt')

# Perform inference on an image
results = model('/content/try.png')
# Load the original image
image = "/content/try.png"
img = cv2.imread(image)

# Extract bounding boxes
boxes = results[0].boxes.xyxy.tolist()
a=["N","A","O","K","U","Y","T","H"]
# Iterate through the bounding boxes
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    #print(i)
    #print(x1,x2,y1,y2)
    crop_img = img[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite('crop' + str(i) + '.jpg',crop_img)
    plt.imshow(crop_img) 
    plt.show() 
    pl = "/content/crop"+str(i)+".jpg"
    imagepl = Image.open(pl)
    gray_image = imagepl.convert('L')
    resized_image = gray_image.resize((28, 28))
    image_array = np.array(resized_image)
    flattened_array = image_array.flatten()
    print(flattened_array.tolist())
    d1d=flattened_array.tolist()
    #d1d.append(0)
    d1d=pd.Series(d1d)
    #xs=d1d[1:].values
    xs=d1d.values
    xs=xs.reshape((28, 28))
    plt.imshow(xs, cmap='gray')
    print(np.size(xs))
    plt.show()
    result = chr((svm_model.predict([flattened_array.tolist()])+65)[0])
    print(result)
    #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, a[i], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)

# Display or save the combined result
#cv2.imshow('Combined', img)
cv2.imwrite('combimed.jpg',img)