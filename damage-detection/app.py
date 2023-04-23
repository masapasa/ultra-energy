
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Conv2D ,Flatten,Dropout,MaxPool2D, BatchNormalization
from keras.utils import np_utils
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory  
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
import keras
from PIL import Image
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix , classification_report
import os
import cv2
from skimage.transform import resize
import streamlit as st

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, COLORS):

    label = f'damage:{confidence}'

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # plt.imshow(img)
    # plt.show()

def detection_inference(image, scale = 1/255, image_size = 416, conf_threshold = 0.1, nms_threshold = 0.4):
  Width = image.shape[1]
  Height = image.shape[0]

  net=cv2.dnn.readNet('yolov4-custom_best.weights','yolov4-custom.cfg')
  COLORS = np.random.uniform(0, 255, size=(1, 3))

  blob = cv2.dnn.blobFromImage(image, scale, (image_size, image_size), (0,0,0), True, crop=False)
  net.setInput(blob)

  outs = net.forward(get_output_layers(net))

  class_ids = []
  confidences = []
  boxes = []

  for out in outs:
    for detection in out:
      scores=detection[5:]
      class_id=np.argmax(scores)
      confidence=scores[class_id]
      if confidence > 0.1:
        center_x = int(detection[0] * Width)
        center_y = int(detection[1] * Height)
        w = int(detection[2] * Width)
        h = int(detection[3] * Height)
        x = center_x - w / 2
        y = center_y - h / 2
        class_ids.append(class_id)
        confidences.append(float(confidence))
        boxes.append([x, y, w, h])

  indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

  for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), COLORS)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.imshow(image)
  plt.show()
  
  return image
  # st.image(image, caption='Object detection output', use_column_width=True)

def _predict(img, model):
  m = keras.models.load_model(model)
  img2 = img.resize((224, 224))

  image_array = np.asarray(img2)
  new_one = image_array.reshape((1, 224, 224, 3))

  y_pred = m(new_one)
  print(y_pred)
  val = np.argmax(y_pred, axis = 1)
  return y_pred, val
  
@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad

def gradcam(img, model):
  m = keras.models.load_model(model)
  LAYER_NAME = 'block5_conv4'
  gb_model = tf.keras.models.Model(
    inputs = [m.inputs],    
    outputs = [m.get_layer(LAYER_NAME).output]
  )
  layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]

  for layer in layer_dict:
    if layer.activation == tf.keras.activations.relu:
      layer.activation = guidedRelu

  img2 = img.resize((224, 224))

  image_array = np.asarray(img2)
  print(image_array.shape)
  new_one = image_array.reshape((1, 224, 224, 3))

  with tf.GradientTape() as tape:
    inputs = tf.cast(new_one, tf.float32)
    tape.watch(inputs)
    outputs = gb_model(inputs)[0]
  grads = tape.gradient(outputs,inputs)[0]

  weights = tf.reduce_mean(grads, axis=(0, 1))
  grad_cam = np.ones(outputs.shape[0: 2], dtype = np.float32)
  for i, w in enumerate(weights):
      grad_cam += w * outputs[:, :, i]

  grad_cam_img = cv2.resize(grad_cam.numpy(), (img.size[0], img.size[1]))
  grad_cam_img = np.maximum(grad_cam_img, 0)
  heatmap = (grad_cam_img - grad_cam_img.min()) / (grad_cam_img.max() - grad_cam_img.min())
  grad_cam_img = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
  output_image = cv2.addWeighted(np.asarray(img).astype('uint8'), 1, grad_cam_img, 0.4, 0)
  
  output_img = Image.fromarray(output_image)
  
  st.image(output_img, caption='Class Activation Visualization', use_column_width=True)

  plt.imshow(output_image)
  plt.axis("off")
  plt.show()

  # guided_back_prop = grads
  # guided_cam = np.maximum(grad_cam, 0)
  # guided_cam = guided_cam / np.max(guided_cam) # scale 0 to 1.0
  # guided_cam = resize(guided_cam, (224,224), preserve_range=True)

  # #pointwise multiplcation of guided backprop and grad CAM 
  # gd_gb = np.dstack((
  #         guided_back_prop[:, :, 0] * guided_cam,
  #         guided_back_prop[:, :, 1] * guided_cam,
  #         guided_back_prop[:, :, 2] * guided_cam,
  #     ))
  # plt.imshow(gd_gb)
  # plt.axis("off")
  # plt.show()

uploaded_file = st.file_uploader(
    "Choose an image of your infrastructure", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    cv_img = np.array(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    # img2 = Image.open('test.jpg')
    st.image(img, caption='Uploaded file of your infrastructure', use_column_width=True)

    # similarity = ssim(img, img2)
    # st.write("")
    # st.write(f'This is {similarity * 100}% histopathological image')

    # if similarity >= 0.85:
    st.write("")
    st.write("Classifying...")

    y_pred, val = _predict(img, 'damage-detections.h5')
    if val == 0:
      st.write(f'The infrastructure has damage.')
      final_img = detection_inference(cv_img)
      final_pil_image = Image.fromarray(final_img)
      gradcam(final_pil_image, 'damage-detections.h5')
    else:
      st.write(f'The infrastructure does not have damage.')
      gradcam(img, 'damage-detections.h5')
      
    

