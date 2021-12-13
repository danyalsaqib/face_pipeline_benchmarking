import numpy as np
import cv2 as cv
import os
import onnx
import onnxruntime
from onnx import numpy_helper
import json
import matplotlib.pyplot as plt
import tensorflow as tf

from os import path

# ENTER MODEL PATH HERE
model_path = 'retinanet.onnx'

# The only function to be called from here is 'infer_image'

def infer_image_landmark(out0, threshold=0.9, allow_upscaling = True):
    print("\n*********************")
    print("Inference for Face Detection and Landmarking")
    #print("\n***********************\n")
    #print("Starting inference on input image using ONNX Model")
    #print("Shape of input image: ", image.shape)
    #print("Shape of im_tensor: ", im_tensor.shape)
    retina_session = onnxruntime.InferenceSession(model_path, None)
    retina_input_name = retina_session.get_inputs()[0].name
    retina_output_name = []
    for i in range(len(retina_session.get_outputs())):
        retina_output_name.append(retina_session.get_outputs()[i].name)
    im_array, _ = out0
    im_tensor, _, _ = im_array
    #print(retina_input_name)
    #print(retina_output_name)
    retina_data = json.dumps({'data': im_tensor.tolist()})
    retina_data = np.array(json.loads(retina_data)['data']).astype('float32')
    net_out = retina_session.run(retina_output_name, {retina_input_name: retina_data})
    print("Retina Network Result Length: ", len(net_out))
    
    return net_out, out0, threshold
