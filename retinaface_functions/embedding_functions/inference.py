import numpy as np
import cv2 as cv
import os
import onnx
import onnxruntime
from onnx import numpy_helper
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
from retinaface.commons import preprocess, postprocess
import json
from os import path

# Model dorectory
model_dir ="./arcface.onnx"

# Only function to be called is infer_image

def infer_image_embed(data):
    print("\n*********************")
    print("Inference for Face Recognition")
    session = onnxruntime.InferenceSession(model_dir, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    #print(input_name)
    #print(output_name)
    result = session.run([output_name], {input_name: data})
    print("Output Shape: ", result[0].shape)
    return result[0]

