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
#from helper_functions import warp_and_crop_face, get_reference_facial_points
from embedding_functions.helper_functions import warp_and_crop_face, get_reference_facial_points

# Only function to be called is preprocess_image


# Extracting landmarks from RetinaFace output
# This function outputs x and y coordinates of all landmarks, relative to bounding box values.
# This is because our embedding generation module needs relative coordinates.
def retinaface_landmarking(faces, x_start = 0, y_start = 0):
    points = []
    points.append(faces['face_1']['landmarks']['right_eye'][0] - x_start)
    points.append(faces['face_1']['landmarks']['left_eye'][0] - x_start)
    points.append(faces['face_1']['landmarks']['nose'][0] - x_start)
    points.append(faces['face_1']['landmarks']['mouth_right'][0] - x_start)
    points.append(faces['face_1']['landmarks']['mouth_left'][0] - x_start)
    points.append(faces['face_1']['landmarks']['right_eye'][1] - y_start)
    points.append(faces['face_1']['landmarks']['left_eye'][1] - y_start)
    points.append(faces['face_1']['landmarks']['nose'][1] - y_start)
    points.append(faces['face_1']['landmarks']['mouth_right'][1] - y_start)
    points.append(faces['face_1']['landmarks']['mouth_left'][1] - y_start)
    return points

def preprocess_image_embed(out2):
    print("\n*********************")
    print("Preprocessing for Face Recognition")
    crop_img, faces, x_start, y_start = out2
    points = retinaface_landmarking(faces, x_start, y_start)
    for i in range(len(points)):
        points[i] = int(points[i])
    output_size=(224 , 224)
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    facial5points = np.reshape(points, (2, 5))
    reference_5pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)
    dst_img = warp_and_crop_face(crop_img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
    mid_x, mid_y = int(112), int(112)
    cw2, ch2 = int(150/2), int(150/2)
    crop_img2 = dst_img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    #cv.imwrite('alignedImage.jpg', crop_img2)
    #Recognition
    img = cv.resize(crop_img2, dsize=(112, 112), interpolation=cv.INTER_AREA)
    img.resize((1, 3, 112, 112))
    #img = img - 127.5
    #img = img / 128
    img = img / 255
    print("Passed Image Shape: ", img.shape)
    data = json.dumps({'data': img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')
    return data

