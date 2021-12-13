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

# From this, only call the function 'postprocess_image'

# Python program for implementation of MergeSort
def mergeSort(arr):
    if len(arr) > 1:
        # Finding the mid of the array
        mid = len(arr)//2

        # Dividing the array elements
        L = arr[:mid]
        # into 2 halves
        R = arr[mid:]

        # Sorting the first half
        mergeSort(L)
        # Sorting the second half
        mergeSort(R)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if np.sum(L[i].shape) < np.sum(R[j].shape):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def get_output(net_out, im_array, threshold):
    im_tensor, im_info, im_scale = im_array
    nms_threshold = 0.4; decay4=0.5
    mergeSort(net_out)

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        'stride32': np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
        'stride16': np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
        'stride8': np.array([[-8., -8., 23., 23.], [ 0.,  0., 15., 15.]], dtype=np.float32)
    }

    _num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}

    #---------------------------

    proposals_list = []
    scores_list = []
    landmarks_list = []
    
    sym_idx = 0

    for _idx, s in enumerate(_feat_stride_fpn):
        #print("sym_idx: ", sym_idx)
        _key = 'stride%s'%s
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors['stride%s'%s]:]
        
        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
        
        A = _num_anchors['stride%s'%s]
        K = height * width
        anchors_fpn = _anchors_fpn['stride%s'%s]
        anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        
        scores = scores.reshape((-1, 1))
        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_deltas = bbox_deltas
        bbox_pred_len = bbox_deltas.shape[3]//A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)

        proposals = postprocess.clip_boxes(proposals, im_info[:2])

        if s==4 and decay4<1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel>=threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3]//A
        #print("Landmarks Deltas Shape: ", landmark_deltas.shape)
        #print("landmark_pred_len//5: ", landmark_pred_len//5)
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)
    if proposals.shape[0]==0:
        landmarks = np.zeros( (0,5,2) )
        return np.zeros( (0,5) ), landmarks
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

    #nms = cpu_nms_wrapper(nms_threshold)
    #keep = nms(pre_det)
    keep = postprocess.cpu_nms(pre_det, nms_threshold)

    det = np.hstack( (pre_det, proposals[:,4:]) )
    det = det[keep, :]
    landmarks = landmarks[keep]

    resp = {}
    for idx, face in enumerate(det):

        label = 'face_'+str(idx+1)
        resp[label] = {}
        resp[label]["score"] = face[4]

        resp[label]["facial_area"] = list(face[0:4].astype(int))

        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])

    return resp
    

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


# Functions for finding angles

def find_roll(pts):
    #print("Roll: ", pts[6] - pts[5])
    return pts[6] - pts[5]

def find_yaw(pts):
    le2n = pts[2] - pts[0]
    re2n = pts[1] - pts[2]
    #print("Yaw: ", le2n - re2n)
    return le2n - re2n

def find_pitch(pts):
    eye_y = (pts[5] + pts[6]) / 2
    mou_y = (pts[8] + pts[9]) / 2
    e2n = eye_y - pts[7]
    n2m = pts[7] - mou_y
    #print("Pitch: ", e2n/n2m)
    return e2n/n2m

class Error(Exception):
    """Base class for other exceptions"""
    pass


class PathNotFound(Error):
    pass


class MultipleFaceDetected(Error):
    pass
class InvalidPose(Error):
    pass

class NoFaceDetected(Error):
    pass


class SmallFaceDetected(Error):
    pass

def postprocess_image_landmark(out1):
    print("\n*********************")
    print("Postprocessing for Face Detection and Landmarking")
    net_out, out0, threshold = out1
    threshold = 0.5
    im_array, image = out0
    faces = get_output(net_out, im_array, threshold)
    check_multi_detection = len(faces)
    if check_multi_detection > 1:
        return -1
    confidence = faces['face_1']['score']
    try:
        if confidence > threshold:
            if check_multi_detection > 1:
                raise MultipleFaceDetected
            elif check_multi_detection == 1:
                boundingBoxes = faces['face_1']['facial_area']

                # Get start and end values of Bounding Boxes
                print("Bounding Boxes: ", boundingBoxes)
                x_start, y_start, x_end, y_end = boundingBoxes
                #print("x_start: ", x_start)
                #print("x_end: ", x_end)
                #print("y_start: ", y_start)
                #print("y_end: ", y_end)
                crop_img = image[y_start:y_end, x_start:x_end]
                #print("crop_img shape: ", crop_img.shape)
                #cv.imwrite('croppedImage.jpg', crop_img)
                origin_h1, origin_w1 = crop_img.shape[:2]
                #print("About to print origin_h1 and origin_w1")
                #print(origin_h1, origin_w1)
                print("Single Face Detected")
                points = retinaface_landmarking(faces, x_start, y_start)
                for i in range(len(points)):
                    points[i] = int(points[i])
                #print("Modified Landmarks: ", points)

                if origin_h1 > 80 and origin_w1 > 80:
                    #print("\n*************************\n")
                    print("\nRoll, Yaw, and Pitch: (", find_roll(points), ", ", find_yaw(points), ", ", find_pitch(points), ")")
                    roll_bound = int(0.05 * origin_h1)
                    yaw_bound = int(0.17 * origin_w1)
                    print("Roll Bound: ", roll_bound)
                    print("Yaw Bound: ", yaw_bound)
                    if find_roll(points) > (-1 * roll_bound) and  find_roll(points) < roll_bound and find_yaw(points) > (-1 * yaw_bound) and  find_yaw(points) < yaw_bound and find_pitch(points) < 2.5 and find_pitch(points) > 0.5:
                        print("valid face")   
                        print("\nEnd of Face Detection and Landmarking Pipeline\n")                 
                        return crop_img, faces, x_start, y_start
                    else:
                        raise InvalidPose
                else:
                    raise SmallFaceDetected
                            
            else:
                raise NoFaceDetected
                
                

                
    except PathNotFound:
        return -1
        print("Check input path")
    except MultipleFaceDetected:
        return -1
        print("Multiple face detected")
    except NoFaceDetected:
        return -1
        print("No face Detected")
    except SmallFaceDetected:
        return -1
        print("Detected face is smaller than requiured for embedding")
    except InvalidPose:
        return -1
        print("Pose is not valid")
