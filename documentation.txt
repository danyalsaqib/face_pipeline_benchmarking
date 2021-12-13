# face_landmarking_pipeline
The link to ONNX models used can be found at: https://drive.google.com/drive/folders/1N2WusFEbFUaaj82NxfjADcRT_VWAG_Uo?usp=sharing

## RetinaFace Functions
1. pre_processing.py:

Preprocesses image, and outputs an image tensor. Only call 'preprocess_image' function

2. inference.py:

Performs inference on image tensor, and outputs RetinaFace output. Only call 'infer_image' function

2. post_processing.py:

Postprocesses RetinaFace output, and outputs cropped image and 'relative landmarks'. Only call 'postprocess_image' function

## Embedding Functions
Should be given output of postprocessing of RetinaFace
1. pre_processing.py:

Preprocesses previous results, and outputs aligned images data. Only call 'preprocess_image_embed' function

2. inference.py:

Performs inference on aligned images, and outputs Raw Embeddings. Only call 'infer_image' function

2. post_processing_embed.py:

Postprocesses Raw Embeddings, and outputs Final Embeddings. Only call 'postprocess_image_embed' function
