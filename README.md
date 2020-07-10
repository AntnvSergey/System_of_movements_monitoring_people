# System of movements monitoring people #
This project implements automatic detecting people in a photo or video using a trained model of the ssdlite_mobilenet_v2_coco neural network and finding real
coordinates of people using a perspective transformation.
The project also implements a facial recognition system using the face_recognition library.

## Creating your own dataset ##
To train the neural network model, we created our own dataset, which includes an annotation only about finding people in the images.
The new dataset was obtained by merging two datasets such as PASCAL VOC and COCO by the [GluonCV](https://gluon-cv.mxnet.io/contents.html) and TensorFlow libraries.

You can create your own dataset using `MergedDataset.ipynb` and also collect statistics on the created dataset using `Hist_script.ipynb`

Since the model was trained using the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection),
training requires files in the ".record" format.
You can convert the created dataset to this format using a script `pascalvoc_to_tfrecords.py`.
However, this script works with TensorFlow (v1).
After creating the ".record" files, change the version of the TensorFlow library from v1 to v2.

## People detection ##
In this work a neural network model ssdlite_mobilenet_v2_coco was trained to find people in real time.
This model was taken from [TensorFlow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
repository and trained with TensorFlow Object Detection API on its own dataset.
Scripts were used for training and further use of the model:

- `model_main.py`
- `train_ssdlite_mobilenet_v2_coco.sh`
- `export_inference_graph.py`
- `export_ssdlite_mobilenet_v2_coco.sh`

You can use them to train your model in the future.

This project contains 2 folders for the presented model:

- pre-trained-model: Pre-trained model on the COCO dataset.
- trained_model: Trained model on the created dataset for detecting only people.

The `image_detection.py` and `video_detection.py` scripts implement the use of a trained neural network to find a person in a photo or video, respectively,
along with finding the real coordinates of a person using a perspective transformation.
Therefore, before using these scripts, find 4 points for perspective transformation using the script `bird_eye_view.py` (see "Finding the real coordinates of a person").

For the most convenient work with the neural network model, I ran this model using the [OpenCV library](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)
with `tf_text_graph_ssd.py` and `tf_text_graph_common.py` scripts.

## Finding the real coordinates of a person ##
We want to find the real coordinates of a person and to do this, we need to convert the original image into a bird's-eye view.
To compute the perspective transform, M, given the source and destination points we use `cv2.getPerspectiveTransform(src,dst)` from OpenCV library.
We need to manually identify four (4) source coordinates points for the perspective transform.
For this task, 4 points are located in the corners of the room where people are located.
If it is not possible to select points in the corners of the room, select the rectangle that would represent the plane on which people are.
Select points from the upper left corner of the rectangle.

Perspective transformation of the source image is implemented in the script `bird_eye_view.py`.
To avoid highlighting points on the image every time, their coordinates are written in a csv file, the location of which you can specify yourself.
The "coodinates" folder is created for such files.

For a better understanding of how perspective transformation works, you can read my paper: [image transformation to bird eye view](https://digiratory.ru/1674),
but this paper is only in Russian language.

## Face recognition ##
This work also provides facial recognition of people in the image. The [face_recognition](https://github.com/ageitgey/face_recognition) library was used
to solve this problem.
All scripts for facial recognition are located in the folder "face_recognition".

This system was created using a lesson by [pyimagesearch](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/).
This identification system assumes that there is a created dataset, which is also located in the folder "face_recognition".
You can create your own dataset similar to the one shown here. For the best accuracy, select multiple images for a single person.

## Installation ##
To install this project, use the `requirements.txt` file.
However in my version of the project the TensorFlow and face_recognition libraries are installed without gpu support.
To install [TensorFlow Object Detection API](https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api)
and [face_recognition](https://github.com/ageitgey/face_recognition) libraries with gpu-enabled libraries, see the installation sections.
