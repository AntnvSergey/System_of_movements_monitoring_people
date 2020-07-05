import cv2
import numpy as np
import bird_eye_view

# Load Neural Network model
cvNet = cv2.dnn.readNetFromTensorflow('trained_model/frozen_inference_graph.pb', 'annotations/label_map.pbtxt')

# Enter image path
image_path = 'image_for_paper/image2.jpg'

# Read image
image = cv2.imread(image_path)
rows = image.shape[0]
cols = image.shape[1]

# Enter filename coordinates
coordinates = 'coordinates/file1.csv'

# Read coordinates from file
pts_src = bird_eye_view.read_coordinates(coordinates)

# Find homography matrix
M = bird_eye_view.find_homography(pts_src, image)

# Detect people on image with Neural Network model
cvNet.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

# List of coordinates middle bounding box people
list_coordinates = []
for detection in cvOut[0, 0, :, :]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        list_coordinates.append([left + (right - left)/2, bottom, 1])
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

# List of actual coordinates of people
result_coordinates = []
for box_coordinates in list_coordinates:
    # Use homography matrix to find actual coordinates
    result = M.dot(box_coordinates)
    coordinates = list(map(lambda x: x / result[2], result[:2]))
    result_coordinates.append(coordinates)

# Draw bird eye view image
bird_view = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
for point in result_coordinates:
    cv2.circle(bird_view, tuple(map(int, point)), 40, (0, 0, 255), -1)


# Show output image
cv2.imshow('People detection', image)
cv2.imshow('Bird eye view', bird_view)
cv2.waitKey(0)
