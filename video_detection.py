import numpy as np
import cv2

# Load Neural Network model and capture video
cvNet = cv2.dnn.readNetFromTensorflow('trained_model/frozen_inference_graph.pb', 'annotations/label_map.pbtxt')
cap = cv2.VideoCapture('camera_dataset/terrace2-c1.avi')

# Capture frame to find homography matrix
ret, frame = cap.read()

# Enter filename coordinates
coordinates = 'coordinates/file1.csv'

# Read coordinates from file
pts_src = bird_eye_view.read_coordinates(coordinates)

# Find homography matrix
M = bird_eye_view.find_homography(pts_src, frame)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    rows = frame.shape[0]
    cols = frame.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
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
            list_coordinates.append([left + (right - left) / 2, bottom, 1])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

    # List of actual coordinates of people
    result_coordinates = []
    for box_coordinates in list_coordinates:
        # Use homography matrix to find actual coordinates
        result = M.dot(box_coordinates)
        coordinates = list(map(lambda x: x / result[2], result[:2]))
        result_coordinates.append(coordinates)

    # Display the resulting frame and bird eye view map
    bird_view = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
    for point in result_coordinates:
        cv2.circle(bird_view, tuple(map(int, point)), 40, (0, 0, 255), -1)

    # Show output image
    cv2.imshow('frame', frame)
    cv2.imshow('bird eye view', bird_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
