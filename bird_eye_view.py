import cv2
import numpy as np
import csv


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x, y), 10, (0, 0, 255), -1)
        cv2.imshow("Image", data['im'])
        if len(data['points']) < 4:
            data['points'].append([x, y])


def get_four_points(im):
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    points = np.float32(data['points'])

    return points


def get_coordinates(im_src, file_name):
    pts_src = get_four_points(im_src)
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(pts_src)
    return pts_src


def read_coordinates(file_name):
    pts_src = []
    with open(file_name, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            pts_src.append(row)
    return np.float32(pts_src)


def find_homography(pts_src, image):
    pts_dst = np.float32(
        [
            [0, 0],
            [image.shape[1], 0],
            [image.shape[1], image.shape[0]],
            [0, image.shape[0]]
        ]
    )

    # Calculate the homography
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    return M


if __name__ == '__main__':

    im_src = cv2.imread('book1.jpg')

    # Path to coordinates csv file
    coordinate_path = 'coordinates/file1.csv'

    # Get 4 points and write coordinates of points to file
    pts_src = get_coordinates(im_src, coordinate_path)

    # Find homography`
    M = find_homography(pts_src, im_src)

    # Warp source image to destination
    im_out = cv2.warpPerspective(im_src, M, (im_src.shape[1], im_src.shape[0]))
    cv2.imshow("Warped image", im_out)
    cv2.waitKey(0)

