import cv2
import numpy as np


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16)
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


def resized(image):
    scale_percent = 20  # Процент от изначального размера
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(im_src, dim, interpolation=cv2.INTER_AREA)
    return resized


if __name__ == '__main__':
    # Читаем исходное изображение
    im_src = cv2.imread('./images/img1.jpg')
    # Уменьшаем изображение (тк. исходное слишком большого размера) и выделяем 4 точки
    im_src_resized = resized(im_src)
    pts_src = get_four_points(im_src_resized)


    # Пока что руками задаю конечное положение выделеных точек
    size = (300, 400)
    pts_dst = np.float32(
        [
            [200, 200],
            [size[0] - 1 + 200, 200],
            [size[0] - 1 + 200, size[1] - 1 + 200],
            [200, size[1] - 1 + 200]
        ]
    )

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # Деформируем исходное изображение
    im_out = cv2.warpPerspective(im_src, M, (im_src.shape[1], im_src.shape[0]))
    cv2.imwrite("im_out.jpg", im_out)

    # Вывод результата на экран
    im_out_resized = resized(im_out)
    cv2.imshow("Warped Source Image", im_out)

    cv2.waitKey(0)
