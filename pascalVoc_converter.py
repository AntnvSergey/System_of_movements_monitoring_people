import os
import numpy as np
import cv2
from jinja2 import Environment, FileSystemLoader
from jinja2 import PackageLoader
import xml.etree.ElementTree as ET


def get_pascal(img_path, type_coordinates_list, path_to_save):
    """
    converts image and points coordinates to pascal XML

    :param img_path: full path to image
    :param type_coordinates_list: numpy array of all coordinates - [xmin,ymin,xmax,ymax....]
    :param path_to_save: path where the xml will be saved
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    writer = Writer(img_path, width=img.shape[1], height=img.shape[0])

    for xmin, ymin, xmax, ymax in zip(type_coordinates_list[::4], type_coordinates_list[1::4],
                                      type_coordinates_list[2::4], type_coordinates_list[3::4]):
        writer.addObject(img_path, xmin, ymin, xmax, ymax)

    writer.save(path_to_save)


def get_numpy(pascal_path):
    """
    converts pascal xml to numpy array
    [xmin,ymin,xmax,ymax]

    :param pascal_path:
    :return:
    """

    ls = []
    root = ET.parse(pascal_path).getroot()
    for child in root.iter('bndbox'):
        for data in child:
            ls.append(float(data.text))
    return np.array(ls)


class Writer:
    """
    class that creates template and adds objects to XML
    """

    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__)),
                                                          encoding='utf-8', followlinks=False),
                                  keep_trailing_newline=True)

        # annotation.xml is needed for template
        self.annotation_template = environment.get_template('annotation.xml')
        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, xmin, ymin, xmax, ymax, name="person", pose='Unspecified', truncated=1, difficult=0):
        """
        adds objects to xml according to template
        """
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        """
        saves the xml in specific location
        :param annotation_path: where to save
        :return:
        """
        with open(annotation_path, 'w+') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)


if __name__ == '__main__':
    get_pascal(r"C:\Users\gabi9\Desktop\Lab\Auto\OKETZ\raw_data\Images\n02085620_7.jpg", np.array([1,3,4,6])
               ,r"C:\Users\gabi9\Desktop\Lab\Auto\automatic-behavior-analysis\AI_module\dog_detector")
