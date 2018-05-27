import random
import time

import cv2
from heatmappy import Heatmapper
from skimage.color import label2rgb
from skimage.feature import CENSURE
from skimage.filters import roberts, sobel, threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from time import sleep
from os.path import isfile
from main import get_model


class LogoFinder:
    def __init__(self):
        self.heatmapper = Heatmapper(
            point_diameter=15,
            point_strength=0.05,
            opacity=3,
            colours='reveal',
        )
        self.heatmap_name = 'heatmap.png'
        self.detector = CENSURE()

    @staticmethod
    def read_image(image_path):
        image = cv2.imread(image_path)
        bgr = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return bgr, rgb, gray

    def make_heatmap(self, image_path):
        bgr_heatmap, rgb_heatmap, gray_heatmap = self.read_image(image_path)

        # Find contours and edges
        contours = find_contours(gray_heatmap, 100, fully_connected='high', positive_orientation='high')
        edge_roberts = roberts(gray_heatmap)
        edge_sobel = sobel(gray_heatmap)

        self.detector.detect(gray_heatmap)
        thresh = threshold_otsu(gray_heatmap)
        bw = closing(gray_heatmap > thresh, square(3))
        cleared = clear_border(bw)
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=gray_heatmap)

        edges = edge_roberts / 2 + edge_sobel / 2
        dots = list()

        for x in range(edges.shape[1]):
            for y in range(edges.shape[0]):
                for i in range(int(edges[y][x] * 2)):
                    dots.append([x, y])

        for n, point in enumerate(self.detector.keypoints):
            for i in range(self.detector.scales[n] * 10):
                dots.append([point[1] + (random.random() - 0.5) * 10, point[0] + (random.random() - 0.5) * 10])

        for n, contour in enumerate(contours):
            if len(contour) > 50:
                for dot in contour:
                    dots.append([dot[1], dot[0]])

        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 100:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                for x in range(minc, maxc):
                    for y in range(minr, maxr, 8):
                        dots.append([x, y])

        heatmap = self.heatmapper.heatmap_on_img_path(dots, image_path)
        heatmap.save(self.heatmap_name)

        bgr_heatmap, rgb_heatmap, gray_heatmap = self.read_image(self.heatmap_name)

        thresh = threshold_otsu(gray_heatmap)
        bw = closing(gray_heatmap > thresh, square(3))
        cleared = clear_border(bw)
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=gray_heatmap)
        for n, region in enumerate(regionprops(label_image)):
            minr, minc, maxr, maxc = region.bbox
            if maxr - minr > 30 and maxc - minc > 30:
                yield minr, minc, maxr, maxc


image_dir = '/Users/thelacker/PycharmProjects/logos/test_photos/'
image_name = 'test2.jpeg'
image_path = image_dir+image_name

l = LogoFinder()
bgr, rgb, gray = l.read_image(image_path)

t = time.time()
croped_images = l.make_heatmap(image_path)

model = get_model()

for n, crop in enumerate(croped_images):
    minr, minc, maxr, maxc = crop
    crop_img = bgr[minr:maxr, minc:maxc]
    tmp_name = 'res/{0}-{1}.jpg'.format(image_name[:-4], n)
    cv2.imwrite(tmp_name, crop_img)
    img = load_img(tmp_name,False,target_size=(300,300))
    x = img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict_classes(x)
    prob = model.predict_proba(x)
    if prob[0][0] < 0.8:
        print (tmp_name, prob)
        cv2.imwrite('result/{0}-{1}.jpg'.format(image_name[:-4], n), crop_img)


