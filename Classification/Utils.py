from math import sqrt
import cv2
import numpy
from scipy import ndimage

THRESHOLD_AREA = 51540.75
THRESHOLD_MATURE = 0.0018790620379149914

YELLOW = (0, 233, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def getMinPoint(points, static_point):
    point_min = None
    min_dist = None
    for point in points:
        if not (min_dist is None):
            temp_dist = dist2points(point, static_point)
            if temp_dist < min_dist:
                min_dist = temp_dist
                point_min = point
        else:
            point_min = point
            min_dist = dist2points(point, static_point)
    return point_min


def getFourPoints(points, image):
    width = len(image[0])
    height = len(image)

    return getMinPoint(points, [0, 0]), getMinPoint(points, [width, 0]),\
        getMinPoint(points, [0, height]), getMinPoint(points, [width, height])


def cutCrown(points, image):
    width = len(image[0])
    height = len(image)

    threshold = int(height * .27)
    threshold2 = int(width / 2)

    right = list()
    left = list()
    for po in points:
        if not ((height - threshold) > po[0][1] > threshold):
            continue

        if po[0][0] < threshold2:
            left.append(po[0])
        else:
            right.append(po[0])

    right = numpy.asarray(sorted(right, key=lambda x: x[1]))[::6]
    left = numpy.asarray(sorted(left, key=lambda x: x[1]),)[::6]
    final_list = []
    for p1 in right:
        for p2 in left:
            final_list.append([list(p1), list(p2), dist2points(p1, p2)])
            if len(final_list) < 20:
                continue
            # print(len(final_list))
            maxi = max(final_list, key=lambda x: x[2])
            final_list.remove(maxi)
    return final_list


def dist2points(point1, point2):
    return sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))


def cutImage(img, mask, box_points):
    cut_mask = mask[box_points[1][1]: box_points[3][1], box_points[0][0]: box_points[1][0]]
    cut_image = img[box_points[1][1]: box_points[3][1], box_points[0][0]: box_points[1][0]]
    return cut_image, cut_mask


def boxPointsFromMask(mask):
    contours3, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_cont = max(contours3, key=cv2.contourArea)
    area = cv2.minAreaRect(max_cont)
    return numpy.int0(cv2.boxPoints(area)), max_cont


def mainProcess(image):
    image = image.copy()
    gauss = cv2.GaussianBlur(image, (9, 9), 0)
    canny = cv2.Canny(cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY), 25, 110)

    kernel = numpy.ones((7, 7), numpy.uint8)
    fat_borders = cv2.dilate(canny.copy(), kernel, iterations=2)

    mask_fill = ndimage.binary_fill_holes(fat_borders).astype('uint8')

    box, _ = boxPointsFromMask(mask_fill)
    box = getFourPoints(box, image)

    cut_image, cut_mask = cutImage(image, mask_fill, box)

    contours2, _ = cv2.findContours(cut_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_cont = max(contours2, key=cv2.contourArea)

    # ***Cut Crown***
    crown = cutCrown(max_cont, cut_mask)
    maxim = max(crown, key=lambda x: x[0][1] + x[1][1])

    no_crown = cut_image[maxim[0][1]:, :]
    no_crown_mask = cut_mask[maxim[0][1]:, :]

    # *** Find Body ***
    box2, final_cont = boxPointsFromMask(no_crown_mask)
    final_cut_image, final_cut_mask = cutImage(no_crown, no_crown_mask, getFourPoints(box2, no_crown))

    # ***Calc Histogram***
    final_cut_image[final_cut_mask != 1] = 0
    histogram = cv2.calcHist([final_cut_image], [1], None, [256], [0, 256])
    cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mean = numpy.mean(histogram)

    histogram2 = cv2.calcHist([final_cut_image], [2], None, [256], [0, 256])
    cv2.normalize(histogram2, histogram2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mean2 = numpy.mean(histogram2[1:])

    histogram2 = cv2.calcHist([final_cut_image], [0], None, [256], [0, 256])
    cv2.normalize(histogram2, histogram2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mean3 = numpy.mean(histogram2)

    return numpy.std([mean, mean2, mean3]), cv2.contourArea(final_cont), box


def Classify(image):
    mature, area, points = mainProcess(image)

    if (mature < THRESHOLD_MATURE) and (area > THRESHOLD_AREA):
        # Yellow if is mature and big
        rectangleColor = YELLOW
    elif mature > THRESHOLD_MATURE and area > THRESHOLD_AREA:
        # Green if is green and big
        rectangleColor = GREEN
    elif mature < THRESHOLD_MATURE and area < THRESHOLD_AREA:
        # Red if is mature and small
        rectangleColor = RED
    else:
        # Blue if is green and small
        rectangleColor = BLUE

    return cv2.rectangle(image.copy(), points[0], points[3], rectangleColor, 3)


def putDefaultText(image):
    t1 = "Yellow: Big and Mature"
    t2 = "Green: Big and Green"
    t3 = "Red: Small and Mature"
    t4 = "Blue: Small and Green"
    image = cv2.putText(image.copy(), t1, (4, 60), cv2.FONT_ITALIC, .7, YELLOW, thickness=2)
    image = cv2.putText(image, t2, (4, 90), cv2.FONT_ITALIC, .7, GREEN, thickness=2)
    image = cv2.putText(image, t3, (4, 120), cv2.FONT_ITALIC, .7, RED, thickness=2)
    return cv2.putText(image, t4, (4, 150), cv2.FONT_ITALIC, .7, BLUE, thickness=2)