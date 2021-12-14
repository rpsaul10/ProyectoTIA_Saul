import Utils
import os
from cv2 import imread


def ProximityToThresholds(image):
    mature, area, _ = Utils.mainProcess(image)

    proximity_mature = mature - Utils.THRESHOLD_MATURE
    proximity_area = area - Utils.THRESHOLD_AREA

    return [proximity_mature, proximity_area]


def printDic(dic: dict):
    print('Results'.center(21, '='))
    print()
    print('Mature - <-------- 0 --------> + Green')
    print('Small - <-------- 0 --------> + Big')
    print()
    print('FileName |  Mature Proximity  |  Area Proximity  |\n')
    for key in dic.keys():
        print(f'{key}: | {dic[key][0]} | {dic[key][1]} |')


result_dic = {}

for path in os.listdir('images'):
    result_dic[path] = ProximityToThresholds(imread(f'images/{path}'))

printDic(result_dic)