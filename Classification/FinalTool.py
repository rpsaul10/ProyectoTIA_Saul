import cv2
import os
import Utils
from time import sleep


for path in os.listdir('images'):
    image = cv2.imread(f'images/{path}')
    image = Utils.Classify(image)
    image = Utils.putDefaultText(image)
    cv2.imshow('Pineapple Classification', image)
    cv2.waitKey(1)
    sleep(2)
cv2.destroyAllWindows()