import time
import cv2 as cv
from display import Display
import numpy as np

H = 1080//2
W = 1920//2

disp = Display(H, W)


class FeatureExtractor(object):
    GX = 16//8
    GY = 12//8

    def __init__(self):
        self.orb = cv.ORB_create(1000)

    def extract(self, img):
        # sy = img.shape[0]//self.GY
        # sx = img.shape[1]//self.GX
        # okp = []
        # for ry in range(0, img.shape[0], sy):
        #     for rx in range(0, img.shape[1], sx):
        #         im = img[ry:ry+sy, rx:rx+sx]
        #         kp = self.orb.detect(im, None)
        #         for p in kp:
        #             p.pt = (p.pt[0]+rx, p.pt[1]+ry)
        #             okp.append(p)

        # return okp

        fe =  cv.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        return fe

feat = FeatureExtractor()


def process_frame(img): 
    img = cv.resize(img, (W, H))

    kp = feat.extract(img)
    for p in kp:
        u, v = map(lambda x: int(round(x)), p[0])
        cv.circle(img, (u, v), color=(0, 0, 255), radius=2)
    disp.paint(img)


if __name__ == "__main__":
    cap = cv.VideoCapture("./video/vid1.webm")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
