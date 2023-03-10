import time
import cv2 as cv
from display import Display
import numpy as np
import math

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

        fe = cv.goodFeaturesToTrack(np.mean(img, axis=2).astype(
            np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        return fe


feat = FeatureExtractor()


def process_frame(img):
    img = cv.resize(img, (W, H))

    kp = feat.extract(img)

    for p in kp:
        u, v = map(lambda x: int(round(x)), p[0])
        if (v > H//2):
            print(u, v)
            cv.circle(img, (u, v), color=(0, 0, 255), radius=2)
    disp.paint(img)


def apply_Hough(img):
    img = cv.Canny(img, 50, 200, None, 3)
    cdst = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(img, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 100*(-b)), int(y0 + 100*(a)))
            pt2 = (int(x0 - 100*(-b)), int(y0 - 100*(a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if (l[1] > H//2 and l[1] > H//2):
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]),
                        (0, 0, 255), 3, cv.LINE_AA)

    # cv.imshow("Source", src)
    process_frame(cdstP)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


if __name__ == "__main__":
    cap = cv.VideoCapture("./video/v.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            # process_frame(frame)
            apply_Hough(frame)
        else:
            break
