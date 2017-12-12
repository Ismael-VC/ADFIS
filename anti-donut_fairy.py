#!/usr/bin/env python2

# coding:utf-8

from __future__ import print_function

from time import sleep

import cv2 as opencv
from cv2 import error as OpenCVError

from user_train import Model
from utils import (
    adfis, EXIT_CODE, CASCADE_PATH, is_user, is_face, lockscreen, banner,
    terminate
)

LIMIT = 10

def main():
    capture = opencv.VideoCapture(0)

    model = Model()
    model.load()

    time_since_NO_user = 0
    time_since_user = 0

    while True:
        try:
            _, frame = capture.read()

            try:
                frame_gray = opencv.cvtColor(frame, opencv.COLOR_BGR2GRAY)
            except OpenCVError:
                raise OpenCVError("ADFIS: Camera not found or already in use.")

            cascade = opencv.CascadeClassifier(CASCADE_PATH)
            face_rect = cascade.detectMultiScale(
                frame_gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(10, 10)
            )

            if is_face(face_rect):
                time_since_user += 1
                time_since_NO_user = 0

                for rect in face_rect:
                    x, y = rect[0:2]
                    width, height = rect[2:4]
                    image = frame[y - 10: y + height, x: x + width]

                    result = model.predict(image)

                    if is_user(result):
                        adfis('User recognized.')
                        sleep(1)
                    else:
                        adfis('User not recognized, locking screen.')
                        lockscreen()

            else:
                adfis("Face not detected: {0}.".format(time_since_NO_user))

                if time_since_NO_user == LIMIT:
                    adfis("User is not present, locking screen.")
                    lockscreen()

                time_since_NO_user += 1
                time_since_user = 0
                sleep(1)

            key_press = opencv.waitKey(100)

            if key_press == EXIT_CODE:
                terminate()
                break
        except KeyboardInterrupt:
            terminate()
            break

    capture.release()
    opencv.destroyAllWindows()


if __name__ == '__main__':
    main()
