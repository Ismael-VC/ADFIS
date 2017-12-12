#!/usr/bin/env python2

# coding:utf-8

from __future__ import print_function

import os
import subprocess

import cv2 as opencv
import numpy as np

from textwrap import dedent as dd

IMAGE_SIZE = 64
EXIT_CODE = 27
CASCADE_PATH = ("/usr/share/opencv/haarcascades/"
                "haarcascade_frontalface_default.xml")
IMAGES = []
LABELS = []


def banner():
    print(dd(
        r"""
           _____  ________  ___________.___  _________
          /  _  \ \______ \ \_   _____/|   |/   _____/
         /  /_\  \ |    |  \ |    __)  |   |\_____  \
        /    |    \|    `   \|     \   |   |/        \
        \____|__  /_______  /\___  /   |___/_______  /
                \/        \/     \/                \/

        May the "Donut Fairy" NOT be with you!

        """
    ))


def clear():
    subprocess.call("clear")


def adfis(*messages):
    clear()
    banner()
    print("ADFIS: ", *messages)


def lockscreen():
    subprocess.call(
        "qdbus org.freedesktop.ScreenSaver /ScreenSaver Lock",
        shell=True
    )
# TODO: use walk funtion


def get_padding_size(image):
    height, width, _ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = (0, 0, 0, 0)

    if height < longest_edge:
        height_delta = longest_edge - height
        top = height_delta // 2
        bottom = height_delta - top

    elif width < longest_edge:
        width_delta = longest_edge - width
        left = width_delta // 2
        right = width_delta - left

    else:
        pass

    return top, bottom, left, right


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = get_padding_size(image)
    black = [0, 0, 0]
    constant = opencv.copyMakeBorder(
        image, top, bottom, left, right,
        opencv.BORDER_CONSTANT, value=black
    )
    resized_image = opencv.resize(constant, (height, width))

    return resized_image


def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        adfis(abs_path)
        if os.path.isdir(abs_path):
            traverse_dir(abs_path)

        else:
            file_name, extension = os.path.splitext(file_or_dir)
            if extension == '.jpg':
                image = read_image(abs_path)
                IMAGES.append(image)
                LABELS.append(file_name)

    return IMAGES, LABELS


def read_image(file_path):
    image = opencv.imread(file_path, 0)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('user') else 1 for label in labels])

    return images, labels


def is_user(prediction):
    return prediction == 0


def is_face(face_rect):
    return len(face_rect) > 0


def terminate():
    adfis('Sutting down ...watch out of the "Donut Fairy"!\n')
