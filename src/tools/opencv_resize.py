"""
OpenCV resize
comparing various methods of resize interpolation
"""
from __future__ import print_function
import argparse

import os
import cv2
import numpy as np


if __name__ == '__main__':
    # Parse
    parser = argparse.ArgumentParser(description='OpenCV resize comparison')
    parser.add_argument('--input', '-i', default=None, help='input file path')
    parser.add_argument('--output', '-o', default=None, help='output folder directory')
    args = parser.parse_args()

    filepath = os.path.dirname(os.path.realpath(__file__))

    if args.input is not None:
        photo_file_path = args.input
    else:
        photo_file_path = os.path.join(filepath, '../../assets/compare/0/photo0_xinput.jpg')

    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = os.path.join(filepath, '../../assets/compare/0')

    input_img = cv2.imread(photo_file_path, cv2.IMREAD_COLOR)
    input_image_height = input_img.shape[0]
    input_image_width = input_img.shape[1]
    output_image_height = 2 * input_image_height
    output_image_width = 2 * input_image_width

    scaled_input_img = cv2.resize(input_img, (output_image_width, output_image_height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(output_dir, 'nearest.jpg'), scaled_input_img)
    scaled_input_img = cv2.resize(input_img, (output_image_width, output_image_height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_dir, 'linear.jpg'), scaled_input_img)
    scaled_input_img = cv2.resize(input_img, (output_image_width, output_image_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, 'area.jpg'), scaled_input_img)
    scaled_input_img = cv2.resize(input_img, (output_image_width, output_image_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, 'cubic.jpg'), scaled_input_img)
    scaled_input_img = cv2.resize(input_img, (output_image_width, output_image_height), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(output_dir, 'lanczos.jpg'), scaled_input_img)


