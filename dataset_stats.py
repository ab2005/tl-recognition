#!/usr/bin/env python
"""
Sample script to show some numbers for the dataset.
"""
import os
import glob
import sys
import logging
import cv2

MIN_WIDTH = 2
MAX_WIDTH = 200
MIN_HEIGHT = 2
MAX_HEIGHT = 300

def get_all_labels(annotation='*/train*.txt'):
    lines = []
    for file in glob.glob(annotation):
        with open(file) as f:
            lines.extend(f.read().splitlines())
    return lines

def quick_stats(annotation, output_troubled=None):
    if output_troubled != None:
        troubled = open(output_troubled, 'w')

    images = get_all_labels(annotation)

    widths = []
    heights = []
    sizes = []

    num_images = len(images)
    num_lights = 0
    invalidCount = 0
    for image in images:
        ss = image.split(' ')
        if len(ss) < 2:
            logging.warning('Invalid line:'+image)
            continue
        path = ss[0]
        num_boxes = int(ss[1])
        num_lights += num_boxes

        valid = True
        for i in range(0,num_boxes):
            n = i * 4 + 2
            x_min = int(ss[n])
            y_min = int(ss[n+1])
            x_max = int(ss[n+2])
            y_max = int(ss[n+3])
            if x_min >= x_max:
                x_min, x_max = x_max, x_min
            if y_min >= y_max:
                y_min, y_max = y_max, y_min

            width = x_max - x_min
            height = y_max - y_min
            if width < MIN_WIDTH:
                logging.warning('Box width smaller than min ('+str(width)+'x'+str(height)+') at ' + image)
                valid = False
            if width > MAX_WIDTH:
                logging.warning('Box width bigger than max ('+str(width)+'x'+str(height)+') at ' + image)
                valid = False
            if height < MIN_HEIGHT:
                logging.warning('Box height smaller than min ('+str(width)+'x'+str(height)+') at ' + image)
                valid = False
            if height > MAX_HEIGHT:
                logging.warning('Box height bigger than max ('+str(width)+'x'+str(height)+') at ' + image)
                valid = False

            if valid:
                widths.append(width)
                heights.append(height)
                sizes.append(width * height)

        if not valid:
            invalidCount += 1
            if output_troubled is not None:
                troubled.write(image+'\n')

    avg_width = sum(widths) / float(len(widths))
    avg_height = sum(heights) / float(len(heights))
    avg_size = sum(sizes) / float(len(sizes))

    median_width = sorted(widths)[len(widths) // 2]
    median_height = sorted(heights)[len(heights) // 2]
    median_size = sorted(sizes)[len(sizes) // 2]

    print('Number of images:', num_images - invalidCount, 'invalid:',invalidCount)
    print('Number of traffic lights:', num_lights, '\n')

    print('Minimum width:', min(widths))
    print('Average width:', avg_width)
    print('median width:', median_width)
    print('maximum width:', max(widths), '\n')

    print('Minimum height:', min(heights))
    print('Average height:', avg_height)
    print('median height:', median_height)
    print('maximum height:', max(heights), '\n')

    print('Minimum size:', min(sizes))
    print('Average size:', avg_size)
    print('median size:', median_size)
    print('maximum size:', max(sizes), '\n')

    if output_troubled is not None:
        troubled.close()

def write_labeled_images(input_annotation, output_folder=None, output_video=None):
    """
    Draws and saves pictures with labeled traffic lights.
    """

    if not os.path.isfile(input_annotation):
        return

    video = None
    images = get_all_labels(input_annotation)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for i, image in enumerate(images):
        ss = image.split(' ')
        if len(ss) < 6:
            logging.warning('Invalid line:'+image)
            continue

        path = ss[0]
        img = cv2.imread(path)
        if img is None:
            logging.warning('Missed image:'+path)
            continue

        num_boxes = int(ss[1])
        for i in range(0,num_boxes):
            n = i * 4 + 2
            x_min = int(ss[n])
            y_min = int(ss[n+1])
            x_max = int(ss[n+2])
            y_max = int(ss[n+3])
            cv2.rectangle(img,(x_min,y_min), (x_max,y_max), (0, 255, 0))
            if output_folder is not None:
                cv2.imwrite(os.path.join(output_folder,os.path.basename(path)),img)
            #print(cv2.mean(img))
            if output_video is not None:
                if video is None:
                    height , width , layers =  img.shape
                    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                    print(img.shape)
                    video = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))
                video.write(img)

    if video is not None:
        cv2.destroyAllWindows()
        video.release()


quick_stats('/data/traffic_lights/nauto/train*.txt', 'nauto_troubled.txt')
write_labeled_images('nauto_troubled.txt', output_video='nauto_troubled.avi')
quick_stats('/data/traffic_lights/bosch/train*.txt', 'bosch_troubled.txt')
write_labeled_images('bosch_troubled.txt', output_video='bosch_troubled.avi')
quick_stats('/data/traffic_lights/viva/train*.txt', 'viva_troubled.txt')
write_labeled_images('viva_troubled.txt', output_video='viva_troubled.avi')
quick_stats('/data/traffic_lights/lara/train*.txt', 'lara_troubled.txt')
write_labeled_images('lara_troubled.txt', output_video='lara_troubled.avi')