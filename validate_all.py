#!/usr/bin/python
import os
import glob

MIN_WIDTH=2
MAX_WIDTH=200
MIN_HEIGHT=2
MAX_HEIGHT=400

lines = []
for file in glob.glob('/data/traffic_lights/*/train*.txt'):
    with open(file) as f:
        lines.extend(f.read().splitlines())

for i, val in enumerate(lines):
    ss = val.split(' ')
    if len(ss) < 6: continue
    path = ss[0]
    num_boxes = int(ss[1])
    valid = True
    for i in range(0, num_boxes):
        n = i * 4 + 2
        x_min = int(ss[n])
        y_min = int(ss[n + 1])
        x_max = int(ss[n + 2])
        y_max = int(ss[n + 3])
        width = x_max - x_min
        height = y_max - y_min
        if width < MIN_WIDTH:
            print('Box width smaller than min (' + str(width) + 'x' + str(height) + ') at ' + val)
        if width > MAX_WIDTH:
            print('Box width bigger than max (' + str(width) + 'x' + str(height) + ') at ' + val)
        if height < MIN_HEIGHT:
            print('Box height smaller than min (' + str(width) + 'x' + str(height) + ') at ' + val)
        if height > MAX_HEIGHT:
            print('Box height bigger than max (' + str(width) + 'x' + str(height) + ') at ' + val)

        if x_min >= x_max or y_min >= y_max:
            print('invalid:', val)
            valid = False
            break

    if not os.path.isfile(path):
        print('Missing image:', path)
