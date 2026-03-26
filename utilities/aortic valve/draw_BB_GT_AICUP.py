# encoding: utf-8
import cv2
import numpy as np
import os

IMAGE_PATH = '/home/ai/mmdetection3x/imgs/patient0051_0222.png'
GT_PATH    = '/home/ai/mmdetection3x/imgs/ground_truth/patient0051_0222.txt'
OUTPUT_PATH = '/home/ai/mmdetection3x/imgs/test_gt_box.png'

img = cv2.imread(IMAGE_PATH)
H, W = img.shape[:2]
print('Image size: W={} H={}'.format(W, H))

# load GT - format: class cx cy w h (no image_name!)
with open(GT_PATH, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_id = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        w  = float(parts[3])
        h  = float(parts[4])

        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        print('class={} cx={} cy={} w={} h={} -> ({},{},{},{})'.format(
              class_id, cx, cy, w, h, x1, y1, x2, y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, 'GT aortic_valve', (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imwrite(OUTPUT_PATH, img)
print('Saved:', OUTPUT_PATH)