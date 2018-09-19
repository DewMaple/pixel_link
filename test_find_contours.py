import sys

import cv2

from tf_extended import find_contours, filter_contours

args = sys.argv[1:]

image_file = args[0]

img = cv2.imread(image_file)
contours = find_contours(img)
print('Len of contours: {}'.format(len(contours)))
contours, bboxes = filter_contours(contours, 10, 8)

print('Filtered len of contours: {}'.format(len(contours)))
cnt_img = img.copy()
cv2.drawContours(cnt_img, contours, -1, (0, 0, 255), 2)

cv2.imwrite('contours.jpg', cnt_img)

for box in bboxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('bboxes.jpg', img)
