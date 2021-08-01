import cv2
import numpy


def canny_edge(image, sigma=0.33):
    v = numpy.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    return edges


img = cv2.imread("board.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.blur(gray, (5, 5))

edges = canny_edge(gray)

cv2.imwrite("gray-board.jpg", gray_blur)
cv2.imwrite("edges.jpg", edges)
