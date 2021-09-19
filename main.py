import cv2
import numpy
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean


def write_crop_images(img, points, img_count=0, folder_path='./Data/raw_data/'):
    num_list = []
    shape = list(numpy.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= 8:
        range_num = 8
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - 8) - (row * 11)
        num_list.append(range(start, end, -1))

    for row in num_list:
        for s in row:
            base_len = numpy.linalg.norm(numpy.array(points[s]) - numpy.array(points[s + 1]))
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            img_count += 1
            cv2.imwrite('./dist/' + str(img_count) + '.jpeg', cropped)
            # print(folder_path + 'data' + str(img_count) + '.jpeg')
    return img_count


def canny_edge(image, sigma=0.33):
    v = numpy.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    return edges


def hough_line(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLines(edges, 1, numpy.pi / 180, 125, min_line_length, max_line_gap)
    lines = numpy.reshape(lines, (-1, 2))
    return lines


def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < numpy.pi / 4 or theta > numpy.pi - numpy.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines


def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = numpy.array([[numpy.cos(t_h), numpy.sin(t_h)], [numpy.cos(t_v), numpy.sin(t_v)]])
            b = numpy.array([r_h, r_v])
            inter_point = numpy.linalg.solve(a, b)
            points.append(inter_point)
    return numpy.array(points)

def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (numpy.mean(numpy.array(arr)[:, 0]), numpy.mean(numpy.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])


def augment_points(points):
    points_shape = list(numpy.shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 11)):
        start = row * 11
        end = (row * 11) + 10
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = mean(rw_y)
        for i in range(len(rw_x)):
            point = (rw_x[i], y_mean)
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points


img = cv2.imread("board2.jpeg")
img_copy = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.blur(gray, (5, 5))

edges_image = canny_edge(gray)
board_lines = hough_line(edges_image)
h_lines, v_lines = h_v_lines(board_lines)
intersection_points = line_intersections(h_lines, v_lines)
points = cluster_points(intersection_points)
points = augment_points(points)


for p in points:
    cv2.circle(img, (int(p[0]), int(p[1])), radius=3, color=(255, 0, 0), thickness=-1)

if board_lines is not None:
    for i in range(0, len(board_lines)):
        rho = board_lines[i][0]
        theta = board_lines[i][1]
        a = numpy.cos(theta)
        b = numpy.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img_copy, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)


x_list = write_crop_images(img, points, 0)

cv2.imwrite("gray-board.jpg", gray_blur)
cv2.imwrite("edges.jpg", edges_image)
cv2.imwrite("result.jpg", img)
cv2.imwrite("result-lines.jpg", img_copy)