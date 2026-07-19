import numpy as np


def distance(a, b):
    return np.linalg.norm(a - b)


def angle(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return np.degrees(np.arctan2(dy, dx))


def mirror(point, mid_x):
    p = point.copy()
    p[0] = 2 * mid_x - p[0]
    return p


def eye_center(landmarks, indices):
    return np.mean(landmarks[indices], axis=0)