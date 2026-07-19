import numpy as np


CENTER_POINTS = [
    10,
    168,
    1,
    2,
    152
]


class MidlineEstimator:

    def estimate(self, landmarks):

        center = landmarks[CENTER_POINTS]

        x = np.mean(center[:, 0])

        return x