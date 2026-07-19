import cv2
import numpy as np


class LandmarkHeatmap:

    def __init__(self, radius=35):

        self.radius = radius

    def normalize(self, values):

        values = np.asarray(values, dtype=np.float32)

        if values.max() == values.min():
            return np.ones_like(values)

        return (values - values.min()) / (
            values.max() - values.min()
        )

    def generate(
        self,
        image,
        landmarks,
        landmark_errors
    ):

        h, w = image.shape[:2]

        heat = np.zeros(
            (h, w),
            dtype=np.float32
        )

        ids = list(landmark_errors.keys())

        values = self.normalize(
            list(landmark_errors.values())
        )

        for idx, error in zip(ids, values):

            x = int(landmarks[idx][0])
            y = int(landmarks[idx][1])

            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            layer = np.zeros_like(heat)

            cv2.circle(
                layer,
                (x, y),
                self.radius,
                float(error),
                -1
            )

            layer = cv2.GaussianBlur(
                layer,
                (0, 0),
                self.radius / 2
            )

            heat = np.maximum(
                heat,
                layer
            )

        heat = np.clip(
            heat * 255,
            0,
            255
        ).astype(np.uint8)

        return cv2.applyColorMap(
            heat,
            cv2.COLORMAP_JET
        )