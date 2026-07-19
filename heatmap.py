import cv2
import numpy as np


class FacialHeatmap:

    def __init__(self, sigma=55):

        self.sigma = sigma

    def _normalize_errors(self, region_errors):

        values = np.array(
            list(region_errors.values()),
            dtype=np.float32
        )

        if values.max() == values.min():

            return {
                k: 1.0
                for k in region_errors
            }

        values = (
            values - values.min()
        ) / (
            values.max() - values.min()
        )

        return {
            key: float(value)
            for key, value in zip(
                region_errors.keys(),
                values
            )
        }

    def generate(
        self,
        image,
        landmarks,
        region_errors,
        region_points
    ):

        h, w = image.shape[:2]

        heat = np.zeros(
            (h, w),
            dtype=np.float32
        )

        errors = self._normalize_errors(
            region_errors
        )

        for region, indices in region_points.items():

            if region not in errors:
                continue

            intensity = errors[region]

            pts = np.array(
                [
                    landmarks[i][:2]
                    for i in indices
                ],
                dtype=np.int32
            )

            mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            cv2.fillConvexPoly(
                mask,
                pts,
                255
            )

            heat[mask > 0] = np.maximum(
                heat[mask > 0],
                intensity
            )

        heat = cv2.GaussianBlur(
            heat,
            (0, 0),
            self.sigma
        )

        heat = np.clip(
            heat * 255,
            0,
            255
        ).astype(np.uint8)

        colored = cv2.applyColorMap(
            heat,
            cv2.COLORMAP_JET
        )

        return colored