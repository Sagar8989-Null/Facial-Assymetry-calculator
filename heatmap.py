import cv2
import numpy as np


class AsymmetryHeatmap:

    def __init__(
        self,
        sigma=45,
        alpha=0.55
    ):

        self.sigma = sigma
        self.alpha = alpha

    def normalize(self, pair_errors):

        values = np.array(
            [p["error"] for p in pair_errors],
            dtype=np.float32
        )

        mn = values.min()
        mx = values.max()

        if mx == mn:
            values[:] = 1.0
        else:
            values = (values - mn) / (mx - mn)

        for p, v in zip(pair_errors, values):

            p["weight"] = float(v)

        return pair_errors

    def generate(
        self,
        image,
        pair_errors
    ):

        h, w = image.shape[:2]

        heat = np.zeros(
            (h, w),
            dtype=np.float32
        )

        pair_errors = self.normalize(
            pair_errors
        )

        yy, xx = np.mgrid[
            0:h,
            0:w
        ]

        for pair in pair_errors:

            x = pair["midpoint"][0]
            y = pair["midpoint"][1]  

            weight = pair["weight"]
            print(pair["midpoint"], pair["error"])

            gaussian = np.exp(

                -(
                    (xx - x) ** 2 +
                    (yy - y) ** 2
                )

                /

                (2 * self.sigma ** 2)

            )

            heat += gaussian * weight

        heat /= heat.max()

        heat = (heat * 255).astype(np.uint8)

        heat = cv2.applyColorMap(
            heat,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image,
            1.0,
            heat,
            self.alpha,
            0
        )
        print(heat.max(), heat.min())

        return heat, overlay