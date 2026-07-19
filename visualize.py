import os
import cv2
import numpy as np


class HeatmapVisualizer:

    def __init__(self, alpha=0.45):

        self.alpha = alpha

    def overlay(
        self,
        image,
        heatmap
    ):

        return cv2.addWeighted(
            image,
            1.0,
            heatmap,
            self.alpha,
            0
        )

    def save(
        self,
        image,
        heatmap,
        output_dir="output"
    ):

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        overlay = self.overlay(
            image,
            heatmap
        )

        cv2.imwrite(
            os.path.join(
                output_dir,
                "heatmap.png"
            ),
            heatmap
        )

        cv2.imwrite(
            os.path.join(
                output_dir,
                "overlay.png"
            ),
            overlay
        )

        return overlay