import numpy as np

from constants import (
    LANDMARK_PAIRS,
    CENTER_POINTS,
    SHAPE_REGIONS
)

from scipy.spatial import procrustes

from constants import SHAPE_REGIONS

from utils import mirror
from kabsch import KabschAligner
from landmark_pairs import REGIONS

class SymmetryAnalyzer:

    def __init__(self):
        self.kabsch = KabschAligner()
        self.regions = REGIONS

    # ------------------------------

    def midline(self, landmarks):

        pts = landmarks[CENTER_POINTS]

        return np.mean(pts[:,0])

    # ------------------------------

    def pair_error(self, landmarks, left, right, mid_x):

        L = landmarks[left]

        R = landmarks[right]

        L = mirror(L, mid_x)

        return np.linalg.norm(L - R)

    # ------------------------------

    def region_error(self, landmarks, pairs, mid_x):

        errors = []

        for left, right in pairs:

            errors.append(

                self.pair_error(
                    landmarks,
                    left,
                    right,
                    mid_x
                )

            )

        return np.mean(errors)

    # ------------------------------

    def overall_error(self, region_errors):

        return np.mean(

            list(region_errors.values())

        )

    # ------------------------------

    def analyze(self, landmarks):

        mid_x = self.midline(landmarks)

        distance_errors = {}

        for region, pairs in self.regions.items():

            distance_errors[region] = self.region_error(
                landmarks,
                pairs,
                mid_x
            )

        shape_errors = self.shape_errors(
            landmarks,
            mid_x
        )

        # normalize by interpupillary distance

        left_eye = np.mean(
            landmarks[[33,133]],
            axis=0
        )

        right_eye = np.mean(
            landmarks[[263,362]],
            axis=0
        )

        ipd = np.linalg.norm(
            right_eye-left_eye
        )

        for region in distance_errors:
            distance_errors[region] /= ipd

        for region in shape_errors:
            shape_errors[region] /= ipd

        return {

            "midline": mid_x,

            "distance_errors": distance_errors,

            "shape_errors": shape_errors

        }

        return{

            "midline":mid_x,

            "distance_errors":region_errors,

            "shape_errors":shape_errors

        }
    
    def shape_errors(self, landmarks, mid_x):

        results = {}

        for region, data in SHAPE_REGIONS.items():

            left = landmarks[data["left"]]

            right = landmarks[data["right"]]

            results[region] = self.kabsch.compare(
                left,
                right,
                mid_x
            )

        return results