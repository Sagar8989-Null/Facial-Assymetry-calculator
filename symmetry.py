import numpy as np

from constants import (
    LANDMARK_PAIRS,
    CENTER_POINTS
)

from utils import mirror


class SymmetryAnalyzer:

    def __init__(self):

        self.regions = {

            "eyes":[
                (33,263),
                (133,362),
                (160,387),
                (159,386),
                (158,385),
                (157,384),
                (173,398)
            ],

            "eyebrows":[
                (70,300),
                (63,293),
                (105,334),
                (66,296),
                (107,336)
            ],

            "nose":[
                (129,358),
                (98,327),
                (97,326)
            ],

            "mouth":[
                (61,291),
                (40,270),
                (39,269),
                (37,267),
                (84,314),
                (181,405),
                (91,321),
                (146,375)
            ],

            "jaw":[
                (234,454),
                (93,323),
                (132,361),
                (58,288),
                (172,397),
                (136,365),
                (150,379),
                (149,378)
            ]

        }

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

        region_errors = {}

        for region, pairs in self.regions.items():

            region_errors[region] = self.region_error(

                landmarks,
                pairs,
                mid_x

            )

        total = self.overall_error(

            region_errors

        )

        return {

            "midline": mid_x,

            "regions": region_errors,

            "overall_error": total

        }