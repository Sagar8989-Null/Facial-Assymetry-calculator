import numpy as np

from constants import (
    LANDMARK_PAIRS,
    CENTER_POINTS
)

from scipy.spatial import procrustes

from constants import SHAPE_REGIONS

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

    # def analyze(self, landmarks):

    #     mid_x = self.midline(landmarks)

    #     region_errors = {}

    #     for region,pairs in self.regions.items():

    #         region_errors[region]=self.region_error(

    #             landmarks,

    #             pairs,

    #             mid_x

    #         )

    #     shape_errors=self.shape_errors(

    #         landmarks,

    #         mid_x

    #     )

    #     pair_errors = self.pair_errors(
    #         landmarks,
    #         mid_x
    #     )   

    #     return{

    #         "midline":mid_x,

    #         "distance_errors":region_errors,

    #         "shape_errors":shape_errors,

    #         "pair_errors": pair_errors

    #     }

    def analyze(self, landmarks, normalize=False):

        if normalize:

            from normalize import LandmarkNormalizer

            landmarks = LandmarkNormalizer().normalize(
                landmarks
            )

        mid_x = self.midline(
            landmarks
        )

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

        pair_errors = self.pair_errors(
            landmarks,
            mid_x
        )

        return {

            "midline": mid_x,

            "distance_errors": distance_errors,

            "shape_errors": shape_errors,

            "pair_errors": pair_errors

        }
    
    def procrustes_error(
        self,
        landmarks,
        left_indices,
        right_indices,
        mid_x
        ):

        left = landmarks[left_indices].copy()

        right = landmarks[right_indices].copy()

        left[:,0] = 2*mid_x-left[:,0]

        _,_,error = procrustes(
            left,
            right
        )

        return error
    
    def shape_errors(
        self,
        landmarks,
        mid_x
    ):

        results={}

        for region,data in SHAPE_REGIONS.items():

            results[region]=self.procrustes_error(

                landmarks,

                data["left"],

                data["right"],

                mid_x

            )

        return results
    
    def landmark_errors(self, landmarks, mid_x):

        errors = {}

        for left, right in LANDMARK_PAIRS.items():

            left_point = landmarks[left]

            right_point = landmarks[right]

            mirrored = mirror(
                left_point,
                mid_x
            )

            error = np.linalg.norm(
                mirrored - right_point
            )

            errors[left] = error
            errors[right] = error

        return errors
    
    def pair_errors(self, landmarks, mid_x):

        errors = []

        for left, right in LANDMARK_PAIRS.items():

            left_point = landmarks[left]
            right_point = landmarks[right]

            mirrored = mirror(
                left_point,
                mid_x
            )

            error = np.linalg.norm(
                mirrored - right_point
            )

            midpoint = (
                mirrored + right_point
            ) / 2

            errors.append({

                "left": left,
                "right": right,

                "midpoint": midpoint,

                "error": float(error)

            })

        return errors