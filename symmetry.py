import numpy as np

from regions import REGIONS


class SymmetryCalculator:

    def mirror(self, point, mid_x):

        p = point.copy()

        p[0] = 2 * mid_x - p[0]

        return p

    def region_error(self, landmarks, pairs, mid_x):

        errors = []

        for left,right in pairs:

            L = landmarks[left]

            R = landmarks[right]

            L = self.mirror(L,mid_x)

            d = np.linalg.norm(L-R)

            errors.append(d)

        return np.mean(errors)

    def calculate(self, landmarks, mid_x):

        results = {}

        for region,pairs in REGIONS.items():

            results[region] = self.region_error(
                landmarks,
                pairs,
                mid_x
            )

        return results