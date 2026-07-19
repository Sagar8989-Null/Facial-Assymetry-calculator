import numpy as np

from constants import REGION_WEIGHTS


class SymmetryScore:

    def error_to_score(self, error):

        score = 100 * np.exp(-6 * error)

        return np.clip(score,0,100)

    def calculate(self, regions):

        weighted = 0

        total_weight = 0

        for region, error in regions.items():

            weight = REGION_WEIGHTS[region]

            weighted += self.error_to_score(error) * weight

            total_weight += weight

        return round(weighted / total_weight,2)