import numpy as np

from constants import (
    REGION_WEIGHTS,
    FEATURE_WEIGHTS
)


class SymmetryScore:

    def __init__(self):

        self.max_distance_error = {
            "eyes": 12.0,
            "eyebrows": 10.0,
            "nose": 8.0,
            "mouth": 14.0,
            "jaw": 18.0
        }

        self.max_shape_error = {
            "eyes": 6.0,
            "eyebrows": 6.0,
            "jaw": 8.0
        }

    def normalize(self, value, maximum):

        value = min(value, maximum)

        return value / maximum

    def score_from_error(self, error):

        return (1.0 - error) * 100

    def calculate(self, results):

        report = {}

        total = 0.0
        total_weight = 0.0

        for region in REGION_WEIGHTS:

            distance_error = self.normalize(
                results["distance_errors"][region],
                self.max_distance_error[region]
            )

            if region in results["shape_errors"]:

                shape_error = self.normalize(
                    results["shape_errors"][region],
                    self.max_shape_error[region]
                )

            else:

                shape_error = 0

            fusion = FEATURE_WEIGHTS[region]

            final_error = (
                fusion["distance"] * distance_error +
                fusion["shape"] * shape_error
            )

            score = self.score_from_error(
                final_error
            )

            report[region] = round(score, 2)

            total += score * REGION_WEIGHTS[region]

            total_weight += REGION_WEIGHTS[region]

        overall = round(total / total_weight, 2)

        return overall, report