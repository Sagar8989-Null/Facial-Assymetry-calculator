import numpy as np


WEIGHTS = {

    "eyes":0.30,

    "eyebrows":0.10,

    "nose":0.20,

    "mouth":0.25,

    "jaw":0.15
}


class SymmetryScore:

    def region_score(self,error):

        score = 100 - error*600

        return np.clip(score,0,100)

    def calculate(self,results):

        total = 0

        for region,error in results.items():

            total += self.region_score(error) * WEIGHTS[region]

        return round(total,2)