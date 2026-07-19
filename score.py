import numpy as np

from constants import REGION_WEIGHTS


class SymmetryScore:

    def error_to_score(self,error):

        return 100*np.exp(-6*error)

    def calculate(self,results):

        total=0

        weight_sum=0

        report={}

        for region,weight in REGION_WEIGHTS.items():

            d=results["distance_errors"][region]

            if region in results["shape_errors"]:

                s=results["shape_errors"][region]

                error=0.7*d+0.3*s

            else:

                error=d

            score=self.error_to_score(error)

            report[region]=round(score,2)

            total+=score*weight

            weight_sum+=weight

        overall=round(total/weight_sum,2)

        return overall,report