import numpy as np


class KabschAligner:

    def mirror(self, points, mid_x):
        mirrored = points.copy()
        mirrored[:, 0] = 2 * mid_x - mirrored[:, 0]
        return mirrored

    def align(self, source, target):
        """
        source -> mirrored left landmarks
        target -> right landmarks

        Returns:
            aligned_source
            rmse
        """

        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        # centroids
        c1 = np.mean(source, axis=0)
        c2 = np.mean(target, axis=0)

        A = source - c1
        B = target - c2

        # covariance
        H = A.T @ B

        U, S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        # reflection correction
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        aligned = (A @ R) + c2

        rmse = np.sqrt(
            np.mean(
                np.sum((aligned - target) ** 2, axis=1)
            )
        )

        return aligned, rmse

    def compare(self, left_points, right_points, mid_x):

        left_points = self.mirror(left_points, mid_x)

        _, rmse = self.align(
            left_points,
            right_points
        )

        return rmse