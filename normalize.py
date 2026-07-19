import numpy as np
from constants import LEFT_EYE, RIGHT_EYE
from utils import eye_center


class LandmarkNormalizer:

    def normalize(self, landmarks):

        left = eye_center(landmarks, LEFT_EYE)
        right = eye_center(landmarks, RIGHT_EYE)

        eye_distance = np.linalg.norm(right - left)

        landmarks = landmarks.copy()

        landmarks /= eye_distance

        return landmarks