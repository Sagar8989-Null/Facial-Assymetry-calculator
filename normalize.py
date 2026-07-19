import numpy as np

LEFT_EYE = [
    33, 133, 160, 159, 158, 157, 173, 246
]

RIGHT_EYE = [
    362, 263, 387, 386, 385, 384, 398, 466
]


class LandmarkNormalizer:

    def eye_center(self, landmarks, indices):
        return np.mean(landmarks[indices], axis=0)

    def normalize(self, landmarks):

        left_eye = self.eye_center(landmarks, LEFT_EYE)
        right_eye = self.eye_center(landmarks, RIGHT_EYE)

        eye_distance = np.linalg.norm(right_eye - left_eye)

        if eye_distance < 1:
            raise ValueError("Eye distance too small.")

        normalized = landmarks / eye_distance

        return normalized, eye_distance