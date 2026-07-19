import cv2
import numpy as np

LEFT_EYE = [
    33,133,160,159,158,157,173,246
]

RIGHT_EYE = [
    362,263,387,386,385,384,398,466


]

class FaceAligner:

    def eye_center(self, landmarks, indices):

        pts = landmarks[indices]

        return np.mean(pts, axis=0)

    def align(self, image, landmarks):

        left_eye = self.eye_center(landmarks, LEFT_EYE)
        right_eye = self.eye_center(landmarks, RIGHT_EYE)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        angle = np.degrees(np.arctan2(dy, dx))

        center = (
            image.shape[1] // 2,
            image.shape[0] // 2
        )

        rotation = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        aligned_image = cv2.warpAffine(
            image,
            rotation,
            (
                image.shape[1],
                image.shape[0]
            )
        )

        ones = np.ones((landmarks.shape[0],1))

        points = np.hstack([landmarks,ones])

        aligned_landmarks = rotation.dot(points.T).T

        return aligned_image, aligned_landmarks, angle