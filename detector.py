import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect(self, image):

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        h, w = image.shape[:2]

        landmarks = []

        for landmark in face.landmark:

            x = landmark.x * w
            y = landmark.y * h

            landmarks.append((x, y))

        return np.array(landmarks)