import cv2

from detector import FaceDetector
from align import FaceAligner
from normalize import LandmarkNormalizer
from midline import MidlineEstimator
from symmetry import SymmetryCalculator
from score import SymmetryScore

# image = cv2.imread("images/test.jpg")
image = cv2.imread("images/test0.png")

detector = FaceDetector()
landmarks = detector.detect(image)

aligner = FaceAligner()

aligned_image, aligned_landmarks, angle = aligner.align(
    image,
    landmarks
)

normalizer = LandmarkNormalizer()

normalized_landmarks, eye_distance = normalizer.normalize(
    aligned_landmarks
)

midline = MidlineEstimator()

mid_x = midline.estimate(normalized_landmarks)

calculator = SymmetryCalculator()

results = calculator.calculate(
    normalized_landmarks,
    mid_x
)

# mean_error, distances = calculator.calculate(
#     normalized_landmarks,
#     mid_x
# )

scorer = SymmetryScore()

overall = scorer.calculate(results)


# score = scorer.calculate(mean_error)

print(f"Eye Distance : {eye_distance:.2f}")

print(normalized_landmarks[:5])

print("Facial Midline:", mid_x)

# print(distances)
# 
# print(mean_error)
# 
# print(score)


print()

for region,error in results.items():
    print(region,error)

print()

print("Overall :",overall)