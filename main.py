import cv2

from detector import FaceDetector
from align import FaceAligner
from symmetry import SymmetryAnalyzer
from score import SymmetryScore

image = cv2.imread("images/test4.jpeg")
# image = cv2.imread("images/test0.png")

detector = FaceDetector()
landmarks = detector.detect(image)

aligner = FaceAligner()

aligned_image, aligned_landmarks, angle = aligner.align(
    image,
    landmarks
)


analyzer = SymmetryAnalyzer()

results = analyzer.analyze(
    aligned_landmarks
)

scorer = SymmetryScore()

overall, report = scorer.calculate(
    results
)


print()

print("Region Scores")

for region,score in report.items():

    print(f"{region:<12}: {score}")

print()

print("Overall :",overall)