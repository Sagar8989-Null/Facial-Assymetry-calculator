import cv2

from detector import FaceDetector
from align import FaceAligner
from normalize import LandmarkNormalizer
from symmetry import SymmetryAnalyzer
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

normalized_landmarks = normalizer.normalize(
    aligned_landmarks
)


analyzer = SymmetryAnalyzer()

results = analyzer.analyze(
    normalized_landmarks
)


scorer = SymmetryScore()

overall_score = scorer.calculate(
    results["regions"]
)


print("Region Errors")

for name, value in results["regions"].items():

    print(f"{name:<12}: {value:.5f}")

print()

print("Overall Error :", results["overall_error"])

print("Score :", overall_score)