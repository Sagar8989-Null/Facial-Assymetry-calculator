import cv2

from detector import FaceDetector
from align import FaceAligner
from normalize import LandmarkNormalizer
from symmetry import SymmetryAnalyzer
from score import SymmetryScore
from heatmap import FacialHeatmap
from visualize import HeatmapVisualizer
from constants import HEATMAP_REGIONS

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

overall, report = scorer.calculate(
    results
)

# ---------------------------------------
# Heatmap
# ---------------------------------------

heatmap_generator = FacialHeatmap()

heatmap = heatmap_generator.generate(
    aligned_image,
    aligned_landmarks,
    results["distance_errors"],
    HEATMAP_REGIONS
)

visualizer = HeatmapVisualizer()

overlay = visualizer.save(
    aligned_image,
    heatmap
)

print()

print("Region Scores")

for region,score in report.items():

    print(f"{region:<12}: {score}")

print()

print("Overall :",overall)

cv2.imshow("Heatmap", heatmap)

cv2.imshow("Overlay", overlay)

cv2.waitKey(0)

cv2.destroyAllWindows()