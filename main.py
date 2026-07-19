import cv2

from detector import FaceDetector
from align import FaceAligner
from normalize import LandmarkNormalizer
from symmetry import SymmetryAnalyzer
from score import SymmetryScore
from landmark_heatmap import LandmarkHeatmap
from visualize import HeatmapVisualizer
from constants import HEATMAP_REGIONS

image = cv2.imread("images/test1.jpeg")
# image = cv2.imread("images/test0.png")

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

heatmap_generator = LandmarkHeatmap()

heatmap = heatmap_generator.generate(
    aligned_image,
    aligned_landmarks,
    results["landmark_errors"]
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

while True:
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC key
        break
    if cv2.getWindowProperty("Heatmap", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()