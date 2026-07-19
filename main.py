import cv2

from detector import FaceDetector
from align import FaceAligner
# from normalize import LandmarkNormalizer
from symmetry import SymmetryAnalyzer
from score import SymmetryScore
from constants import HEATMAP_POLYGONS
from heatmap import AsymmetryHeatmap

image = cv2.imread("images/test1.jpeg")
# image = cv2.imread("images/test0.png")

detector = FaceDetector()
landmarks = detector.detect(image)

aligner = FaceAligner()

aligned_image, aligned_landmarks, angle = aligner.align(
    image,
    landmarks
)

# normalizer = LandmarkNormalizer()

# normalized_landmarks = normalizer.normalize(
#     aligned_landmarks
# )


analyzer = SymmetryAnalyzer()

# results = analyzer.analyze(
#     normalized_landmarks
# )

# pixel_results = analyzer.analyze(
#     aligned_landmarks
# )


results = analyzer.analyze(
    aligned_landmarks,
    normalize=True
)

pixel_results = analyzer.analyze(
    aligned_landmarks,
    normalize=False
)

scorer = SymmetryScore()

overall, report = scorer.calculate(
    results
)

heatmap = AsymmetryHeatmap()

heat, overlay = heatmap.generate(
    aligned_image,
    pixel_results["pair_errors"]
)

print()

print("Region Scores")

for region,score in report.items():

    print(f"{region:<12}: {score}")

print()

print("Overall :",overall)


cv2.imwrite(
    "output/heatmap.png",
    heat
)

cv2.imwrite(
    "output/overlay.png",
    overlay
)

cv2.imshow("Heatmap", heat)

cv2.imshow("Overlay", overlay)

while True:
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC key
        break
    if cv2.getWindowProperty("Heatmap", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()