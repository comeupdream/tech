import io
import random
import uuid

import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from flask import Flask, render_template, request, send_file
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def strip_metadata(img: Image.Image) -> Image.Image:
    """Re-create the image from raw pixel data, dropping all EXIF/IPTC/XMP/ICC."""
    return Image.fromarray(np.array(img))


def shift_colors(img: Image.Image, pct: float = 0.015) -> Image.Image:
    """Randomly shift each channel by up to ±pct (default 1.5%)."""
    pixels = np.array(img, dtype=np.float64)
    for ch in range(pixels.shape[2]):
        factor = 1.0 + random.uniform(-pct, pct)
        pixels[:, :, ch] = np.clip(pixels[:, :, ch] * factor, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def add_edge_noise(img: Image.Image, border: int = 6, strength: int = 8) -> Image.Image:
    """Add subtle random noise to a border region around the image edges."""
    pixels = np.array(img, dtype=np.int16)
    h, w = pixels.shape[:2]

    # Create a mask for edge pixels
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True   # top
    mask[-border:, :] = True  # bottom
    mask[:, :border] = True   # left
    mask[:, -border:] = True  # right

    noise = np.random.randint(-strength, strength + 1, size=pixels.shape, dtype=np.int16)
    pixels[mask] = np.clip(pixels[mask] + noise[mask], 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def scatter_pixels(img: Image.Image, count: int = 30) -> Image.Image:
    """Randomly nudge a handful of interior pixels by ±1 to break perceptual hashes."""
    pixels = np.array(img, dtype=np.int16)
    h, w = pixels.shape[:2]
    for _ in range(count):
        y = random.randint(1, h - 2)
        x = random.randint(1, w - 2)
        ch = random.randint(0, pixels.shape[2] - 1)
        pixels[y, x, ch] = np.clip(pixels[y, x, ch] + random.choice([-1, 1]), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def _measure_asymmetry(face_crop: np.ndarray) -> float:
    """Return 0-1 score of how asymmetric a face crop is (0 = perfectly symmetric)."""
    h, w = face_crop.shape[:2]
    mid = w // 2
    # Compare left side with mirror of right side
    left = face_crop[:, :mid].astype(np.float32)
    right = np.flip(face_crop[:, -mid:], axis=1).astype(np.float32)
    # Trim to same size in case of odd width
    min_w = min(left.shape[1], right.shape[1])
    left = left[:, :min_w]
    right = right[:, :min_w]
    diff = np.mean(np.abs(left - right)) / 255.0
    return float(diff)


def symmetrize_face(img: Image.Image, max_strength: float = 0.12) -> Image.Image:
    """Subtly improve facial symmetry by blending with a midline-flipped version.

    - Detects faces via mediapipe FaceLandmarker
    - Measures asymmetry; skips correction if already symmetric
    - Blends at 10-15% toward the symmetric version (never more)
    - Only touches the face region, leaves background untouched
    """
    arr = np.array(img)
    h, w = arr.shape[:2]

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return img  # no face detected, return as-is

    landmarks = result.face_landmarks[0]

    # Get all landmark pixel coords
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

    # Compute face bounding box from landmarks with padding
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad_x = int((x_max - x_min) * 0.1)
    pad_y = int((y_max - y_min) * 0.1)
    x1 = max(0, int(x_min) - pad_x)
    y1 = max(0, int(y_min) - pad_y)
    x2 = min(w, int(x_max) + pad_x)
    y2 = min(h, int(y_max) + pad_y)

    face_crop = arr[y1:y2, x1:x2].copy()

    # Compute midline from nose landmarks (indices 6=nose tip, 10=forehead, 152=chin)
    nose_tip = pts[6]
    forehead = pts[10]
    chin = pts[152]
    # Midline x relative to crop
    mid_x = ((nose_tip[0] + forehead[0] + chin[0]) / 3.0) - x1

    # Measure asymmetry
    asym = _measure_asymmetry(face_crop)

    # Threshold: if asymmetry is below 3%, face is already quite symmetric
    if asym < 0.03:
        return img

    # Scale strength by how asymmetric the face is (more asymmetric = stronger, capped)
    strength = min(max_strength, asym * 1.5)

    # Create the symmetric blend:
    # Flip the face crop around the midline
    crop_h, crop_w = face_crop.shape[:2]
    mid_col = int(round(mid_x))
    mid_col = max(1, min(crop_w - 1, mid_col))

    flipped_crop = face_crop.copy()

    # Left side: pixels from [0, mid_col]
    left_half = face_crop[:, :mid_col]
    right_half = face_crop[:, mid_col:]

    # Build symmetric version by averaging each side with its mirror
    left_w = left_half.shape[1]
    right_w = right_half.shape[1]

    # Mirror left onto right
    if left_w > 0 and right_w > 0:
        # Left mirrored
        left_mirrored = np.flip(left_half, axis=1)
        # Right mirrored
        right_mirrored = np.flip(right_half, axis=1)

        # Trim to matching widths
        overlap_r = min(left_mirrored.shape[1], right_half.shape[1])
        overlap_l = min(right_mirrored.shape[1], left_half.shape[1])

        # Blend right side with mirrored-left at strength %
        sym_right = right_half.copy().astype(np.float64)
        sym_right[:, :overlap_r] = (
            sym_right[:, :overlap_r] * (1 - strength)
            + left_mirrored[:, :overlap_r].astype(np.float64) * strength
        )
        flipped_crop[:, mid_col:mid_col + sym_right.shape[1]] = np.clip(
            sym_right, 0, 255
        ).astype(np.uint8)

        # Blend left side with mirrored-right at strength %
        sym_left = left_half.copy().astype(np.float64)
        sym_left[:, -overlap_l:] = (
            sym_left[:, -overlap_l:] * (1 - strength)
            + right_mirrored[:, -overlap_l:].astype(np.float64) * strength
        )
        flipped_crop[:, :mid_col] = np.clip(sym_left, 0, 255).astype(np.uint8)

    # Feathered mask so the edit blends smoothly into the background
    mask = np.zeros((crop_h, crop_w), dtype=np.float32)
    feather = min(pad_x, pad_y, 20)
    mask[feather:-feather, feather:-feather] = 1.0
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather / 2)
    mask = mask[:, :, np.newaxis]

    # Composite
    blended = (
        face_crop.astype(np.float64) * (1 - mask)
        + flipped_crop.astype(np.float64) * mask
    )
    arr[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def scramble(img: Image.Image) -> Image.Image:
    """Full pipeline: symmetry → strip metadata → color shift → edge noise → pixel scatter."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = symmetrize_face(img)
    img = strip_metadata(img)
    img = shift_colors(img)
    img = add_edge_noise(img)
    img = scatter_pixels(img)
    return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "photo" not in request.files:
        return "No file uploaded", 400
    f = request.files["photo"]
    if not f.filename:
        return "Empty filename", 400

    try:
        img = Image.open(f.stream)
        result = scramble(img)

        # Save with randomized JPEG quality to further vary the file
        quality = random.randint(88, 95)
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=quality)
        buf.seek(0)

        filename = f"scrambled_{uuid.uuid4().hex[:8]}.jpg"
        return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name=filename)
    except Exception as e:
        app.logger.error("Scramble failed: %s", e, exc_info=True)
        return f"Processing failed: {e}", 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
