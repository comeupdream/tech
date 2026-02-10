import io
import random
import uuid

from flask import Flask, render_template, request, send_file
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit


def strip_metadata(img: Image.Image) -> Image.Image:
    """Re-create the image from raw pixel data, dropping all EXIF/IPTC/XMP/ICC."""
    clean = Image.new(img.mode, img.size)
    clean.putdata(list(img.getdata(band=None)))
    return clean


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


def scramble(img: Image.Image) -> Image.Image:
    """Full pipeline: strip metadata → color shift → edge noise → pixel scatter."""
    if img.mode != "RGB":
        img = img.convert("RGB")
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

    img = Image.open(f.stream)
    result = scramble(img)

    # Save with randomized JPEG quality to further vary the file
    quality = random.randint(88, 95)
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=quality)
    buf.seek(0)

    filename = f"scrambled_{uuid.uuid4().hex[:8]}.jpg"
    return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
