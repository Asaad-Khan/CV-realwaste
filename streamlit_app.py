import logging
import threading
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.inference import WastePredictor


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "baseline.pt"
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RealWaste Classifier", page_icon="♻️")


@st.cache_resource
def load_predictor(checkpoint_path: str, checkpoint_modified_ns: int):
    """Share one model and inference lock across app sessions."""
    torch.set_num_threads(1)
    predictor = WastePredictor(Path(checkpoint_path))
    return predictor, threading.Lock()


def decode_image(content: bytes) -> Image.Image:
    """Validate actual image content before decoding pixels."""
    if not content:
        raise ValueError("The uploaded file is empty.")

    if len(content) > MAX_BYTES:
        raise ValueError("Please upload an image no larger than 5 MiB.")

    with Image.open(BytesIO(content)) as image:
        if image.format not in ALLOWED_FORMATS:
            raise ValueError("Only JPEG, PNG, and WebP images are supported.")

        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Please use an image below 12 million pixels.")

        if getattr(image, "is_animated", False):
            raise ValueError("Please upload a single, non-animated image.")

        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def main():
    st.title("♻️ RealWaste Classifier")
    st.write(
        "Upload a photo of a waste item to classify it into "
        "one of nine material categories."
    )
    st.caption(
        "Research prototype trained on RealWaste. Predictions may be "
        "incorrect, particularly for unfamiliar backgrounds or mixed waste. "
        "This is not guidance on local recycling rules."
    )

    try:
        predictor, inference_lock = load_predictor(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime_ns,
        )
    except Exception:
        logger.exception("Model initialization failed")
        st.error("The model could not be loaded. Please contact the app owner.")
        st.stop()

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        max_upload_size=5,
        help="Maximum 5 MiB and 12 million pixels.",
    )

    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    if st.button("Classify image", type="primary"):
        # Reject overlapping predictions to limit peak memory usage.
        if not inference_lock.acquire(blocking=False):
            st.warning("The model is busy. Please try again shortly.")
            return

        try:
            with st.spinner("Classifying image..."):
                image = decode_image(uploaded.getvalue())

                started = time.perf_counter()
                result = predictor.predict_image(image, top_k=3)
                inference_ms = (time.perf_counter() - started) * 1000

                # Resize only the display copy; inference uses the original.
                preview = image.copy()
                preview.thumbnail((640, 640))

            st.image(preview, caption="Uploaded image")
            st.subheader(f"Prediction: {result['predicted_class']}")

            st.table(
                [
                    {
                        "Class": item["label"],
                        "Model score": f"{item['score']:.1%}",
                    }
                    for item in result["top_predictions"]
                ]
            )

            st.caption(
                "Scores are softmax outputs, not calibrated probabilities "
                "of correctness."
            )
            st.caption(
                f"Inference: {inference_ms:.1f} ms · "
                f"Model: {result['model_name']} · "
                f"Version: {result['model_version']}"
            )

        except Image.DecompressionBombError:
            st.error("The image is too large to process safely.")
        except (UnidentifiedImageError, OSError):
            st.error("The file could not be read as an image.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            logger.exception("Prediction failed")
            st.error("Prediction failed. Please try another image.")
        finally:
            inference_lock.release()


if __name__ == "__main__":
    main()