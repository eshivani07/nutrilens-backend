from fastapi import FastAPI, File, UploadFile, HTTPException
import easyocr
import cv2
import numpy as np
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# OCR Reader (lazy loading)
reader = None


def get_reader():
    global reader

    if reader is None:
        print("Loading EasyOCR model...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR loaded successfully!")

    return reader


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Resize and convert image to grayscale
    for better OCR performance.
    """

    image = Image.open(io.BytesIO(image_bytes))

    # Convert RGBA → RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize large images
    max_size = 1024

    if max(image.size) > max_size:
        image.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS
        )

    img_np = np.array(image)

    img_cv = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        img_cv,
        cv2.COLOR_BGR2GRAY
    )

    return gray


# --------------------
# Basic Routes
# --------------------

@app.get("/")
def home():
    return {
        "message": "NutriLens Backend Running"
    }


@app.get("/test")
def test():
    return {
        "status": "success"
    }


# --------------------
# OCR Analysis Route
# --------------------

@app.post("/analyze")
async def analyze_food_package(
    file: UploadFile = File(...)
):

    try:

        print("Receiving image...")

        contents = await file.read()

        processed_img = preprocess_image(
            contents
        )

        print("Loading OCR Reader...")

        ocr_reader = get_reader()

        print("Starting OCR Extraction...")

        extracted_text_list = ocr_reader.readtext(
            processed_img,
            detail=0
        )

        full_text = " ".join(
            extracted_text_list
        )

        print("OCR Complete")

        if not full_text.strip():

            raise HTTPException(
                status_code=400,
                detail="No text detected. Please capture a clearer image."
            )

        return {
            "status": "success",
            "extracted_text": full_text,
            "health_score": 0,
            "warnings": [
                "OCR successful. Gemini integration pending."
            ],
            "alternatives": []
        }

    except HTTPException:
        raise

    except Exception as e:

        print(f"Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Backend Error: {str(e)}"
        )