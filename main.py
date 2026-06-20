from fastapi import FastAPI, File, UploadFile, HTTPException
import easyocr
import cv2
import numpy as np
from PIL import Image
import io

# 1. Initialize the FastAPI application
app = FastAPI()

# 2. Initialize the EasyOCR reader globally (using English 'en')
# Loading this outside the endpoint ensures it loads into memory ONCE at server startup,
# preventing high latency and server crashes on every scan.
try:
    print("Initializing EasyOCR Reader...")
    reader = easyocr.Reader(['en'], gpu=False) # gpu=False ensures it runs safely on Render's CPU tiers
    print("EasyOCR initialized successfully!")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Downscales and converts the uploaded image to grayscale.
    This saves massive memory on your cloud server and dramatically increases OCR accuracy.
    """
    # Open the image from the uploaded bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize the image if it is too massive (max width/height of 1024px is perfect for scanning text)
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    img_np = np.array(image)
    
    # Convert color space from RGB (PIL) to BGR (OpenCV standard)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for better text separation
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return gray

# --- Your Existing Base Routes ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the NutriLens Backend Api!"}

@app.get("/test")
def read_test():
    return {"status": "Backend is online and working perfectly!"}


# --- Your Updated Core Route ---
@app.post("/analyze")
async def analyze_food_package(file: UploadFile = File(...)):
    """
    Accepts an image file from the Flutter app, processes it to save memory,
    extracts the raw text from the food label, and returns it.
    """
    try:
        # Read the raw image bytes uploaded from your Flutter device's camera
        contents = await file.read()
        
        # Preprocess the image to optimize RAM usage and improve text clarity
        processed_img = preprocess_image(contents)
        
        print("Starting text extraction via EasyOCR...")
        # detail=0 tells EasyOCR to return clean strings directly, skipping heavy coordinate boxes
        extracted_text_list = reader.readtext(processed_img, detail=0)
        
        # Combine all separate text line fragments into one unified string block
        full_text = " ".join(extracted_text_list)
        print(f"Extraction complete! Extracted text length: {len(full_text)}")
        
        # Guard clause: If the image is blurry or has no text, alert the app
        if not full_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Could not read any text. Please try again with a clearer image of the ingredients label."
            )
            
        # Returning the real text back to your Flutter app.
        # This allows us to verify your OCR pipeline works perfectly before we drop in the Gemini AI!
        return {
            "status": "success",
            "extracted_text": full_text,
            "health_score": 0,  # Keeping these keys temporarily so your Flutter UI doesn't break
            "warnings": ["OCR phase successful! Ready for Gemini integration."],
            "alternatives": []
        }
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")