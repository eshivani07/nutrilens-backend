from fastapi import FastAPI, UploadFile, File
from google import genai
from PIL import Image
import io
import os

app = FastAPI()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        print("Analyze endpoint hit")

        contents = await file.read()

        print(f"Image size: {len(contents)} bytes")

        image = Image.open(io.BytesIO(contents))

        prompt = """
        Analyze this food label.

        Return:
        - Product Name
        - Ingredients
        - Nutrition Information
        - Health Score
        - Warnings
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                image
            ]
        )

        print("Gemini response:")
        print(response)

        print("Response text:")
        print(response.text)

        return {
            "status": "success",
            "result": response.text
        }

    except Exception as e:
        print("ERROR:")
        print(str(e))

        return {
            "status": "error",
            "message": str(e)
        }