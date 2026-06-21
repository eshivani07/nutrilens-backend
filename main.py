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

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        prompt = """
        Analyze this food product label.

        Return:
        - Product Name
        - Ingredients
        - Nutrition Information
        - Health Score (0-100)
        - Warnings
        - Healthier Alternatives

        Keep the response concise.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                image,
                prompt
            ]
        )

        return {
            "status": "success",
            "result": response.text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }