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
You are a nutrition assistant.

Do NOT copy the food label text verbatim.

Instead, summarize the label and provide:

1. Product name
2. Main ingredients
3. Nutrition summary
4. Health score from 0 to 100
5. Health warnings
6. Healthier alternatives

Respond in plain text.
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash",
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