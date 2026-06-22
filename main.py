from fastapi import FastAPI, UploadFile, File
from google import genai
from pydantic import BaseModel
from PIL import Image
import io
import os

app = FastAPI()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

class ProductAnalysis(BaseModel):
    product_name: str
    health_score: int
    nutrition_summary: str
    warnings: list[str]
    alternatives: list[str]

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                image,
                """
                Analyze this food product.

                Extract:
                - product name
                - health score (0-100)
                - nutrition summary
                - health warnings
                - healthier alternatives
                """
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ProductAnalysis,
            }
        )

        return {
            "status": "success",
            "data": response.parsed.model_dump()
        }

    except Exception as e:

        print("ERROR:")
        print(str(e))

        return {
            "status": "error",
            "message": str(e)
        }