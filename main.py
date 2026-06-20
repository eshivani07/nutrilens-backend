from fastapi import FastAPI, UploadFile, File

app = FastAPI()

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
async def analyze(image: UploadFile = File(...)):

    filename = image.filename

    return {
        "status": "success",
        "filename": filename,
        "health_score": 80,
        "warnings": [
            "High Sugar"
        ],
        "alternatives": [
            "Low Sugar Option"
        ]
    }