from fastapi import FastAPI

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