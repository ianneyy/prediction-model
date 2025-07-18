from fastapi import FastAPI, Query
import fitz
import requests
from io import BytesIO

app = FastAPI()

@app.get("/extract-from-url")
def extract_from_url(file_url: str = Query(...)):
    response = requests.get(file_url)

    if response.status_code != 200:
        return {"error": "Failed to download PDF"}

    file_bytes = BytesIO(response.content)
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    text = ""
    for page in pdf:
        text += page.get_text()

    return {"text": text}
