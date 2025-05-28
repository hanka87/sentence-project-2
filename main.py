from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from similarity import TextSimilarity

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

def get_score(text1, text2):
    similarity = TextSimilarity()
    return similarity.calculate_similarity(text1, text2)

@app.post("/")
async def calculate_similarity_form(request: Request, text1: str = Form(...), text2: str = Form(...)):
    score = get_score(text1, text2)
    result = f"Similarity Score: {score:.2f}"
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "text1": text1, "text2": text2})