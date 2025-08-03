from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import uvicorn
import os
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates_dir = os.path.join(BASE_DIR, "MLOPS Project", "templates")
templates = Jinja2Templates(directory=templates_dir)

model_path = os.path.join(BASE_DIR, "MLOPS Project", "saved_models", "flight_model.pkl")

try:
    print("Loading model from:", model_path)
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define unique categories
from_options = ['Recife (PE)', 'Florianopolis (SC)', 'Brasilia (DF)', 'Aracaju (SE)',
                'Salvador (BH)', 'Campo Grande (MS)', 'Sao Paulo (SP)', 'Natal (RN)', 'Rio de Janeiro (RJ)']

to_options = ['Florianopolis (SC)', 'Recife (PE)', 'Brasilia (DF)', 'Salvador (BH)',
              'Aracaju (SE)', 'Campo Grande (MS)', 'Sao Paulo (SP)', 'Natal (RN)', 'Rio de Janeiro (RJ)']

flightType_options = ['firstClass', 'economic', 'premium']

# Create LabelEncoders and fit on options
from_encoder = LabelEncoder()
from_encoder.fit(from_options)

to_encoder = LabelEncoder()
to_encoder.fit(to_options)

flightType_encoder = LabelEncoder()
flightType_encoder.fit(flightType_options)


try:
    print("Loading model from:", model_path)
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "from_options": from_options,
        "to_options": to_options,
        "flightType_options": flightType_options,
        "result": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    day: int = Form(...),
    month: int = Form(...),
    year: int = Form(...),
    flightType: str = Form(...),
    to: str = Form(...),
    from_: str = Form(..., alias="from_"),
    distance: float = Form(...)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Prediction unavailable.")
    try:
        # Encode categorical string inputs to integer labels
        flightType_enc = flightType_encoder.transform([flightType])[0]
        to_enc = to_encoder.transform([to])[0]
        from_enc = from_encoder.transform([from_])[0]

        input_data = np.array([[day, month, year, flightType_enc, to_enc, from_enc, distance]])
        prediction = model.predict(input_data)[0]
    except Exception as e:
        print("Error during prediction:", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "from_options": from_options,
        "to_options": to_options,
        "flightType_options": flightType_options,
        "result": f"Predicted Price: ₹{prediction:.2f}"
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
