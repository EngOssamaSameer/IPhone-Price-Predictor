from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from urllib.parse import quote_plus


# -------------------------------------------------
# Flask App
# -------------------------------------------------

app = Flask(__name__)


# -------------------------------------------------
# Project Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------------
# Model Files
# -------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "iphone_price_model.pkl"
)

MODEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "model_encoder.pkl"
)

COLOR_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "color_encoder.pkl"
)

WARRANTY_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "warranty_encoder.pkl"
)


# -------------------------------------------------
# Dataset
# -------------------------------------------------

DATASET_PATH = os.path.join(
    BASE_DIR,
    "used_iphone_egypt_images_dataset(1).csv"
)


# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = joblib.load(MODEL_PATH)

le_model = joblib.load(MODEL_ENCODER_PATH)
le_color = joblib.load(COLOR_ENCODER_PATH)
le_warranty = joblib.load(WARRANTY_ENCODER_PATH)


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv(DATASET_PATH)

# Remove unnecessary columns if they exist
df = df.drop(
    columns=["Unnamed: 7"],
    errors="ignore"
)


# -------------------------------------------------
# Create Model Information
# Model -> Storage + Colors
# -------------------------------------------------

model_data = {}

for phone_model in df["model"].dropna().unique():

    model_df = df[
        df["model"] == phone_model
    ]

    storage_values = sorted(
        model_df["storage_gb"]
        .dropna()
        .unique()
        .tolist()
    )

    color_values = sorted(
        model_df["color"]
        .dropna()
        .unique()
        .tolist()
    )

    model_data[phone_model] = {
        "storage": storage_values,
        "colors": color_values
    }


# -------------------------------------------------
# Home Page
# -------------------------------------------------

@app.route("/")
def home():

    models = sorted(
        df["model"]
        .dropna()
        .unique()
        .tolist()
    )

    warranties = sorted(
        df["warranty"]
        .dropna()
        .unique()
        .tolist()
    )

    return render_template(
        "index.html",
        models=models,
        warranties=warranties,
        model_data=model_data
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ---------------------------------------------
    # Get form data
    # ---------------------------------------------

    model_name = request.form["model"]

    ram = float(
        request.form["ram_gb"]
    )

    storage = float(
        request.form["storage_gb"]
    )

    color = request.form["color"]

    battery = float(
        request.form["battery_health_pct"]
    )

    warranty = request.form["warranty"]


    # ---------------------------------------------
    # Encode categorical values
    # ---------------------------------------------

    model_encoded = le_model.transform(
        [model_name]
    )[0]

    color_encoded = le_color.transform(
        [color]
    )[0]

    warranty_encoded = le_warranty.transform(
        [warranty]
    )[0]


    # ---------------------------------------------
    # Create model input
    # IMPORTANT:
    # Same order used during training
    # ---------------------------------------------

    input_data = [[
        model_encoded,
        ram,
        storage,
        color_encoded,
        battery,
        warranty_encoded
    ]]


    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # ---------------------------------------------
    # Create internet image URL
    # ---------------------------------------------

    search_query = f"{model_name} iPhone"

    image_url = (
        "https://tse1.mm.bing.net/th?q="
        + quote_plus(search_query)
    )


    # ---------------------------------------------
    # Result Page
    # ---------------------------------------------

    return render_template(
        "prediction.html",

        prediction=f"{prediction:,.0f}",

        model=model_name,

        ram=int(ram),

        storage=int(storage),

        color=color,

        battery=int(battery),

        warranty=warranty,

        image_url=image_url
    )


# -------------------------------------------------
# Run Application
# -------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )