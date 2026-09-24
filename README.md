# Used iPhone Price Predictor

An end-to-end Machine Learning and Flask web application for estimating
used-iPhone prices in the Egyptian market.

## Project Overview

This project combines data preprocessing, Machine Learning, model
persistence, and web development into one complete application.

The workflow is:

**Dataset → Cleaning → Encoding → Train/Test Split → Random Forest →
Evaluation → Saved Model → Flask Web App → Prediction**

The application allows a user to enter iPhone specifications and receive
an estimated price in Egyptian Pounds (EGP).

## Project Goals

-   Build a real-world regression Machine Learning project.
-   Prepare and clean used-iPhone pricing data.
-   Encode categorical features.
-   Train and evaluate a regression model.
-   Save the trained model and encoders.
-   Integrate the model with Flask.
-   Build a user-friendly prediction interface.
-   Prepare the application for cloud deployment.

## Main Features

### Machine Learning

-   Random Forest Regression
-   Missing-value handling
-   Label Encoding
-   Train/Test split
-   MAE, RMSE and R² evaluation
-   Model persistence with Joblib

### Web Application

-   Flask backend
-   HTML/Jinja templates
-   iPhone model selection
-   RAM selection
-   Storage selection
-   Color selection
-   Battery-health input
-   Warranty selection
-   Dynamic model/storage/color options
-   Predicted price display
-   Prediction result page
-   iPhone image display

### Development & Deployment

-   Python
-   Git
-   GitHub
-   GitHub Desktop
-   Git LFS
-   Gunicorn
-   Cloud deployment preparation

------------------------------------------------------------------------

# Dataset

Dataset file:

`used_iphone_egypt_images_dataset(1).csv`

The dataset contains approximately **26,424 rows and 8 columns**.

## Dataset Fields

  Column                 Description
  ---------------------- ---------------------------------
  `model`                iPhone model
  `ram_gb`               RAM capacity in GB
  `storage_gb`           Storage capacity in GB
  `color`                iPhone color
  `battery_health_pct`   Battery health percentage
  `price_egp`            Target price in Egyptian Pounds
  `warranty`             Warranty status
  `Unnamed: 7`           Extra unused column

Target:

`price_egp`

------------------------------------------------------------------------

# Data Preprocessing

The dataset was inspected for missing values.

There were **7 missing values** in `battery_health_pct`.

They were replaced using the column median:

``` python
df["battery_health_pct"] = df["battery_health_pct"].fillna(
    df["battery_health_pct"].median()
)
```

The unused `Unnamed: 7` column was removed from the application data.

------------------------------------------------------------------------

# Categorical Encoding

The categorical variables:

-   `model`
-   `color`
-   `warranty`

were converted to numerical values using separate `LabelEncoder`
objects.

``` python
from sklearn.preprocessing import LabelEncoder

le_model = LabelEncoder()
le_color = LabelEncoder()
le_warranty = LabelEncoder()

df["model"] = le_model.fit_transform(df["model"])
df["color"] = le_color.fit_transform(df["color"])
df["warranty"] = le_warranty.fit_transform(df["warranty"])
```

The encoders were saved so the Flask application can apply the same
mappings during prediction.

------------------------------------------------------------------------

# Machine Learning

## Features and Target

``` python
X = df.drop("price_egp", axis=1)
y = df["price_egp"]
```

## Train/Test Split

The data was split into:

-   **80% training**
-   **20% testing**

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Model

The project uses a **Random Forest Regressor**:

``` python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
```

------------------------------------------------------------------------

# Model Evaluation

The trained model was evaluated on the test set.

  Metric            Result
  -------- ---------------
  MAE        1220.0985 EGP
  RMSE       1820.2767 EGP
  R²             0.9939105

### Interpretation

-   **MAE ≈ 1,220 EGP:** average absolute prediction error on the test
    split.
-   **RMSE ≈ 1,820 EGP:** gives more weight to larger errors.
-   **R² ≈ 0.9939:** approximately 99.39% of the target variance is
    explained on this test split.

> These results are specific to the available dataset and test split.
> They do not guarantee the same performance on future market data.

------------------------------------------------------------------------

# Saved Model Artifacts

The trained model and encoders were saved using Joblib:

``` text
iphone_price_model.pkl
model_encoder.pkl
color_encoder.pkl
warranty_encoder.pkl
```

The Flask application loads these files at startup.

------------------------------------------------------------------------

# Flask Web Application

Main backend:

`app.py`

Templates:

``` text
templates/
├── index.html
└── prediction.html
```

## Home / Prediction Form

`index.html` collects:

-   iPhone model
-   RAM
-   Storage
-   Color
-   Battery health
-   Warranty

## Prediction Result

`prediction.html` displays:

-   Selected model
-   RAM
-   Storage
-   Color
-   Battery health
-   Warranty
-   Estimated price
-   iPhone image

------------------------------------------------------------------------

# Application Workflow

``` text
User
 ↓
Prediction Form
 ↓
Select iPhone Model
 ↓
Select RAM / Storage / Color
 ↓
Enter Battery Health
 ↓
Select Warranty
 ↓
Flask receives form data
 ↓
Categorical Encoding
 ↓
Random Forest Prediction
 ↓
Prediction Result
 ↓
Estimated Price in EGP
```

------------------------------------------------------------------------

# Dynamic Form Logic

The application uses the original dataset to determine which storage
options and colors are available for each iPhone model.

Conceptually:

``` python
model_data = {}

for phone_model in df["model"].dropna().unique():
    model_df = df[df["model"] == phone_model]

    storage_values = sorted(
        model_df["storage_gb"].dropna().unique().tolist()
    )

    color_values = sorted(
        model_df["color"].dropna().unique().tolist()
    )

    model_data[phone_model] = {
        "storage": storage_values,
        "colors": color_values
    }
```

This information is passed to the frontend so the available choices can
change according to the selected model.

------------------------------------------------------------------------

# iPhone Image Feature

The prediction page displays an image related to the selected iPhone
model.

The prototype generates a search query based on the selected model:

``` python
search_query = f"{model_name} iPhone"
```

The query is URL-encoded and used to build the image URL.

This is a prototype implementation; it does not guarantee an official
image, exact color, storage variant, or physical condition. A production
version could use a controlled local image dataset or an official
product-image source.

------------------------------------------------------------------------

# Project Structure

``` text
IPhone-Price-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
│
├── iphone_price_model.pkl
├── model_encoder.pkl
├── color_encoder.pkl
├── warranty_encoder.pkl
│
├── used_iphone_egypt_images_dataset(1).csv
│
└── templates/
    ├── index.html
    └── prediction.html
```

------------------------------------------------------------------------

# Technologies Used

## Programming

-   Python 3

## Data

-   Pandas
-   NumPy

## Machine Learning

-   Scikit-learn
-   Random Forest Regression
-   LabelEncoder
-   Joblib

## Web

-   Flask
-   Jinja2
-   HTML
-   CSS

## Version Control / Deployment

-   Git
-   GitHub
-   GitHub Desktop
-   Git LFS
-   Gunicorn

------------------------------------------------------------------------

# Requirements

The project uses:

``` text
Flask
pandas
scikit-learn
joblib
gunicorn
```

They are listed in:

`requirements.txt`

------------------------------------------------------------------------

# Local Installation

Clone the repository:

``` bash
git clone https://github.com/EngOssamaSameer/IPhone-Price-Predictor.git
```

Enter the project directory:

``` bash
cd IPhone-Price-Predictor
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the application:

``` bash
python app.py
```

Open:

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

# GitHub

Repository:

https://github.com/EngOssamaSameer/IPhone-Price-Predictor

GitHub Desktop was used to manage the repository, commits, and pushes.

------------------------------------------------------------------------

# Git LFS

The trained `.pkl` files include large binary model artifacts.

The project therefore uses Git Large File Storage (Git LFS).

`.gitattributes` contains:

``` text
*.pkl filter=lfs diff=lfs merge=lfs -text
```

Tracked LFS files:

``` text
color_encoder.pkl
iphone_price_model.pkl
model_encoder.pkl
warranty_encoder.pkl
```

Git LFS allows these large model files to be versioned without storing
their full binary contents directly in normal Git objects.

------------------------------------------------------------------------

# Deployment

The application was prepared for cloud deployment as a Flask/Python web
service.

Production-style Flask serving can use:

``` bash
gunicorn app:app
```

The deployment workflow is:

``` text
GitHub Repository
 ↓
Cloud Deployment
 ↓
Python Environment
 ↓
pip install -r requirements.txt
 ↓
Gunicorn
 ↓
Flask app.py
 ↓
Load Model + Encoders
 ↓
Web Application
```

------------------------------------------------------------------------

# Example Prediction

A user provides:

``` text
Model
RAM
Storage
Color
Battery Health
Warranty
```

The application converts the categorical values into the numerical
representation expected by the trained model and returns an estimated
price:

``` text
Estimated Price: XXXX EGP
```

The actual value depends on the selected inputs and the trained model.

------------------------------------------------------------------------

# Limitations

The model is trained on the available project dataset, so its
predictions can be affected by:

-   Dataset quality
-   Dataset coverage
-   Changes in used-phone prices
-   Supply and demand
-   Currency changes
-   Physical condition
-   Repair history
-   Original/replacement components
-   Accessories
-   Location
-   Warranty differences

The current feature set does not fully represent every
physical-condition factor.

The reported test metrics should not be treated as a guarantee of future
real-world accuracy.

------------------------------------------------------------------------

# Future Improvements

## Machine Learning

-   Hyperparameter tuning
-   Cross-validation
-   XGBoost comparison
-   Gradient Boosting
-   Extra Trees
-   Feature importance
-   SHAP explainability
-   Model monitoring

## Dataset

-   More recent market data
-   Phone physical condition
-   Screen condition
-   Repair history
-   Original/replacement parts
-   Accessories
-   Seller type
-   Location
-   More brands

## Web Application

-   Better UI/UX
-   Price range instead of a single estimate
-   Confidence/uncertainty information
-   Interactive charts
-   Price history
-   Search history
-   Authentication
-   Database integration

## Deployment

-   CI/CD
-   Automated model updates
-   Dataset update pipeline
-   Application monitoring
-   Production logging

------------------------------------------------------------------------

# Security Considerations

For a production deployment:

-   Validate user input.
-   Keep secrets in environment variables.
-   Disable Flask debug mode.
-   Use HTTPS.
-   Keep dependencies updated.
-   Avoid exposing sensitive files.
-   Add proper application logging.

------------------------------------------------------------------------

# 📚 Learning Outcomes

This project demonstrates an end-to-end Machine Learning workflow:

-   Dataset handling
-   Data cleaning
-   Missing-value treatment
-   Categorical encoding
-   Feature/target separation
-   Train/test splitting
-   Regression modeling
-   Model evaluation
-   Model persistence
-   Flask integration
-   HTML/Jinja templates
-   Dynamic forms
-   Git/GitHub
-   Git LFS
-   Gunicorn
-   Cloud deployment preparation

------------------------------------------------------------------------

# Project Status

``` text
✅ Dataset prepared
✅ Data preprocessing completed
✅ Missing values handled
✅ Categorical encoding completed
✅ Random Forest model trained
✅ Model evaluated
✅ Model and encoders saved
✅ Flask application developed
✅ Prediction page implemented
✅ Dynamic model/storage/color logic implemented
✅ Git repository created
✅ Git LFS configured
✅ GitHub repository prepared
Deployment prepared
```

------------------------------------------------------------------------

# Disclaimer

This application provides an estimated used-iPhone price based on
Machine Learning and historical/project data.

The prediction is an estimate, not a guaranteed selling price.

Actual market prices may differ based on device condition, market
demand, seller, location, warranty, accessories, repairs, and current
market conditions.

------------------------------------------------------------------------

# Author

## Eng. Ossama Samir

**Computer Science & Artificial Intelligence**

Machine Learning • Data Analysis • Python • Flask

------------------------------------------------------------------------

# Repository

GitHub:

https://github.com/EngOssamaSameer/IPhone-Price-Predictor

If you find this project useful, feel free to explore the repository and
⭐ the project.
