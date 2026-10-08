from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

CSV_FILE = "parking.csv"


# =========================================================
# PARKING INFORMATION
# =========================================================

PARKING_INFO = {
    "P1": {
        "name": "P1 - Car Parking",
        "capacity": 25,
        "type": "CAR"
    },

    "P2": {
        "name": "P2 - Car Parking",
        "capacity": 30,
        "type": "CAR"
    },

    "P3": {
        "name": "P3 - Bike Parking",
        "capacity": 35,
        "type": "BIKE"
    }
}


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_FILE)

df.columns = df.columns.str.strip()

print("CSV loaded successfully")
print("Rows:", len(df))


# =========================================================
# CLEAN DATA
# =========================================================

df["PARKING ID"] = (
    df["PARKING ID"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["PARKING TYPE"] = (
    df["PARKING TYPE"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["OCCUPIED"] = pd.to_numeric(
    df["OCCUPIED"],
    errors="coerce"
)

df["CAPACITY"] = pd.to_numeric(
    df["CAPACITY"],
    errors="coerce"
)


# =========================================================
# DATETIME
# =========================================================

df["DATETIME"] = pd.to_datetime(
    df["DATE"].astype(str)
    + " "
    + df["TIME"].astype(str),
    dayfirst=True,
    errors="coerce"
)


# =========================================================
# REMOVE INVALID DATA
# =========================================================

df = df.dropna(
    subset=[
        "DATETIME",
        "OCCUPIED",
        "CAPACITY",
        "PARKING ID",
        "PARKING TYPE"
    ]
)


# =========================================================
# SORT
# =========================================================

df = df.sort_values(
    ["PARKING ID", "DATETIME"]
).reset_index(drop=True)


# =========================================================
# AVAILABILITY
# =========================================================

df["AVAILABLE"] = (
    df["CAPACITY"] -
    df["OCCUPIED"]
)

df["OCCUPANCY_PERCENT"] = (
    df["OCCUPIED"] /
    df["CAPACITY"]
) * 100

df["AVAILABILITY_PERCENT"] = (
    100 -
    df["OCCUPANCY_PERCENT"]
)


# Keep percentages valid

df["OCCUPANCY_PERCENT"] = (
    df["OCCUPANCY_PERCENT"]
    .clip(0, 100)
)

df["AVAILABILITY_PERCENT"] = (
    df["AVAILABILITY_PERCENT"]
    .clip(0, 100)
)


# =========================================================
# TIME FEATURES
# =========================================================

df["HOUR"] = df["DATETIME"].dt.hour

df["MINUTE"] = df["DATETIME"].dt.minute

df["DAY_NUMBER"] = (
    df["DATETIME"].dt.dayofweek
)


# =========================================================
# CYCLICAL TIME FEATURES
# =========================================================

df["HOUR_SIN"] = np.sin(
    2 * np.pi * df["HOUR"] / 24
)

df["HOUR_COS"] = np.cos(
    2 * np.pi * df["HOUR"] / 24
)


# =========================================================
# HOLIDAY
# =========================================================

if "HOLIDAY" in df.columns:

    df["HOLIDAY_FLAG"] = (
        df["HOLIDAY"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "yes",
            "true",
            "1",
            "holiday"
        ])
        .astype(int)
    )

else:

    df["HOLIDAY_FLAG"] = 0


# =========================================================
# EVENT
# =========================================================

if "EVENT" in df.columns:

    df["EVENT_FLAG"] = (
        df["EVENT"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(
            lambda x:
            0 if x in ["", "nan", "none", "no", "false", "0"]
            else 1
        )
    )

else:

    df["EVENT_FLAG"] = 0


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "HOUR",
    "MINUTE",
    "DAY_NUMBER",
    "HOUR_SIN",
    "HOUR_COS",
    "HOLIDAY_FLAG",
    "EVENT_FLAG",
    "PARKING ID",
    "PARKING TYPE"
]

TARGET = "OCCUPANCY_PERCENT"


# =========================================================
# DATASET
# =========================================================

X = df[FEATURES]

y = df[TARGET]


# =========================================================
# COLUMNS
# =========================================================

CATEGORICAL = [
    "PARKING ID",
    "PARKING TYPE"
]

NUMERICAL = [
    "HOUR",
    "MINUTE",
    "DAY_NUMBER",
    "HOUR_SIN",
    "HOUR_COS",
    "HOLIDAY_FLAG",
    "EVENT_FLAG"
]


# =========================================================
# PREPROCESSOR
# =========================================================

def make_pipeline(model):

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                CATEGORICAL
            ),

            (
                "numerical",

                "passthrough",

                NUMERICAL
            )
        ]
    )


    return Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


# =========================================================
# TRAIN / TEST
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


# =========================================================
# MODELS
# =========================================================

MODELS = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
}


# =========================================================
# TRAIN MODELS
# =========================================================

model_results = {}

trained_models = {}


for name, algorithm in MODELS.items():

    pipeline = make_pipeline(
        algorithm
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    model_results[name] = {

        "MAE": round(
            mae,
            2
        ),

        "RMSE": round(
            rmse,
            2
        ),

        "R2": round(
            r2,
            2
        )
    }

    trained_models[name] = pipeline


# =========================================================
# SELECT BEST MODEL
#
# Lowest MAE is preferred
# =========================================================

best_model_name = min(

    model_results,

    key=lambda name:
    model_results[name]["MAE"]

)


final_model = trained_models[
    best_model_name
]


print(
    "Best model:",
    best_model_name
)


# =========================================================
# STATUS
# =========================================================

def get_status(availability):

    if availability >= 70:

        return "HIGH"

    elif availability >= 40:

        return "MEDIUM"

    elif availability > 0:

        return "LOW"

    else:

        return "FULL"


# =========================================================
# STATUS CLASS
# =========================================================

def get_status_class(availability):

    if availability >= 70:

        return "high"

    elif availability >= 40:

        return "medium"

    elif availability > 0:

        return "low"

    else:

        return "full"


# =========================================================
# RECOMMENDATION
# =========================================================

def get_recommendation(
    availability,
    parking_id
):

    if availability >= 70:

        return (
            f"{parking_id} is expected to have "
            "good parking availability."
        )

    elif availability >= 40:

        return (
            f"{parking_id} is expected to have "
            "moderate availability."
        )

    elif availability > 0:

        return (
            f"{parking_id} may be busy. "
            "Consider arriving earlier."
        )

    else:

        return (
            f"{parking_id} is predicted to be full. "
            "Try another time."
        )


# =========================================================
# PEAK HOUR
# =========================================================

def find_peak_hour(parking_id):

    parking_data = df[
        df["PARKING ID"] == parking_id
    ]

    if parking_data.empty:

        return "Not available"


    hourly = (
        parking_data
        .groupby("HOUR")[
            "OCCUPANCY_PERCENT"
        ]
        .mean()
    )


    if hourly.empty:

        return "Not available"


    peak = int(
        hourly.idxmax()
    )


    return f"{peak:02d}:00"


# =========================================================
# BUILD MODEL INPUT
# =========================================================

def create_input(
    parking_id,
    hour,
    minute,
    day
):

    parking_type = PARKING_INFO[
        parking_id
    ]["type"]


    row = {

        "HOUR": hour,

        "MINUTE": minute,

        "DAY_NUMBER": day,

        "HOUR_SIN":
            np.sin(
                2 * np.pi * hour / 24
            ),

        "HOUR_COS":
            np.cos(
                2 * np.pi * hour / 24
            ),

        "HOLIDAY_FLAG": 0,

        "EVENT_FLAG": 0,

        "PARKING ID":
            parking_id,

        "PARKING TYPE":
            parking_type
    }


    return pd.DataFrame(
        [row]
    )


# =========================================================
# HOME
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    result = None

    chart_data = None


    if request.method == "POST":

        parking_id = request.form[
            "parking_id"
        ]

        day = int(
            request.form["day"]
        )

        time = request.form[
            "time"
        ]


        hour = int(
            time.split(":")[0]
        )

        minute = int(
            time.split(":")[1]
        )


        # ----------------------------------------------
        # CURRENT PREDICTION
        # ----------------------------------------------

        model_input = create_input(

            parking_id,

            hour,

            minute,

            day
        )


        predicted_occupancy = (
            final_model
            .predict(model_input)[0]
        )


        predicted_occupancy = max(
            0,
            min(
                predicted_occupancy,
                100
            )
        )


        availability = (
            100 -
            predicted_occupancy
        )


        capacity = PARKING_INFO[
            parking_id
        ]["capacity"]


        occupied_spaces = (
            predicted_occupancy /
            100
        ) * capacity


        available_spaces = (
            capacity -
            occupied_spaces
        )


        # ----------------------------------------------
        # NEXT HOUR
        # ----------------------------------------------

        next_hour = (
            hour + 1
        ) % 24


        next_input = create_input(

            parking_id,

            next_hour,

            minute,

            day
        )


        next_occupancy = (
            final_model
            .predict(next_input)[0]
        )


        next_occupancy = max(
            0,
            min(
                next_occupancy,
                100
            )
        )


        next_availability = (
            100 -
            next_occupancy
        )


        # ----------------------------------------------
        # HISTORICAL HOURLY DATA
        # ----------------------------------------------

        parking_history = df[
            df["PARKING ID"]
            == parking_id
        ]


        hourly = (
            parking_history
            .groupby("HOUR")[
                "OCCUPANCY_PERCENT"
            ]
            .mean()
            .reset_index()
        )


        chart_data = {

            "labels":
                hourly["HOUR"]
                .tolist(),

            "values":
                hourly[
                    "OCCUPANCY_PERCENT"
                ]
                .round(1)
                .tolist()
        }


        # ----------------------------------------------
        # RESULT
        # ----------------------------------------------

        result = {

            "parking_id":
                parking_id,

            "parking_name":
                PARKING_INFO[
                    parking_id
                ]["name"],

            "time":
                time,

            "capacity":
                capacity,

            "occupancy":
                round(
                    predicted_occupancy,
                    1
                ),

            "availability":
                round(
                    availability,
                    1
                ),

            "occupied_spaces":
                round(
                    occupied_spaces,
                    1
                ),

            "available_spaces":
                round(
                    available_spaces,
                    1
                ),

            "status":
                get_status(
                    availability
                ),

            "status_class":
                get_status_class(
                    availability
                ),

            "recommendation":
                get_recommendation(
                    availability,
                    parking_id
                ),

            "peak_hour":
                find_peak_hour(
                    parking_id
                ),

            "next_occupancy":
                round(
                    next_occupancy,
                    1
                ),

            "next_availability":
                round(
                    next_availability,
                    1
                ),

            "next_hour":
                f"{next_hour:02d}:{minute:02d}"
        }


    return render_template(

        "index.html",

        result=result,

        chart_data=chart_data,

        best_model=best_model_name
    )


# =========================================================
# PERFORMANCE
# =========================================================

@app.route(
    "/performance"
)
def performance():

    return render_template(

        "performance.html",

        results=model_results,

        best_model=best_model_name
    )


# =========================================================
# ABOUT
# =========================================================

@app.route(
    "/about"
)
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    print(
        "=========================================="
    )

    print(
        " NITTTR SMART PARKING AI"
    )

    print(
        "=========================================="
    )

    print(
        "Best model:",
        best_model_name
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print(
        "=========================================="
    )


    app.run(
        debug=False,
        port=5000
    )