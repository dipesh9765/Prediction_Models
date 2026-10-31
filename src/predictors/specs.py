"""Declarative description of every model: its features, valid input ranges and metadata.

Everything that used to be scattered across notebook cells, six HTML pages and ``views.py`` (feature
order, categorical encodings, units, valid ranges) lives here once. Training, validation, the API
schema, the web form and the model cards are all generated from these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Task = Literal["regression", "classification"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Feature:
    """One model input."""

    name: str
    label: str
    kind: Literal["number", "category"]
    description: str = ""
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    choices: tuple[str, ...] = ()
    #: ``True`` when the model can cope with a missing value (it is imputed from training data).
    optional: bool = False
    example: float | str | None = None

    def __post_init__(self) -> None:
        if self.kind == "category" and not self.choices:
            raise ValueError(f"{self.name}: categorical features need choices")
        if self.kind == "number" and (self.minimum is None or self.maximum is None):
            raise ValueError(f"{self.name}: numeric features need a valid range")


@dataclass(frozen=True)
class ModelSpec:
    """A prediction task."""

    key: str
    title: str
    task: Task
    description: str
    target: str
    target_label: str
    target_unit: str
    features: tuple[Feature, ...]
    #: names of the candidate estimators to compare (see ``train.CANDIDATES``)
    candidates: tuple[str, ...]
    data_files: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    disclaimer: str = ""
    positive_label: str = ""
    negative_label: str = ""
    #: minimum cross-validated R2 (regression) or ROC AUC (classification) the selected model must
    #: reach; enforced by ``predictors.check_cards`` after training
    quality_floor: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def numeric(self) -> list[Feature]:
        return [f for f in self.features if f.kind == "number"]

    @property
    def categorical(self) -> list[Feature]:
        return [f for f in self.features if f.kind == "category"]

    def data_paths(self) -> list[Path]:
        return [DATA_DIR / name for name in self.data_files]


CAR = ModelSpec(
    key="car",
    quality_floor=0.8,
    title="Used car resale price",
    task="regression",
    description="Estimates what a used car in the Indian market will sell for, from its age, "
    "mileage, fuel, transmission and the price of a comparable new car.",
    target="Selling_Price",
    target_label="Estimated selling price",
    target_unit="lakh INR",
    data_files=("car/car_data.csv",),
    candidates=("ridge", "random_forest", "hist_gradient_boosting"),
    features=(
        Feature(
            "Year",
            "Year of manufacture",
            "number",
            minimum=1990,
            maximum=2030,
            integer=True,
            example=2015,
        ),
        Feature(
            "Present_Price",
            "Price of the equivalent new car",
            "number",
            unit="lakh INR",
            minimum=0.1,
            maximum=200,
            example=6.5,
        ),
        Feature(
            "Kms_Driven",
            "Kilometres driven",
            "number",
            unit="km",
            minimum=0,
            maximum=1_000_000,
            integer=True,
            example=45000,
        ),
        Feature(
            "Fuel_Type", "Fuel", "category", choices=("Petrol", "Diesel", "CNG"), example="Petrol"
        ),
        Feature(
            "Seller_Type", "Seller", "category", choices=("Dealer", "Individual"), example="Dealer"
        ),
        Feature(
            "Transmission",
            "Transmission",
            "category",
            choices=("Manual", "Automatic"),
            example="Manual",
        ),
        Feature(
            "Owner", "Previous owners", "number", minimum=0, maximum=3, integer=True, example=0
        ),
    ),
    limitations=(
        "Only 299 distinct listings, almost all from 2003-2018; prices outside that period are "
        "extrapolations.",
        "The data comes from one marketplace and does not include model, city or condition.",
    ),
)

DIABETES = ModelSpec(
    key="diabetes",
    quality_floor=0.8,
    title="Diabetes risk screening",
    task="classification",
    description="Estimates the probability that a woman of Pima Indian heritage has diabetes from "
    "eight routine measurements (Pima Indians Diabetes dataset).",
    target="Outcome",
    target_label="Probability of diabetes",
    target_unit="",
    positive_label="Higher risk of diabetes",
    negative_label="Lower risk of diabetes",
    data_files=("diabetes/diabetes.csv",),
    candidates=("logistic_regression", "random_forest", "hist_gradient_boosting"),
    disclaimer="Educational demonstration only. This is not a medical device and must not be used "
    "to diagnose, screen or make decisions about anyone's health.",
    features=(
        Feature(
            "Pregnancies", "Pregnancies", "number", minimum=0, maximum=20, integer=True, example=2
        ),
        Feature(
            "Glucose",
            "Plasma glucose",
            "number",
            unit="mg/dL",
            minimum=40,
            maximum=400,
            example=120,
        ),
        Feature(
            "BloodPressure",
            "Diastolic blood pressure",
            "number",
            unit="mmHg",
            minimum=20,
            maximum=200,
            example=70,
        ),
        Feature(
            "SkinThickness",
            "Triceps skin fold thickness",
            "number",
            unit="mm",
            minimum=1,
            maximum=100,
            optional=True,
            example=25,
        ),
        Feature(
            "Insulin",
            "2-hour serum insulin",
            "number",
            unit="mu U/ml",
            minimum=1,
            maximum=1000,
            optional=True,
            example=80,
        ),
        Feature(
            "BMI", "Body mass index", "number", unit="kg/m2", minimum=10, maximum=70, example=28.5
        ),
        Feature(
            "DiabetesPedigreeFunction",
            "Diabetes pedigree function",
            "number",
            minimum=0.01,
            maximum=3,
            example=0.4,
        ),
        Feature(
            "Age", "Age", "number", unit="years", minimum=18, maximum=100, integer=True, example=33
        ),
    ),
    limitations=(
        "768 patients from a single population (women of Pima heritage aged 21+); it does not "
        "generalise to other groups.",
        "In the source data zero is used as a missing-value code for glucose, blood pressure, skin "
        "thickness, insulin and BMI. These are treated as missing and imputed with the training "
        "median.",
        "Labels come from a 1990s study and a 5-fold ROC AUC of about 0.8 is typical for this "
        "dataset; it is not a clinical-grade score.",
    ),
)

EXERCISE = ModelSpec(
    key="exercise",
    quality_floor=0.99,
    title="Calories burnt during exercise",
    task="regression",
    description="Estimates the energy spent in a workout from the person's body measurements and "
    "the session's duration, heart rate and body temperature.",
    target="Calories",
    target_label="Estimated calories burnt",
    target_unit="kcal",
    data_files=("exercise/exercise.csv", "exercise/calories.csv"),
    candidates=("ridge", "random_forest", "hist_gradient_boosting"),
    features=(
        Feature("Gender", "Gender", "category", choices=("male", "female"), example="male"),
        Feature(
            "Age", "Age", "number", unit="years", minimum=10, maximum=100, integer=True, example=30
        ),
        Feature("Height", "Height", "number", unit="cm", minimum=100, maximum=230, example=175),
        Feature("Weight", "Weight", "number", unit="kg", minimum=25, maximum=250, example=70),
        Feature(
            "Duration",
            "Exercise duration",
            "number",
            unit="min",
            minimum=1,
            maximum=300,
            example=20,
        ),
        Feature(
            "Heart_Rate",
            "Average heart rate",
            "number",
            unit="bpm",
            minimum=40,
            maximum=220,
            example=100,
        ),
        Feature(
            "Body_Temp", "Body temperature", "number", unit="C", minimum=35, maximum=42, example=40
        ),
    ),
    limitations=(
        "15,000 records from a public fitness dataset that looks synthetic or heavily smoothed: "
        "calories are almost a deterministic function of duration and heart rate.",
        "Not validated against calorimetry; use for illustration, not for training plans.",
    ),
)

MEDICAL = ModelSpec(
    key="medical_insurance",
    quality_floor=0.8,
    title="Medical insurance charges",
    task="regression",
    description="Estimates the yearly medical insurance charge billed to a person in the US from "
    "age, BMI, smoking status and a few demographics.",
    target="charges",
    target_label="Estimated yearly charge",
    target_unit="USD",
    data_files=("medical_insurance/insurance.csv",),
    candidates=("ridge", "random_forest", "hist_gradient_boosting"),
    disclaimer="Illustrative model trained on a small public sample. It must not be used to price "
    "insurance, set premiums or make decisions about individuals.",
    features=(
        Feature(
            "age", "Age", "number", unit="years", minimum=18, maximum=100, integer=True, example=35
        ),
        Feature("sex", "Sex", "category", choices=("female", "male"), example="female"),
        Feature(
            "bmi", "Body mass index", "number", unit="kg/m2", minimum=10, maximum=70, example=27.5
        ),
        Feature("children", "Dependants", "number", minimum=0, maximum=10, integer=True, example=1),
        Feature("smoker", "Smoker", "category", choices=("no", "yes"), example="no"),
        Feature(
            "region",
            "Region",
            "category",
            choices=("northeast", "northwest", "southeast", "southwest"),
            example="southeast",
        ),
    ),
    limitations=(
        "1,337 people; the charge distribution is strongly skewed and driven by smoking status, "
        "so errors are large for high-cost individuals.",
        "Sex and region are included because they are in the data, not because they should drive "
        "pricing decisions.",
    ),
)

HOUSE = ModelSpec(
    key="california_housing",
    quality_floor=0.8,
    title="California house value",
    task="regression",
    description="Estimates the median house value of a California census block group from its "
    "income, housing and location statistics (1990 census).",
    target="MedHouseVal",
    target_label="Estimated median house value",
    target_unit="x $100,000",
    data_files=("california_housing/california_housing.csv",),
    candidates=("ridge", "random_forest", "hist_gradient_boosting"),
    features=(
        Feature(
            "MedInc",
            "Median income",
            "number",
            unit="x $10,000",
            minimum=0,
            maximum=20,
            example=3.9,
        ),
        Feature(
            "HouseAge",
            "Median house age",
            "number",
            unit="years",
            minimum=1,
            maximum=100,
            example=29,
        ),
        Feature(
            "AveRooms",
            "Average rooms per household",
            "number",
            minimum=0.5,
            maximum=150,
            example=5.4,
        ),
        Feature(
            "AveBedrms",
            "Average bedrooms per household",
            "number",
            minimum=0.3,
            maximum=40,
            example=1.1,
        ),
        Feature(
            "Population", "Block group population", "number", minimum=3, maximum=40000, example=1166
        ),
        Feature(
            "AveOccup",
            "Average occupants per household",
            "number",
            minimum=0.5,
            maximum=1500,
            example=3.0,
        ),
        Feature(
            "Latitude", "Latitude", "number", unit="deg", minimum=32, maximum=42.5, example=34.2
        ),
        Feature(
            "Longitude",
            "Longitude",
            "number",
            unit="deg",
            minimum=-125,
            maximum=-114,
            example=-118.5,
        ),
    ),
    limitations=(
        "1990 census block groups: prices are decades old and only meaningful in California.",
        "Target values are censored at $500,000 in the source data, so the model cannot predict "
        "above that.",
        "Replaces the Boston housing data used by the original project; that dataset was removed "
        "from scikit-learn because one column encodes the racial make-up of neighbourhoods.",
    ),
)

SPECS: dict[str, ModelSpec] = {s.key: s for s in (CAR, DIABETES, EXERCISE, MEDICAL, HOUSE)}
