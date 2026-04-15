"""Colab-ready training script for electric bus energy consumption regression with tuning."""

# 1) Imports
import re
import numpy as np
import pandas as pd
import joblib

from google.colab import files
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor


# Helper: evaluation function to keep code modular

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model and return prediction metrics."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


# Helper: normalize column names for robust matching

def normalize_name(name):
    """Normalize column name for case-insensitive, punctuation-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


# Helper: infer target column from common patterns

def infer_target_column(columns):
    """Infer energy-consumption target from known naming patterns."""
    normalized_map = {normalize_name(col): col for col in columns}

    common_candidates = [
        "energy consumption",
        "energy_consumption",
        "energy_consumption_kwh",
        "energy consumption kwh",
        "energy_kwh",
        "energyused",
        "target",
        "y",
    ]

    for candidate in common_candidates:
        key = normalize_name(candidate)
        if key in normalized_map:
            return normalized_map[key]

    # Fallback heuristic: choose first numeric column containing both energy and consumption
    for col in columns:
        ncol = normalize_name(col)
        if "energy" in ncol and "consumption" in ncol:
            return col

    return None


# 2) Upload dataset from Google Colab
print("Upload your CSV dataset file when prompted...")
uploaded = files.upload()
file_name = next(iter(uploaded))

df = pd.read_csv(file_name)
print(f"Loaded file: {file_name}")
print(f"Dataset shape: {df.shape}")


# 3) Define target and feature matrix
#    Try to auto-detect target; allow user override.
auto_target = infer_target_column(df.columns)

if auto_target is not None:
    print(f"\nAuto-detected target column: '{auto_target}'")
else:
    print("\nCould not auto-detect target column.")

print("Available columns:")
print(list(df.columns))

user_target = input(
    "\nEnter target column name (press Enter to use auto-detected target): "
).strip()

if user_target:
    TARGET_COLUMN = user_target
else:
    TARGET_COLUMN = auto_target

if TARGET_COLUMN is None or TARGET_COLUMN not in df.columns:
    raise ValueError(
        "A valid target column was not selected. "
        "Please rerun and enter one from the available columns list."
    )

print(f"Using target column: '{TARGET_COLUMN}'")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# 4) Detect numerical and categorical columns automatically
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nDetected feature types:")
print(f"Numerical columns ({len(numeric_features)}): {numeric_features}")
print(f"Categorical columns ({len(categorical_features)}): {categorical_features}")


# 5) Build preprocessing pipelines
#    - Missing values handled with SimpleImputer
#    - Numerical features scaled with StandardScaler
#    - Categorical features one-hot encoded
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)


# 6) Split data into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 7) Define models and grids for tuning
lin_pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
)

rf_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ]
)

gb_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42)),
    ]
)

rf_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}

gb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.03, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 1.0],
}


# 8) Tune Random Forest with GridSearchCV
print("\nTuning Random Forest Regressor with GridSearchCV...")
rf_grid = GridSearchCV(
    estimator=rf_pipe,
    param_grid=rf_param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)
print(f"Best RF CV R2: {rf_grid.best_score_:.4f}")


# 9) Tune Gradient Boosting with GridSearchCV
print("\nTuning Gradient Boosting Regressor with GridSearchCV...")
gb_grid = GridSearchCV(
    estimator=gb_pipe,
    param_grid=gb_param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)
gb_grid.fit(X_train, y_train)
gb_best = gb_grid.best_estimator_
print("Best GB params:", gb_grid.best_params_)
print(f"Best GB CV R2: {gb_grid.best_score_:.4f}")


# 10) Train Linear Regression baseline (no tuning needed)
lin_pipe.fit(X_train, y_train)


# 11) Evaluate linear + tuned models
results = []
results.append(evaluate_model(lin_pipe, X_test, y_test, "Linear Regression"))
results.append(evaluate_model(rf_best, X_test, y_test, "Random Forest (Tuned)"))
results.append(evaluate_model(gb_best, X_test, y_test, "Gradient Boosting (Tuned)"))


# 12) Build Voting Regressor using tuned estimators
#     Use fitted preprocessor output from training data to avoid double preprocessing.
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

rf_tuned_est = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=rf_grid.best_params_["model__n_estimators"],
    max_depth=rf_grid.best_params_["model__max_depth"],
    min_samples_split=rf_grid.best_params_["model__min_samples_split"],
    min_samples_leaf=rf_grid.best_params_["model__min_samples_leaf"],
)

gb_tuned_est = GradientBoostingRegressor(
    random_state=42,
    n_estimators=gb_grid.best_params_["model__n_estimators"],
    learning_rate=gb_grid.best_params_["model__learning_rate"],
    max_depth=gb_grid.best_params_["model__max_depth"],
    subsample=gb_grid.best_params_["model__subsample"],
)

voting_model = VotingRegressor(
    estimators=[
        ("lr", LinearRegression()),
        ("rf", rf_tuned_est),
        ("gb", gb_tuned_est),
    ]
)
voting_model.fit(X_train_pre, y_train)

voting_preds = voting_model.predict(X_test_pre)
voting_results = {
    "Model": "Voting Regressor (Tuned Base Models)",
    "MAE": mean_absolute_error(y_test, voting_preds),
    "RMSE": np.sqrt(mean_squared_error(y_test, voting_preds)),
    "R2": r2_score(y_test, voting_preds),
}
results.append(voting_results)


# 13) Print model comparison clearly
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

print("\n=== Model Performance Comparison ===")
print(results_df.to_string(index=False))


# 14) Select best model (highest R2)
best_model_name = results_df.loc[0, "Model"]
print(f"\nBest model based on R2: {best_model_name}")

# Create serializable artifact(s) based on best model
if best_model_name == "Linear Regression":
    best_model = lin_pipe.named_steps["model"]
    fitted_preprocessor = lin_pipe.named_steps["preprocessor"]
elif best_model_name == "Random Forest (Tuned)":
    best_model = rf_best.named_steps["model"]
    fitted_preprocessor = rf_best.named_steps["preprocessor"]
elif best_model_name == "Gradient Boosting (Tuned)":
    best_model = gb_best.named_steps["model"]
    fitted_preprocessor = gb_best.named_steps["preprocessor"]
else:
    # For voting model, preprocessing already fit in `preprocessor`
    best_model = voting_model
    fitted_preprocessor = preprocessor


# 15) Save trained model and preprocessing pipeline
#     - model.pkl contains final estimator only
#     - preprocessor.pkl contains fitted preprocessing steps
joblib.dump(best_model, "model.pkl")
joblib.dump(fitted_preprocessor, "preprocessor.pkl")

print("\nSaved files:")
print("- model.pkl")
print("- preprocessor.pkl")
