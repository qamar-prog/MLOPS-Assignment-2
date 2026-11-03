
import os
import sys
import json
import argparse
import joblib
import yaml
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    recall_score,
    accuracy_score,
    roc_auc_score
)


def main():

    # 1. Read command-line arguments
    parser = argparse.ArgumentParser(description="Train Breast Cancer Triage Models")

    parser.add_argument(
        "--data-path",
        default="data/raw/breast_cancer.csv",
        help="Path to the DVC-tracked dataset"
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to save the trained model and model card"
    )

    parser.add_argument(
        "--model-version",
        default="1.0.0",
        help="Version of the trained model"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Read parameters
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    seed = params["seed"]
    test_size = params["data"]["test_size"]
    folds = params["cross_validation"]["folds"]

    # 3. Load dataset
    df = pd.read_csv(args.data_path)

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    if len(df.columns) != 31:
        raise ValueError("Dataset must contain 30 features and 1 target")

    print("Dataset shape:", df.shape)

    # 4. Separate features and target
    X = df.drop(columns=["target"])
    y = (df["target"] == 0).astype(int)

    # 0 = benign, 1 = malignant for model training
    # This makes malignant the positive class.

    feature_names = list(X.columns)

    # 5. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 6. Read model hyperparameters
    lr_params = params["models"]["logistic_regression"]
    rf_params = params["models"]["random_forest"]

    # 7. Create model Pipelines
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=lr_params["C"],
                max_iter=lr_params["max_iter"],
                random_state=seed
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                min_samples_split=rf_params["min_samples_split"],
                min_samples_leaf=rf_params["min_samples_leaf"],
                class_weight=rf_params["class_weight"],
                random_state=seed,
                n_jobs=-1
            ))
        ])
    }

    # 8. Define evaluation metrics
    scoring = {
        "malignant_recall": make_scorer(
            recall_score,
            pos_label=1
        ),
        "accuracy": "accuracy",
        "roc_auc": "roc_auc"
    }

    # 9. Five-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed
    )

    cv_results = {}

    print("\n========== CROSS-VALIDATION RESULTS ==========")

    for model_name, pipeline in models.items():

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        cv_results[model_name] = {
            "malignant_recall": float(
                scores["test_malignant_recall"].mean()
            ),
            "accuracy": float(
                scores["test_accuracy"].mean()
            ),
            "roc_auc": float(
                scores["test_roc_auc"].mean()
            )
        }

        print(f"\nModel: {model_name}")
        print(
            "Mean Malignant Recall:",
            f"{cv_results[model_name]['malignant_recall']:.4f}"
        )
        print(
            "Mean Accuracy:",
            f"{cv_results[model_name]['accuracy']:.4f}"
        )
        print(
            "Mean ROC AUC:",
            f"{cv_results[model_name]['roc_auc']:.4f}"
        )

    # 10. Select model with highest CV malignant recall
    best_model_name = max(
        cv_results,
        key=lambda name: cv_results[name]["malignant_recall"]
    )

    best_model = models[best_model_name]

    print("\nSelected model:", best_model_name)

    # 11. Train selected model on complete training split
    best_model.fit(X_train, y_train)

    # 12. Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "malignant_recall": float(
            recall_score(y_test, y_pred, pos_label=1)
        ),
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_probability)
        )
    }

    print("\n========== HELD-OUT TEST RESULTS ==========")
    print(f"Malignant Recall: {test_metrics['malignant_recall']:.4f}")
    print(f"Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"ROC AUC:          {test_metrics['roc_auc']:.4f}")

    # 13. Save serialized pipeline
    model_path = os.path.join(args.output_dir, "model.joblib")
    joblib.dump(best_model, model_path)

    # 14. Create model card
    model_card = {
        "model_version": args.model_version,
        "model_family": best_model_name,
        "feature_names": feature_names,
        "metrics": {
            "cross_validation": cv_results,
            "test": test_metrics
        },
        "seed": seed,
        "sklearn_version": sklearn.__version__
    }

    model_card_path = os.path.join(
        args.output_dir,
        "model_card.json"
    )

    with open(model_card_path, "w") as file:
        json.dump(model_card, file, indent=4)

    # 15. Save CV results separately
    cv_path = os.path.join(args.output_dir, "cv_results.json")

    with open(cv_path, "w") as file:
        json.dump(cv_results, file, indent=4)

    print("\n========== TRAINING COMPLETE ==========")
    print("Selected model:", best_model_name)
    print("Model saved:", model_path)
    print("Model card saved:", model_card_path)
    print("CV results saved:", cv_path)


    #16 Save metrics for DVC
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    with open(metrics_path, "w") as file:
          json.dump({
             "selected_model": best_model_name,
             "test": test_metrics
           }, file, indent=4)
    # 17. Quality gate
    minimum_recall = 0.95

    if test_metrics["malignant_recall"] < minimum_recall:
        print(
            f"\nQUALITY GATE FAILED: Malignant recall "
            f"{test_metrics['malignant_recall']:.4f} "
            f"is below {minimum_recall:.2f}"
        )
        sys.exit(1)

    print(
        f"\nQUALITY GATE PASSED: Malignant recall "
        f"is at least {minimum_recall:.2f}"
    )


if __name__ == "__main__":
    main()
