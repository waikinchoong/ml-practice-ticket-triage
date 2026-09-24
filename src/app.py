"""ML PRACTICE: Help-Desk Ticket Triage
======================================

Trains the exact tool combination the Final Assessment's ML task (Task 4)
will check: load and audit a CSV, clean it with pandas, hold out a fixed
test set, compare a DummyClassifier baseline against one trained classifier
through the supplied preprocessing scaffold, and report accuracy, macro-F1
and a confusion matrix.

Complete every TODO below. Do not rename the functions, their arguments, or
TARGET_COLUMN / IDENTIFIER_COLUMN - the tests and the CI workflow call these
functions directly by name.

Run it with:
    python -m src.app --data data/helpdesk_tickets.csv --output outputs/result.json

Check your work with:
    pytest -v
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.pipeline_tools import (
    build_baseline,
    build_pipeline,
    build_preprocessor,
    evaluate_predictions,
)

TARGET_COLUMN = "response_tier"
IDENTIFIER_COLUMN = "ticket_id"


def load_data(path: str) -> pd.DataFrame:
    """TODO (Task 4a): load the CSV at `path` with pandas and return it.

    Before moving on, run this in a notebook or a scratch script and look at:
    df.shape, df.dtypes, df.isna().sum(), and df[TARGET_COLUMN].value_counts().
    You do not need to print these inside this function - just look at them
    yourself so you understand what you are cleaning next.
    """
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """TODO (Task 4b): return (numeric_columns, categorical_columns).

    Look at the columns in data/README.md. Decide which columns are genuine
    predictors and which is only a row identifier - IDENTIFIER_COLUMN and
    TARGET_COLUMN must NOT appear in either list you return.
    """
    numeric_columns = [
        "wait_time_minutes",
        "reopened_count",
        "resolution_notes_length",
    ]

    categorical_columns = [
        "device_type",
        "error_code",
        "reported_severity",
    ]

    return numeric_columns, categorical_columns


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """TODO (Task 4c): build X (features) and y (target), then call
    sklearn's train_test_split with a FIXED random_state and stratify=y so
    the split is reproducible and every class is represented in both sets.

    Return exactly: x_train, x_test, y_train, y_test
    """
    numeric_columns, categorical_columns = get_feature_columns(df)
    feature_columns = numeric_columns + categorical_columns

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_baseline(x_train, y_train, numeric_columns, categorical_columns):
    """TODO (Task 4c): build a preprocessor with build_preprocessor(), wrap it
    in a DummyClassifier baseline with build_baseline(), fit it on the
    training data, and return the fitted pipeline.
    """
    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    baseline = build_baseline(preprocessor)
    baseline.fit(x_train, y_train)

    return baseline


def train_classifier(x_train, y_train, numeric_columns, categorical_columns):
    """TODO (Task 4c): build a preprocessor, wrap ONE real scikit-learn
    classifier of your choice in it with build_pipeline() (for example
    DecisionTreeClassifier, LogisticRegression or KNeighborsClassifier),
    fit it, and return the fitted pipeline.

    Import your chosen classifier at the top of this file.
    """
    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    classifier = LogisticRegression(
        max_iter=1000,
    )

    pipeline = build_pipeline(
        preprocessor,
        classifier,
    )

    pipeline.fit(x_train, y_train)

    return pipeline


def run(
    data_path: str = "data/helpdesk_tickets.csv",
    output_path: str = "outputs/result.json",
) -> dict:
    """Provided - wires your functions together and saves outputs/result.json.
    You should not need to edit this, but read it so you know what it expects
    from the functions above.
    """
    df = load_data(data_path)
    numeric_columns, categorical_columns = get_feature_columns(df)
    x_train, x_test, y_train, y_test = make_split(df)

    baseline = train_baseline(
        x_train,
        y_train,
        numeric_columns,
        categorical_columns,
    )

    classifier = train_classifier(
        x_train,
        y_train,
        numeric_columns,
        categorical_columns,
    )

    result = {
        "baseline": evaluate_predictions(
            baseline,
            x_test,
            y_test,
        ),
        "classifier": evaluate_predictions(
            classifier,
            x_test,
            y_test,
        ),
    }

    # Keep the held-out row references and true labels beside the predictions.
    # Students need these fields to identify and explain specific errors in
    # REFLECTION.md without reconstructing the split in a separate notebook.
    result["test_index"] = [int(index) for index in x_test.index]
    result["actual"] = y_test.tolist()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Baseline   accuracy={result['baseline']['accuracy']:.3f}  "
        f"macro_f1={result['baseline']['macro_f1']:.3f}"
    )

    print(
        f"Classifier accuracy={result['classifier']['accuracy']:.3f}  "
        f"macro_f1={result['classifier']['macro_f1']:.3f}"
    )

    # TODO (Task 4e): inspect at least three rows your classifier got wrong.
    # Compare result['classifier']['predicted'] with result['actual']; use
    # result['test_index'] to locate each original ticket in the CSV.
    # Write one defensible feature/preprocessing improvement, and one
    # bias/fairness/deployment limitation, into REFLECTION.md.

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/helpdesk_tickets.csv")
    parser.add_argument("--output", default="outputs/result.json")
    args = parser.parse_args()

    run(args.data, args.output)


if __name__ == "__main__":
    main()