import json
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("creditcard.csv")
RESULT_PATH = Path("benchmark_result.json")
RANDOM_STATE = 42


def load_data():
    start = time.perf_counter()

    if DATASET_PATH.exists():
        data = pd.read_csv(DATASET_PATH)
        target = data["Class"].astype(int)
        features = data.drop(columns=["Class"])
        source = "kaggle_creditcard"
    else:
        features_array, target_array = make_classification(
            n_samples=284_807,
            n_features=30,
            n_informative=20,
            n_redundant=5,
            weights=[0.998, 0.002],
            random_state=RANDOM_STATE,
        )
        features = pd.DataFrame(features_array, columns=[f"feature_{i}" for i in range(features_array.shape[1])])
        target = pd.Series(target_array)
        source = "synthetic_fallback"

    load_time = time.perf_counter() - start
    return features, target, load_time, source


def main():
    features, target, load_time, source = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    train_start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    training_time = time.perf_counter() - train_start

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    single_row = x_test.iloc[[0]]
    latency_start = time.perf_counter()
    model.predict_proba(single_row)
    single_latency_ms = (time.perf_counter() - latency_start) * 1000

    batch = x_test.iloc[:1000]
    batch_start = time.perf_counter()
    model.predict_proba(batch)
    batch_time = time.perf_counter() - batch_start

    results = {
        "dataset_source": source,
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "load_data_seconds": round(load_time, 4),
        "training_seconds": round(training_time, 4),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or model.n_estimators),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 6),
        "inference_latency_ms_1_row": round(single_latency_ms, 4),
        "inference_throughput_rows_per_second_1000_rows": round(1000 / batch_time, 2),
    }

    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {RESULT_PATH.resolve()}")


if __name__ == "__main__":
    main()
