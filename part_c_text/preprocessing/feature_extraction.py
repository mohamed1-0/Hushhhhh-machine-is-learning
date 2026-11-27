from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_PATH = Path(__file__).resolve().parents[1]
PREPARED_PATH = PROJECT_PATH / "results" / "prepared_data"
FEATURES_PATH = PROJECT_PATH / "results" / "tfidf_features"
REPORT_PATH = PROJECT_PATH / "report" / "feature_extraction_report.txt"

CLASSES = ["Culture", "Finance", "Medical", "Politics", "Religion", "Sports", "Tech"]

MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95
TOP_TERMS = 15

def load_split(name):
    path = PREPARED_PATH / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare_dataset.py first. Missing: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("")
    return df["text"], df["label"]


def count_empty_rows(matrix):
    return int((matrix.getnnz(axis=1) == 0).sum())

def main():
    FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train_text, y_train = load_split("train")
    X_validation_text, y_validation = load_split("validation")
    X_test_text, y_test = load_split("test")
    print("Prepared dataset files loaded successfully.")

    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )

    print("Fitting TF-IDF on training data...")
    X_train = tfidf.fit_transform(X_train_text)
    print("Transforming validation data...")
    X_validation = tfidf.transform(X_validation_text)
    print("Transforming test data...")
    X_test = tfidf.transform(X_test_text)

    assert X_train.shape[1] == X_validation.shape[1] == X_test.shape[1], "feature count mismatch"
    assert X_train.shape[0] == len(y_train), "train rows/labels mismatch"
    assert X_validation.shape[0] == len(y_validation), "validation rows/labels mismatch"
    assert X_test.shape[0] == len(y_test), "test rows/labels mismatch"

    save_npz(FEATURES_PATH / "X_train_tfidf.npz", X_train)
    save_npz(FEATURES_PATH / "X_validation_tfidf.npz", X_validation)
    save_npz(FEATURES_PATH / "X_test_tfidf.npz", X_test)

    y_train.to_csv(FEATURES_PATH / "y_train.csv", index=False)
    y_validation.to_csv(FEATURES_PATH / "y_validation.csv", index=False)
    y_test.to_csv(FEATURES_PATH / "y_test.csv", index=False)

    joblib.dump(tfidf, FEATURES_PATH / "tfidf_vectorizer.joblib")

    feature_names = tfidf.get_feature_names_out()
    n_features = len(feature_names)
    n_bigrams = sum(" " in name for name in feature_names)
    n_unigrams = n_features - n_bigrams

    rows, cols = X_train.shape
    density = X_train.nnz / (rows * cols)
    avg_nonzero = X_train.nnz / rows

    top_rows = []
    for class_name in CLASSES:
        mask = (y_train == class_name).to_numpy()
        mean_tfidf = np.asarray(X_train[mask].mean(axis=0)).ravel()
        for rank, idx in enumerate(np.argsort(mean_tfidf)[::-1][:TOP_TERMS], start=1):
            top_rows.append({
                "class": class_name,
                "rank": rank,
                "term": feature_names[idx],
                "mean_tfidf": round(float(mean_tfidf[idx]), 4),
            })
    pd.DataFrame(top_rows).to_csv(FEATURES_PATH / "top_tfidf_terms_per_class.csv", index=False, encoding="utf-8-sig")

    lines = [
        "TF-IDF Feature Extraction Report",
        "=" * 40,
        "Settings:",
        f"  max_features = {MAX_FEATURES}",
        f"  ngram_range  = {NGRAM_RANGE}",
        f"  min_df       = {MIN_DF}",
        f"  max_df       = {MAX_DF}",
        "  sublinear_tf = True",
        "",
        f"Final vocabulary size (feature dimension): {n_features}",
        f"  unigrams: {n_unigrams}",
        f"  bigrams : {n_bigrams}",
        "",
        f"Train      matrix shape: {X_train.shape}",
        f"Validation matrix shape: {X_validation.shape}",
        f"Test       matrix shape: {X_test.shape}",
        "",
        f"Train non-zero values: {X_train.nnz}",
        f"Average non-zero features per article: {avg_nonzero:.1f}",
        f"Train matrix density: {100 * density:.2f}%",
        "",
        "Articles with no feature at all (empty vectors):",
        f"  train: {count_empty_rows(X_train)} | validation: {count_empty_rows(X_validation)} | test: {count_empty_rows(X_test)}",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
