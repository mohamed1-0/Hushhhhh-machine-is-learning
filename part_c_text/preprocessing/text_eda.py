from pathlib import Path
from collections import Counter
import pandas as pd

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_PATH / "data" / "SANAD"
PREPARED_PATH = PROJECT_PATH / "results" / "prepared_data"
EDA_PATH = PROJECT_PATH / "results" / "eda"
REPORT_PATH = PROJECT_PATH / "report" / "eda_report.txt"

CLASSES = ["Culture", "Finance", "Medical", "Politics", "Religion", "Sports", "Tech"]
TOP_N = 15

def length_stats(df):
    stats = df.groupby("label")["words"].agg(
        ["count", "mean", "median", "min", "max", "std"]
    ).round(1)
    return stats.reindex(CLASSES)

def main():
    EDA_PATH.mkdir(parents=True, exist_ok=True)

    for name in ["train.csv", "validation.csv", "test.csv"]:
        if not (PREPARED_PATH / name).exists():
            raise FileNotFoundError(f"Run prepare_dataset.py first. Missing: {name}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {DATASET_PATH}")

    train_df = pd.read_csv(PREPARED_PATH / "train.csv")
    validation_df = pd.read_csv(PREPARED_PATH / "validation.csv")
    test_df = pd.read_csv(PREPARED_PATH / "test.csv")
    full_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

    report = []

    counts = full_df["label"].value_counts().reindex(CLASSES, fill_value=0)

    report.append("Class distribution (final dataset):")
    report.append(counts.to_string())

    raw_records = []
    for class_name in CLASSES:
        for file_path in sorted((DATASET_PATH / class_name).glob("*.txt")):
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            raw_records.append({"label": class_name, "words": len(text.split())})
    raw_df = pd.DataFrame(raw_records)

    raw_stats = length_stats(raw_df)
    raw_stats.to_csv(EDA_PATH / "raw_length_stats.csv")
    report.append("\nRaw article length in words (before cleaning, all readable files):")
    report.append(raw_stats.to_string())

    train_df["text"] = train_df["text"].fillna("")
    train_df["words"] = train_df["text"].str.split().str.len()

    cleaned_stats = length_stats(train_df)
    cleaned_stats.to_csv(EDA_PATH / "cleaned_length_stats.csv")
    report.append("\nCleaned article length in words (train set):")
    report.append(cleaned_stats.to_string())

    raw_mean = train_df["raw_words"].mean()
    cleaned_mean = train_df["words"].mean()
    if raw_mean > 0:
        report.append(
            f"\nAverage words per article in train set: "
            f"raw = {raw_mean:.1f}, cleaned = {cleaned_mean:.1f}"
        )
        report.append(
            f"Average word count reduction after cleaning: "
            f"{100 * (1 - cleaned_mean / raw_mean):.1f}%"
        )
    else:
        report.append("\nCannot calculate word reduction: raw mean is zero.")

    overall_counter = Counter()
    top_rows = []
    for class_name in CLASSES:
        class_counter = Counter()
        for text in train_df.loc[train_df["label"] == class_name, "text"]:
            class_counter.update(text.split())
        overall_counter.update(class_counter)
        for rank, (word, freq) in enumerate(class_counter.most_common(TOP_N), start=1):
            top_rows.append({"class": class_name, "rank": rank, "word": word, "count": freq})

    pd.DataFrame(top_rows).to_csv(
        EDA_PATH / "top_words_per_class.csv", index=False, encoding="utf-8-sig"
    )

    report.append(f"\nVocabulary size of the cleaned train set (unique words): {len(overall_counter)}")
    report.append(f"Total words in the cleaned train set: {sum(overall_counter.values())}")

    text_report = "\n".join(report)
    print(text_report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text_report, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
