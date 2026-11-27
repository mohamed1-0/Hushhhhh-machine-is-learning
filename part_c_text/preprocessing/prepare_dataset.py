from pathlib import Path
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from text_cleaning import clean_arabic_text

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_PATH / "data" / "SANAD"
RESULTS_PATH = PROJECT_PATH / "results" / "prepared_data"
REPORT_PATH = PROJECT_PATH / "report" / "preparation_report.txt"

CLASSES = ["Culture", "Finance", "Medical", "Politics", "Religion", "Sports", "Tech"]

RANDOM_STATE = 42
MIN_TOKENS = 5

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {DATASET_PATH}")
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    seen_raw_hashes = set()
    seen_clean_texts = {}
    conflict_texts = set()
    rows = []

    total_files = 0
    unreadable_count = 0
    raw_duplicate_count = 0
    clean_duplicate_count = 0
    too_short_count = 0

    for class_name in CLASSES:
        txt_files = sorted((DATASET_PATH / class_name).glob("*.txt"))

        for file_path in txt_files:
            total_files += 1
            try:
                original_text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                unreadable_count += 1
                continue

            raw_hash = hashlib.sha256(original_text.strip().encode("utf-8")).hexdigest()
            if raw_hash in seen_raw_hashes:
                raw_duplicate_count += 1
                continue
            seen_raw_hashes.add(raw_hash)

            raw_words = len(original_text.split())
            cleaned_text = clean_arabic_text(original_text)

            if len(cleaned_text.split()) < MIN_TOKENS:
                too_short_count += 1
                continue

            if cleaned_text in seen_clean_texts:
                clean_duplicate_count += 1
                if seen_clean_texts[cleaned_text] != class_name:
                    conflict_texts.add(cleaned_text)
                continue
            seen_clean_texts[cleaned_text] = class_name

            rows.append({"text": cleaned_text, "label": class_name, "raw_words": raw_words})

        print(f"{class_name}: processed")

    df = pd.DataFrame(rows)

    if conflict_texts:
        df = df[~df["text"].isin(conflict_texts)].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    train_df.to_csv(RESULTS_PATH / "train.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(RESULTS_PATH / "validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(RESULTS_PATH / "test.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("Dataset Preparation Summary")
    lines.append("=" * 40)
    lines.append(f"Total files read: {total_files}")
    lines.append(f"Unreadable files: {unreadable_count}")
    lines.append(f"Exact duplicates removed (raw text): {raw_duplicate_count}")
    lines.append(f"Too short after cleaning (< {MIN_TOKENS} words): {too_short_count}")
    lines.append(f"Near-duplicates removed (same cleaned text): {clean_duplicate_count}")
    lines.append(f"Label-conflict texts removed: {len(conflict_texts)}")
    lines.append(f"Final number of articles: {len(df)}")
    lines.append("\nClass distribution (all data):")
    lines.append(df["label"].value_counts().to_string())
    lines.append(f"\nTrain: {len(train_df)} | Validation: {len(validation_df)} | Test: {len(test_df)}")
    lines.append("\nTrain class distribution:")
    lines.append(train_df["label"].value_counts().to_string())
    lines.append("\nValidation class distribution:")
    lines.append(validation_df["label"].value_counts().to_string())
    lines.append("\nTest class distribution:")
    lines.append(test_df["label"].value_counts().to_string())

    report = "\n".join(lines)
    print("\n" + report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
