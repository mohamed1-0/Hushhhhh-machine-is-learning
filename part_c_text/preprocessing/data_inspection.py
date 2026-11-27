from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_PATH / "data" / "SANAD"
REPORT_PATH = PROJECT_PATH / "report" / "inspection_report.txt"

CLASSES = ["Culture", "Finance", "Medical", "Politics", "Religion", "Sports", "Tech"]

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {DATASET_PATH}")
    for class_name in CLASSES:
        if not (DATASET_PATH / class_name).is_dir():
            raise FileNotFoundError(f"{class_name} folder was not found!")
    print("Dataset folder and all class folders found successfully.\n")

    class_counts = {}
    empty_files = []
    unreadable_files = []
    hashes = defaultdict(list)

    processed = 0
    for class_name in CLASSES:
        txt_files = sorted((DATASET_PATH / class_name).glob("*.txt"))
        class_counts[class_name] = len(txt_files)
        for file_path in txt_files:
            processed += 1
            if processed % 5000 == 0:
                print(f"Processed {processed} files...")

            try:
                content = file_path.read_text(encoding="utf-8").strip()
            except (UnicodeDecodeError, OSError):
                unreadable_files.append(file_path)
                continue

            if not content:
                empty_files.append(file_path)
                continue

            article_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            hashes[article_hash].append((class_name, file_path))

    total_articles = sum(class_counts.values())

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    extra_copies = sum(len(group) - 1 for group in duplicate_groups)

    cross_class_groups = [
        group for group in duplicate_groups
        if len({class_name for class_name, _ in group}) > 1
    ]

    lines = []
    lines.append("SANAD Dataset Inspection Report")
    lines.append("=" * 40)
    lines.append("\nArticles per class:")
    for class_name in CLASSES:
        count = class_counts[class_name]
        percent = 100 * count / total_articles
        lines.append(f"  {class_name:<10} {count:>6}  ({percent:.1f}%)")
    lines.append(f"\nTotal articles (files): {total_articles}")
    lines.append(f"Largest / smallest class ratio: "
                 f"{max(class_counts.values()) / min(class_counts.values()):.2f}")
    lines.append(f"\nEmpty files: {len(empty_files)}")
    lines.append(f"Unreadable files: {len(unreadable_files)}")
    lines.append(f"Duplicate groups: {len(duplicate_groups)}")
    lines.append(f"Extra duplicate copies (to be removed): {extra_copies}")
    lines.append(f"Duplicate groups across DIFFERENT classes: {len(cross_class_groups)}")
    lines.append(f"Expected unique articles: {len(hashes)}")

    report = "\n".join(lines)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
