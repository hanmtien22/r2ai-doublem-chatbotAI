import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.query.pipeline import QueryPipeline


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pipeline = QueryPipeline(reference_year=2024)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Nhap cau hoi: ").strip()

    if not question:
        print("Khong co cau hoi.")
        return

    result = pipeline.process(question)
    print("\n" + "=" * 60)
    print("KET QUA PHASE 1")
    print("=" * 60)
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
