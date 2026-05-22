"""Generate synthetic CSVs for ProScore (legacy entry point).

Writes:

- ``tests/test_data.csv`` — 800 rows, stress-test profile
- ``tests/demo_scorecard_data.csv`` — 6000 rows, notebook demo profile

Usage::

    python tests/generate_test_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_credit import _print_summary, generate_credit_data  # noqa: E402


def main() -> None:
    test_df = generate_credit_data(800, profile="test", seed=42)
    test_df.to_csv("tests/test_data.csv", index=False)
    _print_summary(test_df, "test_data.csv")

    demo_df = generate_credit_data(6000, profile="demo", seed=42)
    demo_df.to_csv("tests/demo_scorecard_data.csv", index=False)
    _print_summary(demo_df, "demo_scorecard_data.csv")


if __name__ == "__main__":
    main()
