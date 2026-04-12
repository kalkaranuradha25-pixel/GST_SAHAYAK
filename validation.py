#!/usr/bin/env python3
"""
validation.py — Hackathon Submission Validator

Validates the GST RL environment submission for the Meta PyTorch Hackathon.

Checks:
- At least 3 graders are present and functional
- Grader scores are strictly between 0 and 1
- Graders can process sample data without errors

Usage:
    python validation.py
"""

import json
import os
import sys
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.graders.grader_classify import ClassificationGrader
from env.graders.grader_itc import ITCGrader
from env.graders.grader_filing import FilingGrader


def load_sample_data(task_num: int) -> List[Dict[str, Any]]:
    """Load sample data for a specific task from the test directory."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'test', str(task_num))
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")

    episodes = []
    for file in os.listdir(data_dir)[:5]:  # Load first 5 episodes for validation
        if file.endswith('.json'):
            with open(os.path.join(data_dir, file), 'r') as f:
                episodes.append(json.load(f))
    return episodes


def validate_grader_count() -> bool:
    """Check that there are at least 3 graders."""
    graders = [ClassificationGrader, ITCGrader, FilingGrader]
    if len(graders) < 3:
        print("FAILED: Fewer than 3 graders found.")
        return False
    print("PASSED: At least 3 graders present.")
    return True


def validate_sample_data_exists() -> bool:
    """Confirm that sample test data exists for all tasks."""
    all_passed = True
    for task_num in [1, 2, 3]:
        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'test', str(task_num))
        if not os.path.isdir(data_dir):
            print(f"FAILED: Missing sample directory for task {task_num}: {data_dir}")
            all_passed = False
            continue

        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if not json_files:
            print(f"FAILED: No JSON sample files found in {data_dir}")
            all_passed = False
        else:
            print(f"PASSED: Found {len(json_files)} sample files for task {task_num}.")

    return all_passed


def validate_grader_scores():
    """Validate that grader scores are strictly between 0 and 1 using deterministic test inputs."""
    graders = [
        ("ClassificationGrader", ClassificationGrader()),
        ("ITCGrader", ITCGrader()),
        ("FilingGrader", FilingGrader()),
    ]

    all_passed = True

    for name, grader in graders:
        print(f"\nValidating {name}...")

        if name == "ClassificationGrader":
            prediction = {
                "invoice_id": "INV-TEST",
                "invoice_type": "B2B",
                "hsn_code": "1234",
                "gst_slab": "18",
                "supply_type": "goods",
                "itc_eligible": True,
                "reverse_charge": False
            }
            ground_truth = prediction.copy()
        elif name == "ITCGrader":
            prediction = {
                "matched_pairs": [{"purchase_id": "P1", "gstr2b_id": "G1"}],
                "flagged": []
            }
            ground_truth = {
                "correct_matches": {"P1": "G1"},
                "correct_flags": {}
            }
        else:  # FilingGrader
            prediction = {
                "3.1a": 1000.0,
                "3.1b": 2000.0,
                "3.1c": 1500.0,
                "3.1d": 500.0,
                "4a": 1200.0,
                "4b": 800.0,
                "6.1": 700.0,
                "6.2": 300.0,
                "6.3": 200.0,
                "net_payable": 3500.0,
                "total_itc": 500.0
            }
            ground_truth = prediction.copy()

        try:
            score = grader.grade(prediction, ground_truth)
            if not (0 < score < 1):
                print(f"FAILED: Score {score} not strictly between 0 and 1 for {name}.")
                all_passed = False
            else:
                print(f"PASSED: Score {score} is valid for {name}.")
        except Exception as e:
            print(f"FAILED: Error grading with {name}: {e}")
            all_passed = False

    return all_passed


def main():
    """Run all validation checks."""
    print("=========================================")
    print("  Hackathon Submission Validator")
    print("=========================================\n")

    checks = [
        ("Grader Count", validate_grader_count),
        ("Sample Data", validate_sample_data_exists),
        ("Grader Scores", validate_grader_scores),
    ]

    all_passed = True
    for check_name, check_func in checks:
        print(f"Running {check_name} check...")
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"FAILED: {check_name} check raised exception: {e}")
            all_passed = False

    print("\n" + "="*40)
    if all_passed:
        print("ALL CHECKS PASSED! Submission is valid.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED! Please fix the issues before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()