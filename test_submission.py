#!/usr/bin/env python
"""Full submission validation test for OpenEnv."""

import json
import sys

import requests


def test_space(space_url: str) -> bool:
    """Test all required endpoints on live Space."""
    base = space_url.rstrip("/")
    tests = [
        ("GET /health", "GET", f"{base}/health", None),
        ("GET /tasks", "GET", f"{base}/tasks", None),
        ("POST /reset", "POST", f"{base}/reset", {}),
        ("POST /step", "POST", f"{base}/step", {"action_type": "finalize"}),
        ("GET /grader", "GET", f"{base}/grader", None),
        ("POST /baseline", "POST", f"{base}/baseline", None),
    ]

    all_pass = True
    for name, method, url, data in tests:
        try:
            if method == "GET":
                r = requests.get(url, timeout=10)
            else:
                r = requests.post(url, json=data, timeout=10)

            if r.status_code < 400:
                print(f"✓ {name} ({r.status_code})")
            else:
                print(f"✗ {name} ({r.status_code}): {r.text[:100]}")
                all_pass = False
        except Exception as e:
            print(f"✗ {name}: {str(e)[:100]}")
            all_pass = False

    return all_pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_submission.py <SPACE_URL>")
        print("Example: python test_submission.py https://superneon-hackathon-submission.hf.space")
        return 1

    space_url = sys.argv[1]
    print(f"Testing Space: {space_url}\n")

    if test_space(space_url):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Check Space is Running and endpoints respond.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
