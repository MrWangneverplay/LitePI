"""Fail closed when the repository is not running the paper protocol."""

from __future__ import annotations

import argparse
import json

from protocol_contract import collect


def main() -> int:
    ap = argparse.ArgumentParser(description="Check LitePI strict protocol compliance")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = collect()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("LitePI strict protocol compliance:", "PASS" if report["compliant"] else "FAIL")
        for item in report["items"]:
            mark = "PASS" if item["passed"] else "FAIL"
            print(f"  [{mark}] {item['name']}: required={item['required']} actual={item['actual']}")
            if item["detail"]:
                print(f"         {item['detail']}")
        if report["failures"]:
            print("Failures:", ", ".join(report["failures"]))
    return 0 if report["compliant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
