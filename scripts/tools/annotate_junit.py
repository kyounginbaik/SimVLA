"""Surface pytest failures as GitHub check annotations when logs require sign-in."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit_xml", type=Path)
    args = parser.parse_args()
    if not args.junit_xml.exists():
        print("::error::pytest did not produce a JUnit XML report")
        return 1

    root = ET.parse(args.junit_xml).getroot()
    count = 0
    for case in root.iter("testcase"):
        for failure in (*case.findall("error"), *case.findall("failure")):
            name = f"{case.get('classname', '')}.{case.get('name', '')}"
            # Collection errors often have a generic ``message`` while the
            # useful ImportError appears only at the end of the traceback.
            detail = failure.text or failure.get("message") or "pytest failed"
            # Workflow commands are single-line and have a small practical limit.
            detail = " ".join(detail.split())[-1400:]
            message = f"{name}: {detail}".replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error::{message}")
            count += 1
    if not count:
        print("::error::pytest failed without a JUnit testcase error; inspect the job log")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
