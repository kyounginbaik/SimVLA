#!/usr/bin/env python3
"""Copy the prepared RB-Y1 root USD while removing its workstation-specific absolute reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf


def relativize(source: Path, output: Path) -> None:
    layer = Sdf.Layer.FindOrOpen(str(source.resolve()))
    if layer is None:
        raise ValueError(f"could not open USD layer: {source}")
    refs = list(layer.GetExternalReferences())
    if len(refs) != 1 or not Path(refs[0]).is_absolute() or Path(refs[0]).name != "model.usd":
        raise ValueError(f"expected exactly one absolute model.usd reference, found {refs}")
    if not layer.UpdateExternalReference(refs[0], "model.usd"):
        raise ValueError(f"failed to replace absolute reference {refs[0]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not layer.Export(str(output)):
        raise OSError(f"could not export portable USD layer: {output}")
    check = Sdf.Layer.FindOrOpen(str(output.resolve()))
    if list(check.GetExternalReferences()) != ["model.usd"]:
        raise ValueError(f"portable USD still contains unexpected references: {check.GetExternalReferences()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    relativize(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
