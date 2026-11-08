"""Files a kitchen generation run writes into the shared asset tree.

generate_example.py refuses a kitchen ID whose USDs already exist, so a run that
crashes after exporting must remove what it wrote or the ID can never be retried.
"""
from pathlib import Path
import re


def kitchen_outputs(kitchen_dir, kitchen_num: int) -> set[Path]:
    """Base/rotation USDs and the BODex fact file for exactly this kitchen number."""
    kitchen_dir = Path(kitchen_dir)
    stem = f"kitchen_{kitchen_num:02d}"
    usd = re.compile(rf"{re.escape(stem)}(_[0-9]{{2}})?\.usd")
    found = {p for p in kitchen_dir.glob(stem + "*.usd") if usd.fullmatch(p.name)}
    facts = kitchen_dir / "bodex" / f"kitchen_data_{kitchen_num:02d}.json"
    if facts.is_file():
        found.add(facts)
    return found


def remove_new_outputs(kitchen_dir, kitchen_num: int, before: set[Path]) -> list[Path]:
    """Delete this kitchen's outputs that did not exist before the run; return them."""
    removed = []
    for path in sorted(kitchen_outputs(kitchen_dir, kitchen_num) - set(before)):
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed
