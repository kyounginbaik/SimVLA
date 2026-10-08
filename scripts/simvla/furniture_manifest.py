"""Read/write side of an Objaverse furniture manifest, for any furniture type.

`chair_manifest` was this file, specialised to chairs. `table_manifest` is the same module
specialised to tables, and the only differences between them are five strings: the asset root's
environment variable and default directory, the manifest's filename, the key its entries live
under, and the human name of the thing. Everything else -- the empty skeleton, the "an absent
manifest is not an error" rule, the accepted() filter, save()'s mkdir -- is identical, and was
duplicated once before this existed.

stdlib only -- no objaverse, no trimesh, no isaaclab -- so anything that just wants to know "which
tables are accepted?" can ask without dragging in the fetch/curate/export/convert stack that
produced the answer. That property belonged to chair_manifest and is preserved here.

Every entry ACCUMULATES fields as it passes through the pipeline's stages rather than being
replaced at each stage; see scripts/tools/build_chair_manifest.py, whose FurnitureType seam decides
which fields a given furniture type records.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Library:
    """One furniture library's location and shape on disk.

    name          -- singular, human: "chair", "table". Used in messages only.
    entries_key   -- the manifest key its candidate list lives under: "chairs", "tables".
    manifest_name -- the JSON file's basename, stored inside the asset root.
    env_var       -- environment variable that overrides the asset root.
    default_dir   -- where the asset root is when nothing overrides it.
    """

    name: str
    entries_key: str
    manifest_name: str
    env_var: str
    default_dir: str

    def root(self, obj_dir=None) -> str:
        """The asset root: an explicit argument, else $<env_var>, else the /lustre default.

        Mirrors grasp_manifest._root() / BODEX_OBJ_DIR (grasp_manifest.py:18-22) so a checkout
        without /lustre can point the environment variable somewhere else rather than failing
        outright.
        """
        return obj_dir or os.environ.get(self.env_var) or self.default_dir

    def manifest_path(self, obj_dir=None) -> str:
        return os.path.join(self.root(obj_dir), self.manifest_name)

    def empty_skeleton(self) -> dict:
        """What load() returns when no manifest file exists yet.

        Carries the provenance keys the pipeline fills in as it runs -- objaverse_version (the
        installed objaverse package's version, once a stage that imports it has run),
        generated_from (the root this manifest describes), and thresholds (curate's criteria) --
        so every stage can write into these keys directly rather than each one checking whether
        they exist.
        """
        return {
            "objaverse_version": None,
            "generated_from": None,
            "thresholds": None,
            self.entries_key: [],
        }

    def load(self, obj_dir=None) -> dict:
        """The manifest, or an empty skeleton when there isn't a readable one yet.

        Unlike grasp_manifest.load (which returns None for "no dataset here" and pushes the
        fallback onto every caller, because BODex data is generated once, offline, ahead of time),
        a furniture manifest is built up incrementally -- fetch appends candidates, curate judges
        them, export and convert fill in more -- so the very first call, on a checkout with no
        manifest at all, must already be safe to mutate and save. An empty skeleton makes every
        stage's first run and its hundredth run the same code path.
        """
        try:
            with open(self.manifest_path(obj_dir), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return self.empty_skeleton()

    def accepted(self, manifest) -> list[dict]:
        """Entries curate has judged and accepted. Excludes rejects and unjudged candidates."""
        return [e for e in manifest.get(self.entries_key, []) if e.get("accepted") is True]

    def save(self, manifest, obj_dir=None) -> None:
        """Write the manifest back beside the dataset, creating the root directory if needed."""
        path = self.manifest_path(obj_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
