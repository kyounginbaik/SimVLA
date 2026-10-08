"""Release tags and artifact names must follow package metadata."""
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_derives_and_checks_version():
    path = ROOT / ".github/workflows/release.yml"
    text = path.read_text(encoding="utf-8")
    yaml.safe_load(text)
    assert "GITHUB_REF_NAME" in text
    assert 'test "$GITHUB_REF_NAME" = "v$version"' in text
    assert "steps.version.outputs.version" in text
    assert "simvla-1.0.0" not in text
    assert "twine check dist/*" not in text
    assert '--notes-file ".github/release-notes/v${VERSION}.md"' in text
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', project).group(1)
    assert (ROOT / ".github" / "release-notes" / f"v{version}.md").is_file()


def test_package_workflow_derives_artifact_version():
    path = ROOT / ".github/workflows/package.yml"
    text = path.read_text(encoding="utf-8")
    yaml.safe_load(text)
    assert "steps.version.outputs.version" in text
    assert "simvla-1.0.0" not in text
    assert "twine check dist/*" not in text


def test_package_version_step_supports_python310_and_follows_install():
    workflow = yaml.safe_load((ROOT / ".github/workflows/package.yml").read_text())
    steps = workflow["jobs"]["cpu"]["steps"]
    version_index = next(i for i, step in enumerate(steps) if step.get("id") == "version")
    script = steps[version_index]["run"]
    assert "importlib.metadata" in script
    assert "tomllib" not in script  # stdlib module is unavailable on Python 3.10
    assert any("pip install '.[scenes,dev]'" in step.get("run", "")
               for step in steps[:version_index])
