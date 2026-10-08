"""Importing shared simulation helpers must not require a desktop/Tcl runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_gui_dependencies_are_local_to_thumbnail_ui():
    path = Path(__file__).parents[1] / "source/isaaclab/isaaclab/simvla/utils.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith("tkinter") for name in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tkinter")
            assert all(name.name != "ImageTk" for name in node.names)
    ui = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_thumbnails")
    assert any(isinstance(n, ast.Import) and n.names[0].name == "tkinter" for n in ui.body)


def test_optional_vr_import_is_lazy():
    path = Path(__file__).parents[1] / "source/isaaclab/isaaclab/devices/__init__.py"
    tree = ast.parse(path.read_text())
    assert not any(isinstance(n, ast.ImportFrom) and n.module == "oculus" for n in tree.body)
    getter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "__getattr__")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "import_module" for n in ast.walk(getter))


def test_headless_authoring_does_not_import_tk():
    path = Path(__file__).parents[1] / "scripts/simvla/simvla_data_generator.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith("tkinter") for name in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tkinter")
        if isinstance(node, ast.ClassDef):
            assert all(not (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                            and base.value.id == "simpledialog") for base in node.bases)


def test_lazy_dialog_preserves_tk_initialization():
    path = Path(__file__).parents[1] / "scripts/simvla/simvla_data_generator.py"
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                   and n.name in {"_SubtaskDialogMixin", "SubtaskDialog"}]
    calls = []
    class FakeDialog:
        def __init__(self, parent, title):
            calls.append((parent, title))
    scope = {"_load_gui": lambda: calls.append("load"),
             "simpledialog": SimpleNamespace(Dialog=FakeDialog)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    dialog = scope["SubtaskDialog"]("parent", "title", {"task": "N_s"})
    assert isinstance(dialog, FakeDialog)
    assert dialog.cache == {"task": "N_s"}
    assert dialog.result is None
    assert calls == ["load", ("parent", "title")]
