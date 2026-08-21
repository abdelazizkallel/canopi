"""
The paper and example scripts need a GPU and hours of compute, so CI cannot
run them. It can still prove that every symbol they import from canopi exists
— which is what would break if the package were reorganised or a function
renamed.
"""
import ast
import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((REPO / "paper").glob("*.py")) + sorted((REPO / "examples").glob("*.py"))

assert SCRIPTS, "no scripts found to check"


def _canopi_imports(path):
    """Yield (module, symbol_or_None) for every canopi import in the file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "canopi":
                for alias in node.names:
                    yield node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "canopi":
                    yield alias.name, None


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_imports_resolve(script):
    imports = list(_canopi_imports(script))
    if not imports:
        # e.g. run_all.py, which only shells out to the other scripts
        pytest.skip("no canopi imports")
    for module, symbol in imports:
        mod = importlib.import_module(module)
        if symbol is not None:
            assert hasattr(mod, symbol), f"{module}.{symbol} missing ({script.name})"


def test_most_scripts_actually_import_canopi():
    """Guard against the extraction above silently finding nothing."""
    with_imports = [s for s in SCRIPTS if list(_canopi_imports(s))]
    assert len(with_imports) >= len(SCRIPTS) - 1
    assert len(with_imports) >= 10
