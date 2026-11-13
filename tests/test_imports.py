"""Import structure of the package: no cycles, a light top level, lazy entry points.

Every module is imported in an interpreter of its own, so an import cycle fails whatever
order the test session happened to import things in.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import surfcensus

PACKAGE_DIR = Path(surfcensus.__file__).parent


def module_names() -> list[str]:
    names = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        parts = path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def run_python(code: str) -> str:
    """Standard output of ``code`` in a fresh interpreter; fails the test if it fails."""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


# prints the modules that `statement` loaded and that are neither in the standard library
# nor allowed
NEW_MODULES = """
import sys
before = set(sys.modules)
{statement}
allowed = {allowed!r}
new = sorted(
    name for name in set(sys.modules) - before
    if name.split(".")[0] not in sys.stdlib_module_names
    and name not in allowed and name.split(".")[0] not in allowed
)
print(" ".join(new))
"""


def loaded_beyond(statement: str, *allowed: str) -> list[str]:
    return run_python(NEW_MODULES.format(statement=statement, allowed=set(allowed))).split()


def test_the_modules_of_this_milestone_are_found() -> None:
    names = module_names()
    for expected in (
        "surfcensus",
        "surfcensus._io",
        "surfcensus.log",
        "surfcensus.records",
        "surfcensus.schemas",
        "surfcensus.privacy",
        "surfcensus.config",
        "surfcensus.store",
        "surfcensus.run",
        "surfcensus.batches",
        "surfcensus.sources",
        "surfcensus.sources.base",
        "surfcensus.sources.patterns",
        "surfcensus.sources.local",
        "surfcensus.inventory",
        "surfcensus.inventory.subjects",
        "surfcensus.inventory.provenance",
        "surfcensus.render.batch",
        "surfcensus.render.api",
        "surfcensus.cli",
    ):
        assert expected in names


APP_EXTRA = ("fastapi", "uvicorn", "sqlmodel", "sqlalchemy", "jinja2")
"""What the ``app`` extra installs. Only modules under ``app/`` may need it."""
CORE_APP_MODULES = ("surfcensus.app", "surfcensus.app.schemas")
"""The modules under ``app/`` that a core install imports."""


def needs_app_extra(module: str) -> bool:
    return module.startswith("surfcensus.app.") and module not in CORE_APP_MODULES


@pytest.mark.parametrize("module", module_names())
def test_every_module_imports_in_a_fresh_interpreter(module: str) -> None:
    if needs_app_extra(module):
        pytest.importorskip("fastapi")
        pytest.importorskip("sqlmodel")
    run_python(f"import {module}")


def test_the_modules_of_milestone_4_are_found() -> None:
    names = module_names()
    for expected in (
        "surfcensus.protocol",
        "surfcensus.calibration",
        "surfcensus.review",
        "surfcensus.app",
        "surfcensus.app.schemas",
        "surfcensus.app.models",
        "surfcensus.app.db",
        "surfcensus.app.marks",
        "surfcensus.app.queue",
        "surfcensus.app.events",
        "surfcensus.app.finalize",
        "surfcensus.app.server",
        "surfcensus.export",
        "surfcensus.export.tables",
    ):
        assert expected in names


BLOCK_APP_EXTRA = "import sys\n" + "".join(f"sys.modules[{name!r}] = None\n" for name in APP_EXTRA)


@pytest.mark.parametrize("module", [name for name in module_names() if not needs_app_extra(name)])
def test_only_the_app_needs_the_app_extra(module: str) -> None:
    # `surfcensus export`, the protocol, the calibration set and the shapes of the API work
    # on a core install: a None in sys.modules makes the import of that name fail
    out = run_python(
        BLOCK_APP_EXTRA + f"import {module}\n"
        "print(sorted(n for n in sys.modules if n.split('.')[0] in "
        f"{APP_EXTRA!r} and sys.modules[n] is not None))\n"
    )
    assert out.strip() == "[]"


def test_the_app_package_names_its_database_without_importing_anything() -> None:
    assert loaded_beyond("import surfcensus.app", "surfcensus") == []
    assert loaded_beyond("import surfcensus.export", "surfcensus") == []
    out = run_python(
        "import sys, surfcensus.app, surfcensus.export\n"
        "print(sorted(m for m in sys.modules if m.startswith('surfcensus')))\n"
        "print(surfcensus.app.APP_DB_FILE, len(surfcensus.app.APP_TABLES))\n"
    )
    assert out.splitlines() == [
        "['surfcensus', 'surfcensus.app', 'surfcensus.export']",
        "app.sqlite 9",
    ]


def test_import_surfcensus_loads_the_standard_library_only() -> None:
    assert loaded_beyond("import surfcensus", "surfcensus") == []


def test_the_top_level_names_are_imported_on_first_use() -> None:
    out = run_python(
        "import sys, surfcensus as sc\n"
        "assert 'surfcensus.render' not in sys.modules and 'surfcensus.run' not in sys.modules\n"
        "print(sc.render.__name__, sc.sources.__name__, sc.Run.__module__, sc.Run.__name__)\n"
        "print(sc.render.RenderConfig.__name__, 'matplotlib' in sys.modules)\n"
    )
    assert out.split() == [
        "surfcensus.render",
        "surfcensus.sources",
        "surfcensus.run",
        "Run",
        "RenderConfig",
        "False",
    ]
    with pytest.raises(AttributeError, match="no attribute 'nothing'"):
        _ = surfcensus.nothing  # pyrefly: ignore[missing-attribute]


def test_the_sources_package_imports_nothing_until_it_is_used() -> None:
    # config.py imports sources/base.py, which runs sources/__init__.py first: an import of
    # config from there would be a cycle
    assert loaded_beyond("import surfcensus.sources", "surfcensus") == []
    assert run_python(
        "import sys, surfcensus.sources\n"
        "print('surfcensus.config' in sys.modules, 'surfcensus.sources.local' in sys.modules)\n"
    ).split() == ["False", "False"]
    with pytest.raises(AttributeError, match="no attribute 'remote'"):
        _ = surfcensus.sources.remote  # pyrefly: ignore[missing-attribute]


LOCAL_IS_CALLABLE = """
{first}
import surfcensus as sc
import surfcensus.sources.local as module
from surfcensus.sources.local import LocalSource, local
assert sc.sources.local is module, "the package attribute is the module"
assert callable(sc.sources.local), "and the module is callable"
assert type(module).__call__.__name__ == "__call__" and callable(local)
assert sc.sources.LocalSource is LocalSource
assert module.local is local and module.detect_layout is not None
print("ok")
"""


@pytest.mark.parametrize(
    "first",
    [
        "",
        "import surfcensus.sources.local",
        "from surfcensus.sources import local",
        "import surfcensus.run",
        "import surfcensus; surfcensus.sources.local",
    ],
)
def test_sources_local_is_callable_whatever_was_imported_first(first: str) -> None:
    # `surfcensus.sources.local` names a module and the function that makes a source
    assert run_python(LOCAL_IS_CALLABLE.format(first=first)).strip() == "ok"


def test_the_shared_modules_stay_light() -> None:
    package = ("surfcensus",)
    # records.py and sources/base.py: the standard library only
    assert loaded_beyond("import surfcensus.sources.base", *package) == []
    assert loaded_beyond("import surfcensus.records", *package) == []
    assert run_python(
        "import sys, surfcensus.records\n"
        "print(sorted(m for m in sys.modules if m.startswith('surfcensus')))\n"
    ).strip() == str(
        ["surfcensus", "surfcensus.records", "surfcensus.sources", "surfcensus.sources.base"]
    )
    # _io.py: numpy on top of that; log.py and privacy.py nothing heavier
    assert loaded_beyond("import surfcensus._io", *package, "numpy") == []
    assert loaded_beyond("import surfcensus.log", *package) == []
    assert loaded_beyond("import surfcensus.privacy", *package, "numpy") == []
    # none of the modules that every phase imports pulls in the renderer's plotting
    out = run_python(
        "import sys\n"
        "import surfcensus.config, surfcensus.store, surfcensus.run, surfcensus.batches\n"
        "import surfcensus.inventory.subjects, surfcensus.sources.local\n"
        "print('matplotlib' in sys.modules, 'surfcensus.render.api' in sys.modules)\n"
    )
    assert out.split() == ["False", "False"]


def test_sources_local_is_callable_under_the_type_checking_hook_of_the_tests() -> None:
    import surfcensus.sources.local as module

    assert surfcensus.sources.local is module and callable(module)
    assert surfcensus.sources.LocalSource is module.LocalSource


# ---------------------------------------------------------------------------------------
# the privacy boundary (decision D20): what code downstream of the inventory may name

DOWNSTREAM = ("metrics", "render", "app", "export", "select", "reference", "calibration.py")
"""Packages and modules that hold the public side of a run only. Those that do not exist
yet are checked from the day they do."""
NO_SIGNATURE = ("app", "export")


def private_names(source: str, *, signatures_too: bool, may_import_run: bool = False) -> list[str]:
    """What a module names of the private side of a run, each as ``line: what``."""
    found = []
    banned_literals = {"accession", "dataset_id"} | (
        {"input_signature"} if signatures_too else set()
    )
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            names: list[str] = []
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            names = [alias.name for alias in node.names]
        else:
            modules, names = [], []
        for module in modules:
            if module == "surfcensus.run" and not may_import_run:
                found.append(f"{node.lineno}: imports surfcensus.run")
            if module == "surfcensus" and "run" in names and not may_import_run:
                found.append(f"{node.lineno}: imports surfcensus.run")
            if module == "surfcensus.store" and "Catalog" in names:
                found.append(f"{node.lineno}: imports the private Catalog")
            if module.endswith("privacy") and {"reveal_table", "write_reveal"} & set(names):
                found.append(f"{node.lineno}: imports the reveal path")
        if isinstance(node, ast.Attribute):
            if node.attr == "_catalog":
                found.append(f"{node.lineno}: reaches for _catalog")
            if node.attr == "Catalog" and isinstance(node.value, ast.Name):
                found.append(f"{node.lineno}: names the private Catalog")
            if node.attr in ("reveal_table", "write_reveal"):
                found.append(f"{node.lineno}: names the reveal path")
        if isinstance(node, ast.Constant) and node.value in banned_literals:
            found.append(f"{node.lineno}: the literal {node.value!r}")
    return found


def test_code_downstream_of_the_inventory_names_nothing_private() -> None:
    checked = 0
    for name in DOWNSTREAM:
        target = PACKAGE_DIR / name
        paths = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in paths:
            if not path.is_file():
                continue
            checked += 1
            found = private_names(path.read_text(), signatures_too=name in NO_SIGNATURE)
            assert found == [], f"{path.relative_to(PACKAGE_DIR)}: {found}"
    assert checked >= 9 + 11  # the render package, and app/, export/ and calibration.py


@pytest.mark.parametrize(
    ("source", "what"),
    [
        ("from surfcensus.store import Catalog", "imports the private Catalog"),
        ("from surfcensus.store import PublicCatalog, Catalog as C", "imports the private Catalog"),
        ("import surfcensus.run", "imports surfcensus.run"),
        ("from surfcensus.run import Run", "imports surfcensus.run"),
        ("from surfcensus import run", "imports surfcensus.run"),
        ("from surfcensus.privacy import reveal_table", "imports the reveal path"),
        ("from surfcensus import store\nstore.Catalog('x')", "names the private Catalog"),
        ("def f(run):\n    return run._catalog", "reaches for _catalog"),
        ("COLUMNS = ['subject_uid', 'accession']", "the literal 'accession'"),
        ("row['dataset_id']", "the literal 'dataset_id'"),
    ],
)
def test_the_boundary_check_finds_what_it_is_for(source: str, what: str) -> None:
    assert [text.split(": ", 1)[1] for text in private_names(source, signatures_too=False)] == [
        what
    ]


def test_the_boundary_check_knows_its_exemptions() -> None:
    signature = "frame['input_signature']"
    assert private_names(signature, signatures_too=False) == []
    assert private_names(signature, signatures_too=True) == ["1: the literal 'input_signature'"]
    public = "from surfcensus.store import PublicCatalog\nfrom surfcensus.records import RunContext"
    assert private_names(public, signatures_too=True) == []
    # review.py is a user of the public API and may open a run; nothing else is waived for it
    opens = "from surfcensus.run import Run\nRun('x')._catalog"
    assert private_names(opens, signatures_too=False, may_import_run=True) == [
        "2: reaches for _catalog"
    ]
