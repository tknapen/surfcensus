"""Import structure of the package: no cycles, a light top level, lazy entry points.

Every module is imported in an interpreter of its own, so an import cycle fails whatever
order the test session happened to import things in.
"""

from __future__ import annotations

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


@pytest.mark.parametrize("module", module_names())
def test_every_module_imports_in_a_fresh_interpreter(module: str) -> None:
    run_python(f"import {module}")


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
