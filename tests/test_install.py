"""The quick start as a user gets it (decision D33): the wheel, installed into an environment
of its own, and ``surfcensus review PATH`` in it up to a stored rating.

The test builds the wheel with ``uv build``, makes an environment outside the repository
with ``uv venv`` on the Python of the test run, and installs the wheel there with its
dependencies from the package index. Nothing but the wheel is named: no extra, no lock
file, so the packages are the ones a ``pip install`` resolves on the day. The commands then
run from another directory with ``PYTHONPATH`` unset, so that nothing is imported from the
source tree.

It needs ``uv`` and the package index (about 230 MB of wheels where nothing is cached), so
it is marked ``install`` and is not part of the default run: ``just test-install``, or
``uv run pytest -m install -s`` to see the times. CI has a job for it.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import httpx
import pytest
from synth import snapshot_tree, write_synthetic_subject  # pyrefly: ignore[missing-import]

pytestmark = [
    pytest.mark.install,
    pytest.mark.skipif(os.name == "nt", reason="signals and paths of a POSIX system"),
]

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "src" / "surfcensus"
LABELS = ("sub-01", "sub-02")
APP_DEPENDENCIES = ("fastapi", "uvicorn", "sqlmodel", "jinja2")

BUILD_S = 600
INSTALL_S = 1800
"""Time allowed for the install: every dependency may have to come from the index."""
COMMAND_S = 300
"""Time allowed for ``review`` to name the address, and for any other command."""

_NOT_INHERITED = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "FREESURFER_HOME",
    "SUBJECTS_DIR",
)


def environment() -> dict[str, str]:
    """The environment of the test run without what would lead a command back to the source
    tree, to the environment of the tests or to a FreeSurfer installation. What the package
    index needs (proxy, certificates) stays."""
    kept = {name: value for name, value in os.environ.items() if name not in _NOT_INHERITED}
    return kept | {"PYTHONUNBUFFERED": "1"}


def run(command: list[object], *, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def checked(command: list[object], *, cwd: Path, timeout: float) -> str:
    """Standard output of a command that has to succeed."""
    result = run(command, cwd=cwd, timeout=timeout)
    assert result.returncode == 0, f"{command}\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def uv() -> str:
    found = shutil.which("uv")
    # whoever asked for this test wanted it to run: no skip
    assert found, "the install test builds and installs with uv, which is not on the PATH"
    return found


@pytest.fixture(scope="module")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory outside the repository: commands run from here."""
    path = tmp_path_factory.mktemp("install").resolve()
    assert REPOSITORY not in path.parents and path != REPOSITORY
    return path


@pytest.fixture(scope="module")
def wheel(uv: str, scratch: Path) -> Path:
    started = time.perf_counter()
    checked(
        [uv, "build", "--wheel", "--out-dir", scratch / "dist", REPOSITORY],
        cwd=scratch,
        timeout=BUILD_S,
    )
    wheels = sorted((scratch / "dist").glob("*.whl"))
    assert len(wheels) == 1, wheels
    print(f"\ninstall test: wheel built in {time.perf_counter() - started:.1f} s")
    return wheels[0]


@dataclass(frozen=True)
class Installed:
    """An environment that holds the wheel and what the index gave for it."""

    python: Path
    command: Path
    site_packages: Path
    packages: dict[str, str]


@pytest.fixture(scope="module")
def installed(uv: str, scratch: Path, wheel: Path) -> Installed:
    env = scratch / "env"
    started = time.perf_counter()
    checked([uv, "venv", "--python", sys.executable, env], cwd=scratch, timeout=BUILD_S)
    python = env / "bin" / "python"
    checked([uv, "pip", "install", "--python", python, wheel], cwd=scratch, timeout=INSTALL_S)
    print(f"install test: environment made and filled in {time.perf_counter() - started:.1f} s")
    listed = checked(
        [uv, "pip", "list", "--python", python, "--format", "json"], cwd=scratch, timeout=BUILD_S
    )
    packages = {row["name"].lower().replace("_", "-"): row["version"] for row in json.loads(listed)}
    site = checked(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        cwd=scratch,
        timeout=COMMAND_S,
    )
    return Installed(python, env / "bin" / "surfcensus", Path(site.strip()).resolve(), packages)


# --- the wheel ------------------------------------------------------------------------------------


def source_files(*parts: str) -> set[str]:
    """The files under ``src/surfcensus/<parts>``, as the wheel names them."""
    root = PACKAGE.joinpath(*parts)
    return {
        path.relative_to(PACKAGE.parent).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_the_wheel_holds_the_app_and_the_protocol_and_nothing_else_beside_the_code(
    wheel: Path,
) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        static = source_files("app", "static")
        templates = source_files("app", "templates")
        # the pages and their scripts are there, byte for byte
        assert {Path(name).suffix for name in static} == {".js", ".css"} and len(static) >= 12
        assert {Path(name).name for name in templates} == {"base.html", "rate.html", "review.html"}
        for name in sorted(static | templates | {"surfcensus/data/protocol.yaml"}):
            assert name in names, f"{name} is not in the wheel"
            assert archive.read(name) == (PACKAGE.parent / name).read_bytes(), name
    # every module of the package, and nothing from beside it
    modules = {name for name in source_files() if name.endswith(".py")}
    assert {name for name in names if name.endswith(".py")} == modules
    info = {name for name in names if ".dist-info/" in name}
    assert {name.split("/", 1)[1] for name in info} == {
        "METADATA",
        "WHEEL",
        "RECORD",
        "entry_points.txt",
        "licenses/LICENSE",
    }
    for name in names - info:
        assert name.startswith("surfcensus/"), name
    for name in names:
        assert not {"tests", "prototypes", "docs", "workflow", "notebooks"} & set(name.split("/"))
    # What is not code: the app's pages, the protocol, and the note that says the shipped
    # reference is empty. Nothing fsaverage-derived ships (decision D22): no array, surface,
    # label or table. A reference that ships one day (milestone 6) is added here knowingly.
    data = {name for name in names - info if not name.endswith(".py")}
    assert data == static | templates | {
        "surfcensus/data/protocol.yaml",
        "surfcensus/data/reference/README.md",
    }


def test_the_wheel_requires_the_packages_of_the_app_and_keeps_the_extra_as_a_name(
    wheel: Path,
) -> None:
    with zipfile.ZipFile(wheel) as archive:
        (metadata,) = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        lines = archive.read(metadata).decode().splitlines()
    required = [line.removeprefix("Requires-Dist: ") for line in lines if line.startswith("Req")]
    always = {re.split(r"[ <>=!~\[;]", line)[0].lower() for line in required if ";" not in line}
    assert set(APP_DEPENDENCIES) <= always
    # `pip install "surfcensus[app]"` still installs: the extra exists and requires nothing
    assert "Provides-Extra: app" in lines
    assert not [line for line in required if re.search(r"extra\s*==\s*['\"]app['\"]", line)]
    version = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())["project"]["version"]
    assert f"Version: {version}" in lines


# --- the environment ------------------------------------------------------------------------------


def test_the_package_is_imported_from_the_environment_and_not_from_the_source_tree(
    installed: Installed, scratch: Path
) -> None:
    out = checked(
        [
            installed.python,
            "-c",
            "import sys, surfcensus, surfcensus.app.server as server\n"
            "print(surfcensus.__file__)\nprint(server.__file__)\n"
            "print(server.STATIC_DIR)\nprint(server.TEMPLATES_DIR)\n"
            "print(sys.prefix)\n",
        ],
        cwd=scratch,
        timeout=COMMAND_S,
    )
    *places, prefix = [Path(line).resolve() for line in out.splitlines()]
    assert len(places) == 4
    for place in places:
        assert installed.site_packages in place.parents, place
        assert REPOSITORY not in place.parents, place
    assert prefix == (scratch / "env").resolve()
    assert (places[2] / "js" / "rate.js").is_file() and (places[3] / "rate.html").is_file()


def test_one_install_brings_the_packages_of_the_app(
    uv: str, installed: Installed, wheel: Path, scratch: Path
) -> None:
    for package in (*APP_DEPENDENCIES, "sqlalchemy", "starlette", "h11"):
        assert package in installed.packages, f"{package} was not installed with the wheel"
    # the server names h11 and asyncio, and here it has nothing else to pick up
    assert not {"httptools", "uvloop"} & set(installed.packages)
    # the line the docs gave while the app was an extra adds nothing and warns of nothing
    again = run(
        [uv, "pip", "install", "--python", installed.python, f"surfcensus[app] @ {wheel.as_uri()}"],
        cwd=scratch,
        timeout=INSTALL_S,
    )
    assert again.returncode == 0, again.stderr
    assert "extra" not in again.stderr.lower(), again.stderr
    listed = checked(
        [uv, "pip", "list", "--python", installed.python, "--format", "json"],
        cwd=scratch,
        timeout=BUILD_S,
    )
    after = {row["name"].lower().replace("_", "-"): row["version"] for row in json.loads(listed)}
    assert after == installed.packages


def test_the_installed_command_lists_its_commands(installed: Installed, scratch: Path) -> None:
    out = checked([installed.command, "--help"], cwd=scratch, timeout=COMMAND_S)
    # whatever the terminal: on CI (GITHUB_ACTIONS) Rich writes escape codes around every
    # word of a help page, and it always draws a frame around the list of commands
    words = re.sub(r"[\u2500-\u257f]", " ", re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", out))
    for command in ("review", "serve", "finalize", "export", "inventory", "metrics", "render"):
        assert re.search(rf"^\s*{command}\b", words, re.M), out
    version = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())["project"]["version"]
    assert checked([installed.command, "--version"], cwd=scratch, timeout=COMMAND_S).split() == [
        version
    ]


# --- review, from the command to a stored rating --------------------------------------------------


class Lines:
    """The lines of a stream, read by a thread, so that waiting for one has a time limit."""

    def __init__(self, stream: IO[str]) -> None:
        self.seen: list[str] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self._thread.start()

    def _read(self, stream: IO[str]) -> None:
        for line in stream:
            self._queue.put(line.rstrip("\n"))
        self._queue.put(None)

    def until(self, pattern: str, timeout: float) -> re.Match[str]:
        """The first line from here on that matches ``pattern`` as a whole."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            said = "\n".join(self.seen)
            assert left > 0, f"no line {pattern!r} within {timeout} s:\n{said}"
            try:
                line = self._queue.get(timeout=left)
            except queue.Empty:
                continue
            assert line is not None, f"the command ended before a line {pattern!r}:\n{said}"
            self.seen.append(line)
            found = re.fullmatch(pattern, line)
            if found:
                return found

    def rest(self, timeout: float) -> list[str]:
        """Every line up to the end of the stream."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            assert left > 0, "the stream did not end:\n" + "\n".join(self.seen)
            try:
                line = self._queue.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                return self.seen
            self.seen.append(line)


def _end_group(process: subprocess.Popen[str]) -> None:
    """Kill the command and every process it started: nothing outlives a failed test."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait(timeout=30)
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


@pytest.fixture(scope="module")
def study(scratch: Path) -> Path:
    """A subjects directory of two synthetic 1 mm subjects, which differ in their midline
    and so in their files."""
    data = scratch / "data" / "subjects"
    data.mkdir(parents=True)
    for position, label in enumerate(LABELS):
        write_synthetic_subject(data, name=label, subdivisions=4, midline_x=-2.0 + 3.0 * position)
    return data


@pytest.fixture
def review(installed: Installed, study: Path, scratch: Path) -> Iterator[subprocess.Popen[str]]:
    elsewhere = scratch / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    process = subprocess.Popen(
        [str(installed.command), "review", str(study), "--no-browser", "--port", "0"],
        cwd=elsewhere,
        env=environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # a group of its own, so that the renderer's worker processes end with it
        start_new_session=True,
    )
    try:
        yield process
    finally:
        _end_group(process)


def test_review_in_the_installed_environment_serves_the_page_and_stores_a_rating(
    installed: Installed, review: subprocess.Popen[str], study: Path, scratch: Path
) -> None:
    before = snapshot_tree(study)
    assert review.stderr is not None and review.stdout is not None
    started = time.perf_counter()
    err = Lines(review.stderr)
    url = err.until(r"Rating app: (http://127\.0\.0\.1:\d+/) \(Ctrl-C to stop\)", COMMAND_S)[1]
    print(f"install test: {time.perf_counter() - started:.1f} s from the command to the address")
    run_dir = study.parent / "surfcensus-run"
    assert err.seen[0] == f"Run directory: {run_dir} (new)"
    assert [line for line in err.seen if line.startswith("Found 2 subjects")], err.seen

    # the test run may sit behind a proxy for the index; the app is on this machine
    with httpx.Client(base_url=url, trust_env=False, timeout=60) as client:
        page = client.get("/")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        scripts = re.findall(r'src="(/static/[^"?]+)', page.text)
        styles = re.findall(r'href="(/static/[^"?]+\.css)', page.text)
        assert scripts and styles, page.text
        for path in (*scripts, *styles):
            served = client.get(path)
            assert served.status_code == 200, path
            # the file of the installed package, not one of the source tree
            assert (
                served.content
                == installed.site_packages.joinpath(
                    "surfcensus", "app", *path.split("/")[1:]
                ).read_bytes()
            )

        session = client.get("/api/session").json()
        assert session["mode"] == "local" and session["rater"]["name"] == "local"
        assert session["app_version"] == installed.packages["surfcensus"]
        assert [criterion["id"] for criterion in session["protocol"]["criteria"]] == ["A", "B", "C"]

        offered = client.get("/api/next").json()
        assert offered["status"] == "assignment", offered
        assert offered["progress"]["n_subjects"] == 2
        shown = offered["assignment"]
        gray = shown["render"]["tiles"]["gray"].format(panel="sagittal", index=0)
        tile = client.get(shown["media_base"] + gray, params={"v": shown["render_key"]})
        assert tile.status_code == 200 and tile.headers["content-type"] == "image/webp"
        assert tile.content[:4] == b"RIFF" and tile.content[8:12] == b"WEBP"
        assert tile.content == (run_dir / "media" / shown["subject_uid"] / gray).read_bytes()

        posted = client.post(
            "/api/rating",
            json={
                "assignment_id": shown["assignment_id"],
                "protocol_version": session["protocol"]["version"],
                "values": {"A": 0, "B": 1, "C": 0},
                "client_ts": "2026-11-14T10:00:00.000Z",
                "ms_on_subject": 1500,
            },
        )
        assert posted.status_code == 200, posted.text
        answer = posted.json()
        assert answer["saved"]["revision"] == 1
        assert answer["next"]["status"] in ("assignment", "rendering")
        # the second subject is rendered behind the app
        deadline = time.monotonic() + COMMAND_S
        while client.get("/api/next").json()["progress"]["rendered"] < 2:
            assert time.monotonic() < deadline, "the second subject was never rendered"
            time.sleep(0.2)

    review.send_signal(signal.SIGTERM)
    lines = err.rest(COMMAND_S)
    assert review.wait(timeout=COMMAND_S) == 0, "\n".join(lines)
    assert review.stdout.read() == ""
    assert lines[-1] == (
        "1 rated, 1 to go. Run the same command to continue. "
        f"Export: surfcensus export --run-dir {run_dir}"
    )
    assert "Traceback" not in "\n".join(lines)
    with pytest.raises(httpx.TransportError), httpx.Client(trust_env=False) as client:
        client.get(url + "healthz")

    # the command it names works in the same environment, and the rating is in its tables
    exported = run(
        [installed.command, "export", "--run-dir", run_dir], cwd=scratch, timeout=COMMAND_S
    )
    assert exported.returncode == 0, exported.stderr
    with (run_dir / "export" / "ratings.csv").open(newline="") as handle:
        (rating,) = list(csv.DictReader(handle))
    assert (rating["subject_uid"], rating["rater"]) == (shown["subject_uid"], "local")
    assert (rating["A"], rating["B"], rating["C"]) == ("0", "1", "0")
    # and the subjects are as they were: a source is only read
    assert snapshot_tree(study) == before
    assert not list(scratch.glob("elsewhere/*"))
