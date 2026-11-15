"""The subject page of a participant whose sessions lie on two rows of the subjects page.

A session whose FreeSurfer subject is session-wise (``sub-01_ses-1``) is on that subject's row,
and the other sessions of the participant on the row of ``sub-01``: the rule of "The unit and the
navigation" in docs/epi_interface.md, which fMRIPrep's ``--subject-anatomical-reference
sessionwise`` gives. The review of step 3 (F1) found the page saying "3 of 2 sessions" there,
because it counted the sessions on the page and not those of the stream. The folder is
``write_review_folder(extra_fs=("sub-01_ses-1",))`` with a third session of ``sub-01`` copied
from its second, inventoried by the real phases.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from synth_func import write_review_folder  # pyrefly: ignore[missing-import]
from typer.testing import CliRunner

from surfcensus.app import epi_schemas as shapes
from surfcensus.app.server import STATIC_DIR, create_app
from surfcensus.cli import app as cli_app
from surfcensus.store import PublicCatalog

NODE = shutil.which("node")


def add_third_session(root: Path) -> None:
    """``ses-3`` of ``sub-01``: every file of ``ses-2`` outside the FreeSurfer subjects, copied
    under the new name, with the name changed inside the small text files."""
    for path in sorted(root.rglob("*")):
        if "freesurfer" in path.parts or not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if "ses-2" not in rel or "sub-01" not in rel:
            continue
        new = root / rel.replace("ses-2", "ses-3")
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, new)
        if new.suffix in (".html", ".json", ".txt", ".tsv"):
            new.write_text(new.read_text().replace("ses-2", "ses-3"))


@pytest.fixture(scope="module")
def split_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("split")
    folder = write_review_folder(tmp / "data", extra_fs=("sub-01_ses-1",))
    add_third_session(folder.raw_root)
    run_dir = tmp / "run"
    result = CliRunner().invoke(
        cli_app, ["inventory", "--run-dir", str(run_dir), "--source", str(folder.raw_root)]
    )
    assert result.exit_code in (0, 1), result.output
    return run_dir


def pages(run_dir: Path) -> dict[str, dict[str, Any]]:
    """The subject pages of the rows that hold sessions of ``sub-01``, by row label."""
    with TestClient(create_app(run_dir), base_url="http://127.0.0.1") as client:
        rows = client.get("/api/subjects").json()["subjects"]
        out = {}
        for row in rows:
            body = client.get(f"/api/subjects/{row['key']}")
            assert body.status_code == 200, body.text
            page = body.json()
            shapes.SubjectPageOut.model_validate(page)
            if any(s["participant"] == "sub-01" for s in page["sessions"]):
                out[row["label"]] = page
        return out


def test_both_rows_name_the_three_sessions_of_the_stream(split_run: Path) -> None:
    got = pages(split_run)
    assert set(got) == {"sub-01", "sub-01_ses-1"}
    places = {
        (label, s["label"]): (s["stream_order"], s["stream_sessions"])
        for label, page in got.items()
        for s in page["sessions"]
    }
    assert places == {
        ("sub-01_ses-1", "1"): (0, 3),
        ("sub-01", "2"): (1, 3),
        ("sub-01", "3"): (2, 3),
    }
    participant = got["sub-01"]["sessions"][0]["participant_uid"]
    with PublicCatalog.open(split_run) as public:
        frame = public.func_sessions()
    assert sum(str(p) == participant for p in frame["participant_uid"]) == 3


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_page_says_the_place_in_the_stream_and_how_many_are_on_it(
    split_run: Path, tmp_path: Path
) -> None:
    got = pages(split_run)
    shutil.copytree(STATIC_DIR / "js", tmp_path / "js")
    (tmp_path / "package.json").write_text('{"type": "module"}\n', encoding="utf-8")
    data = [
        [s, sum(t["participant_uid"] == s["participant_uid"] for t in page["sessions"])]
        for label in ("sub-01_ses-1", "sub-01")
        for page in [got[label]]
        for s in page["sessions"]
    ]
    (tmp_path / "input.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "main.mjs").write_text(
        'import { readFileSync } from "node:fs";\n'
        'import { streamPlace } from "./js/subjects.js";\n'
        'const url = new URL("./input.json", import.meta.url);\n'
        'const input = JSON.parse(readFileSync(url, "utf8"));\n'
        "console.log(JSON.stringify(input.map(([s, n]) => streamPlace(s, n))));\n",
        encoding="utf-8",
    )
    assert NODE is not None
    done = subprocess.run(
        [NODE, str(tmp_path / "main.mjs")], capture_output=True, text=True, check=True
    )
    order = "sessions ordered by session label"
    assert json.loads(done.stdout) == [
        f"session 1 of 3 in this participant's stream; 1 of them is on this page; {order}",
        f"session 2 of 3 in this participant's stream; 2 of them are on this page; {order}",
        f"session 3 of 3 in this participant's stream; 2 of them are on this page; {order}",
    ]
    # a page that holds the whole stream, and a stream of one session
    one = dict(data[1][0], stream_sessions=1, stream_order=0)
    (tmp_path / "input.json").write_text(json.dumps([[data[1][0], 3], [one, 1]]), encoding="utf-8")
    done = subprocess.run(
        [NODE, str(tmp_path / "main.mjs")], capture_output=True, text=True, check=True
    )
    assert json.loads(done.stdout) == [f"session 2 of 3; {order}", f"the only session; {order}"]
