from importlib.resources import files

import yaml
from typer.testing import CliRunner

import surfcensus
from surfcensus.cli import app


def test_version_flag() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == surfcensus.__version__


def test_default_protocol_ships_and_is_consistent() -> None:
    protocol = yaml.safe_load(files("surfcensus").joinpath("data/protocol.yaml").read_text())
    keys = [k for c in protocol["criteria"] for k in c["keys"]]
    mark_keys = [c["mark_key"] for c in protocol["criteria"]] + [protocol["other_mark_key"]]
    assert len(keys) == len(set(keys)), "rating keys must be unique"
    assert len(mark_keys) == len(set(mark_keys)), "mark keys must be unique"
    assert not set(keys) & set(mark_keys)
    assert all(len(c["keys"]) == len(protocol["scale"]) for c in protocol["criteria"])
