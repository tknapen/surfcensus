set shell := ["bash", "-cu"]

default:
    @just --list

setup:
    uv sync --all-extras --group dev

test:
    uv run pytest

test-network:
    uv run pytest -m network

# the rater app in a real browser: SURFCENSUS_BROWSER names its executable, else Chrome is used
test-browser:
    uv run --group browser pytest -m browser

lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyrefly check

fmt:
    uv run ruff format .
    uv run ruff check --fix .

build:
    uv build
