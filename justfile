set shell := ["bash", "-cu"]

default:
    @just --list

setup:
    uv sync --all-extras --group dev

test:
    uv run pytest

test-network:
    uv run pytest -m network

lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyrefly check

fmt:
    uv run ruff format .
    uv run ruff check --fix .

build:
    uv build
