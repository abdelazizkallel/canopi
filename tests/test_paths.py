"""Path resolution must be environment-driven, never machine-specific."""
import pytest

import canopi.paths
from canopi.paths import figure_dir, scene_dir


def _isolate(tmp_path, monkeypatch):
    """Point every scene_dir() search location at an empty temp tree."""
    monkeypatch.setenv("CANOPI_SCENE_DIR", str(tmp_path / "nope"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(canopi.paths, "REPO_ROOT", tmp_path / "norepo")


def test_figure_dir_honours_env(tmp_path, monkeypatch):
    target = tmp_path / "figs"
    monkeypatch.setenv("CANOPI_FIG_DIR", str(target))
    d = figure_dir()
    assert d == target and d.is_dir()


def test_scene_dir_found(tmp_path, monkeypatch):
    d = tmp_path / "JBS_treelocations"
    d.mkdir()
    (d / "ACPL_treelocations.dat").write_text("0.0 0.0 1.0\n")
    monkeypatch.setenv("CANOPI_SCENE_DIR", str(d))
    assert scene_dir() == str(d)


def test_scene_dir_missing_warns_and_returns_none(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        assert scene_dir() is None


def test_scene_dir_required_raises(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        scene_dir(required=True)
