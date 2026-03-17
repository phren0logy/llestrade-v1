from __future__ import annotations

from pathlib import Path

from src.app.ui.workspace.controllers import reports_io


def test_default_prompt_path_prefers_synced_bundled_prompt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundled_dir = tmp_path / "bundled"
    repo_dir = tmp_path / "repo"
    bundled_reports = bundled_dir / "reports"
    repo_reports = repo_dir / "reports"
    bundled_reports.mkdir(parents=True)
    repo_reports.mkdir(parents=True)

    bundled_prompt = bundled_reports / "default_generation_user.md"
    bundled_prompt.write_text("bundled", encoding="utf-8")
    (repo_reports / "default_generation_user.md").write_text("repo", encoding="utf-8")

    monkeypatch.setattr(reports_io, "get_bundled_dir", lambda: bundled_dir)
    monkeypatch.setattr(reports_io, "get_repo_prompts_dir", lambda: repo_dir)

    assert reports_io.default_prompt_path("default_generation_user.md") == str(bundled_prompt)


def test_read_prompt_file_prefers_synced_bundled_prompt_before_repo_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    bundled_dir = tmp_path / "bundled"
    custom_dir = tmp_path / "custom"
    repo_dir = tmp_path / "repo"
    project_dir.mkdir()
    bundled_dir.mkdir()
    custom_dir.mkdir()
    repo_dir.mkdir()

    relative = Path("reports/default_generation_user.md")
    bundled_prompt = bundled_dir / relative
    repo_prompt = repo_dir / relative
    bundled_prompt.parent.mkdir(parents=True)
    repo_prompt.parent.mkdir(parents=True)
    bundled_prompt.write_text("bundled", encoding="utf-8")
    repo_prompt.write_text("repo", encoding="utf-8")

    monkeypatch.setattr(reports_io, "get_custom_dir", lambda: custom_dir)
    monkeypatch.setattr(reports_io, "get_bundled_dir", lambda: bundled_dir)
    monkeypatch.setattr(reports_io, "get_repo_prompts_dir", lambda: repo_dir)

    assert reports_io.read_prompt_file(str(relative), project_dir) == "bundled"


def test_normalize_prompt_path_rewrites_repo_prompt_to_synced_bundled_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundled_dir = tmp_path / "bundled"
    custom_dir = tmp_path / "custom"
    repo_dir = tmp_path / "repo"
    bundled_dir.mkdir()
    custom_dir.mkdir()
    repo_dir.mkdir()

    relative = Path("reports/default_generation_user.md")
    bundled_prompt = bundled_dir / relative
    repo_prompt = repo_dir / relative
    bundled_prompt.parent.mkdir(parents=True)
    repo_prompt.parent.mkdir(parents=True)
    bundled_prompt.write_text("bundled", encoding="utf-8")
    repo_prompt.write_text("repo", encoding="utf-8")

    monkeypatch.setattr(reports_io, "get_custom_dir", lambda: custom_dir)
    monkeypatch.setattr(reports_io, "get_bundled_dir", lambda: bundled_dir)
    monkeypatch.setattr(reports_io, "get_repo_prompts_dir", lambda: repo_dir)

    assert reports_io.normalize_prompt_path(str(repo_prompt)) == str(bundled_prompt)
