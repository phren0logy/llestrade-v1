from pathlib import Path

import pytest

from src.app.core.project_manager import ProjectManager, ProjectMetadata
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup, load_bulk_analysis_groups


@pytest.fixture
def project_manager(tmp_path: Path) -> ProjectManager:
    pm = ProjectManager()
    metadata = ProjectMetadata(case_name="Test Case")
    pm.create_project(tmp_path, metadata)
    return pm


def test_bulk_analysis_group_save_and_load(project_manager: ProjectManager):
    group = BulkAnalysisGroup.create(
        name="Clinical Records",
        files=["medical/doc1.md"],
        directories=["medical"],
        provider_id="anthropic",
        model="claude-sonnet-4-5-20250929",
        system_prompt_path="resources/prompts/system.md",
        user_prompt_path="resources/prompts/user.md",
    )

    saved = project_manager.save_bulk_analysis_group(group)
    assert saved.group_id in {g.group_id for g in project_manager.list_bulk_analysis_groups()}

    reloaded = load_bulk_analysis_groups(project_manager.project_dir)
    assert len(reloaded) == 1
    loaded_group = reloaded[0]
    assert loaded_group.slug == saved.slug
    assert loaded_group.files == ["medical/doc1.md"]
    assert loaded_group.directories == ["medical"]
    assert loaded_group.system_prompt_path == "resources/prompts/system.md"
    assert loaded_group.user_prompt_path == "resources/prompts/user.md"
    assert loaded_group.provider_id == "anthropic"


def test_bulk_analysis_group_slug_uniqueness(project_manager: ProjectManager):
    g1 = project_manager.save_bulk_analysis_group(BulkAnalysisGroup.create(name="Legal Docs", directories=["legal"]))
    g2 = project_manager.save_bulk_analysis_group(BulkAnalysisGroup.create(name="Legal Docs", directories=["legal/case"]))

    assert g1.slug != g2.slug
    assert g2.slug.startswith(g1.slug.split("-")[0])


def test_bulk_analysis_group_delete(project_manager: ProjectManager):
    group = project_manager.save_bulk_analysis_group(BulkAnalysisGroup.create(name="To Delete"))
    assert project_manager.delete_bulk_analysis_group(group.group_id)
    assert project_manager.list_bulk_analysis_groups() == []
    bulk_dir = project_manager.project_dir / "bulk_analysis"
    assert not (bulk_dir / group.slug).exists()


def test_refresh_bulk_analysis_groups_updates_cache(project_manager: ProjectManager):
    project_manager.save_bulk_analysis_group(BulkAnalysisGroup.create(name="Refresh Test"))
    project_manager.bulk_analysis_groups = {}
    groups = project_manager.refresh_bulk_analysis_groups()
    assert len(groups) == 1
