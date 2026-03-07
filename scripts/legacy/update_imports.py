#!/usr/bin/env python3
"""
Update imports after directory reorganization.
This script updates import statements to use the new src structure.
"""

import os
import re
from pathlib import Path


def update_imports_in_file(file_path: Path, dry_run=True):
    """Update imports in a single Python file."""
    if not file_path.suffix == '.py':
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    original_content = content
    changes = []
    
    # Define import mappings
    mappings = [
        # UI imports - for files outside src/legacy
        (r'from ui\.', 'from src.legacy.ui.'),
        (r'import ui\.', 'import src.legacy.ui.'),
        
        # LLM imports - for files outside src/common
        (r'from llm\.', 'from src.common.llm.'),
        (r'import llm\.', 'import src.common.llm.'),
        
        # For files within src/legacy that import from ui
        (r'from \.\.ui\.', 'from ..ui.'),  # Keep relative imports within legacy
        
        # For files within src/common that import from llm
        (r'from \.\.llm\.', 'from ..llm.'),  # Keep relative imports within common
    ]
    
    # Check if file is in src/legacy or src/common to avoid double-prefixing
    file_str = str(file_path)
    is_in_legacy = 'src/legacy/' in file_str
    is_in_common = 'src/common/' in file_str
    
    for pattern, replacement in mappings:
        # Skip certain replacements for files already in the new structure
        if is_in_legacy and 'src.legacy.ui' in replacement:
            continue
        if is_in_common and 'src.common.llm' in replacement:
            continue
            
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes.append(f"  {pattern} -> {replacement} ({len(matches)} occurrences)")
    
    if content != original_content:
        if dry_run:
            print(f"\nWould update {file_path}:")
            for change in changes:
                print(change)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"\nUpdated {file_path}:")
            for change in changes:
                print(change)
        return True
    return False


def find_python_files(root_dir: Path, exclude_dirs=None):
    """Find all Python files in the project."""
    if exclude_dirs is None:
        exclude_dirs = {'.venv', '__pycache__', '.git', 'build', 'dist', '.pytest_cache'}
    
    python_files = []
    for path in root_dir.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in path.parts for excluded in exclude_dirs):
            continue
        python_files.append(path)
    return python_files


def main():
    """Main function to update all imports."""
    project_root = Path(__file__).parent.parent
    
    print("Import Update Script")
    print("=" * 50)
    print(f"Project root: {project_root}")
    
    # Find all Python files
    python_files = find_python_files(project_root)
    print(f"\nFound {len(python_files)} Python files")
    
    # First, do a dry run
    print("\nDry run - showing what would be changed:")
    print("-" * 50)
    
    files_to_update = []
    for file_path in python_files:
        if update_imports_in_file(file_path, dry_run=True):
            files_to_update.append(file_path)
    
    if not files_to_update:
        print("\nNo files need updating!")
        return
    
    print(f"\n{len(files_to_update)} files would be updated")
    
    # Auto-confirm for non-interactive mode
    import sys
    if sys.stdin.isatty():
        response = input("\nProceed with updates? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    else:
        print("\nProceeding with updates (non-interactive mode)...")
    
    # Perform actual updates
    print("\nUpdating files...")
    print("-" * 50)
    
    success_count = 0
    for file_path in files_to_update:
        if update_imports_in_file(file_path, dry_run=False):
            success_count += 1
    
    print(f"\nSuccessfully updated {success_count} files")
    
    # Create __init__.py files if needed
    init_files = [
        project_root / 'src' / '__init__.py',
        project_root / 'src' / 'legacy' / '__init__.py',
        project_root / 'src' / 'common' / '__init__.py',
        project_root / 'src' / 'new' / '__init__.py',
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            init_file.touch()
            print(f"Created {init_file}")


if __name__ == "__main__":
    main()