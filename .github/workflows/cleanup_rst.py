import os
from pathlib import Path

# Paths relative to the project root
docs_path = Path('docs')  # Path to .rst files
src_path = Path('source')    # Path to your Python modules

# Gather .rst filenames without the extension
rst_files = {f.stem for f in docs_path.glob('**/*.rst') if f.stem != 'index'}

# Gather Python module names based on .py files
module_names = {f.stem for f in src_path.glob('**/*.py')}

# Identify orphaned .rst files (without a corresponding .py file)
orphaned_rst = rst_files - module_names

# Remove orphaned .rst files
for orphan in orphaned_rst:
    (docs_path / f"{orphan}.rst").unlink()

print(f"Removed orphaned .rst files: {orphaned_rst}")
