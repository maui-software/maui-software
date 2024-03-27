import os
from pathlib import Path

# Define the path to the Sphinx 'source' directory and the 'generated' subdirectory
source_path = Path('docs/source')
generated_path = source_path / 'generated'

# Path to your Python modules, adjust as needed. Assuming 'maui' is at the project root.
src_path = Path('maui')

# Gather .rst filenames (without the extension) in the 'generated' folder
rst_files = {f.stem for f in generated_path.glob('*.rst') if f.stem != 'index'}

# Gather Python module names. Since modules are in subdirectories, include the path relative to 'maui'
module_names = set()
for f in src_path.glob('**/*.py'):
    # Create a module path relative to 'src_path', replacing separators with '.'
    # This assumes your module structure mirrors your package structure
    module_path = f.relative_to(src_path).with_suffix('')
    module_name = str(module_path).replace(os.sep, '.')
    module_names.add(module_name)

# Identify orphaned .rst files (without a corresponding .py file)
orphaned_rst = rst_files - module_names

# Remove orphaned .rst files
for orphan in orphaned_rst:
    (generated_path / f"{orphan}.rst").unlink()

print(f"Removed orphaned .rst files: {orphaned_rst}")
