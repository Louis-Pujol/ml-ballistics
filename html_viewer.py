import os
import pathlib
import shutil
import sys

from trame_vtk.tools.vtksz2html import HTML_VIEWER_PATH

static_path = pathlib.Path(".")

if not pathlib.Path(static_path, os.path.basename(HTML_VIEWER_PATH)).exists():
        shutil.copy(HTML_VIEWER_PATH, static_path)


path = "/home/louis/Github/pyvista-gallery/"

html = f"""
    <iframe src='static_viewer.html?fileURL=1.vtksz' width='400px' height='400px' frameborder='0'></iframe>
"""

# Create a index.html file and write the html variable in it



with open("index.html", "w") as f:
    f.write(html)
