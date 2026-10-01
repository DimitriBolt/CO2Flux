"""Create reading copies from the executed notebook; never query Oracle.

Run from any directory. Needs the local requirements, Pandoc, XeLaTeX and DejaVu.
The notebook is the only editable report; PDF/HTML are generated views.
"""

from pathlib import Path
import argparse
import copy
import os
import re
import subprocess
import tempfile

import nbformat
from nbconvert import HTMLExporter, LatexExporter
from nbconvert.writers import FilesWriter


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01_october_2017.ipynb"
VERSION = "2026-09-25"
PDF_DIR = HERE / "output/pdf"
HTML_DIR = HERE / "output/html"


def rebase_markdown_links(notebook, output_directory):
    """Keep links to chapters and code useful from the export subdirectories."""
    adjusted = copy.deepcopy(notebook)
    def replace(match):
        label, url = match.groups()
        if url.startswith(("http:", "https:", "mailto:", "#", "attachment:")):
            return match.group(0)
        path, separator, anchor = url.partition("#")
        relative = Path(os.path.relpath(HERE / path, output_directory)).as_posix()
        return f"[{label}]({relative}{separator}{anchor})"
    for cell in adjusted.cells:
        if cell.cell_type == "markdown":
            cell.source = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, cell.source)
    return adjusted


def main(notebook_path=NOTEBOOK, version=None):
    notebook_path = Path(notebook_path).resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    version = version or notebook.metadata.get("research", {}).get("version", VERSION)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells):
        raise ValueError("Сначала выполните все ячейки notebook и сохраните его.")
    if any(output.output_type == "error" for cell in code_cells for output in cell.outputs):
        raise ValueError("В notebook есть ошибка выполнения; экспорт остановлен.")
    for directory in [PDF_DIR, HTML_DIR, HERE / "tmp"]:
        directory.mkdir(parents=True, exist_ok=True)

    html_exporter = HTMLExporter(exclude_input=True, exclude_input_prompt=True,
                                 exclude_output_prompt=True, embed_images=True)
    html, _ = html_exporter.from_notebook_node(
        rebase_markdown_links(notebook, HTML_DIR), resources={"metadata": {"name": notebook_path.stem}})
    html_path = HTML_DIR / f"{notebook_path.stem}.html"
    html_path.write_text(html, encoding="utf-8")

    template = HERE / "templates/co2_report.tex.j2"
    exporter = LatexExporter(exclude_input=True, exclude_input_prompt=True,
                              exclude_output_prompt=True, template_file=str(template))
    source, resources = exporter.from_notebook_node(
        rebase_markdown_links(notebook, PDF_DIR),
        resources={"metadata": {"name": notebook_path.stem, "path": str(HERE)},
                   "report_date": ".".join(reversed(version.split("-"))),
                   "report_chapter": notebook.metadata.get("research", {}).get("chapter", 1)})
    # These report tables are short. Keep their header and rows on one page.
    def reserve_table_space(match):
        table = match.group(0)
        rows = table.count("\\\\")
        needed = min(22, rows + 4)
        return f"\\Needspace{{{needed}\\baselineskip}}\n" + table
    source = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}",
                    reserve_table_space, source, flags=re.DOTALL)
    # Freeze technical navigation as readable references in the print copy.
    # PDF headings themselves remain bookmarks in the resulting document.
    with tempfile.TemporaryDirectory(prefix="latex-", dir=HERE / "tmp") as temporary:
        work = Path(temporary)
        FilesWriter(build_directory=str(work)).write(source, resources, notebook_name="chapter")
        command = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "chapter.tex"]
        combined_log = []
        for _ in range(3):
            completed = subprocess.run(command, cwd=work, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            combined_log.append(completed.stdout)
            if completed.returncode:
                (HERE / "tmp/export-error.log").write_text(completed.stdout)
                (HERE / "tmp/export-error.tex").write_text(source)
                raise RuntimeError("Ошибка XeLaTeX; см. Research_log/tmp/export-error.log")
        (HERE / "tmp/export-latex.log").write_text("\n".join(combined_log))
        pdf_path = PDF_DIR / f"{notebook_path.stem}_{version}.pdf"
        pdf_path.write_bytes((work / "chapter.pdf").read_bytes())
    print(f"HTML: {html_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path, default=NOTEBOOK)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    main(args.notebook, args.version)
