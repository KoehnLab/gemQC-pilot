"""Sphinx extension: write the option tables of gemqc.options as Markdown
files docs/_generated/options_<name>.md (included by the input reference)."""

from pathlib import Path


def _write_tables(app):
    from gemqc.options import DF_OPTIONS, RESOURCES, SYSTEM, TASKS, markdown_table
    out = Path(app.srcdir) / "_generated"
    out.mkdir(exist_ok=True)
    tables = {"system": SYSTEM, "resources": RESOURCES, "df": DF_OPTIONS}
    tables.update({name: t.options for name, t in TASKS.items()})
    for name, options in tables.items():
        path = out / f"options_{name}.md"
        text = markdown_table(options) + "\n"
        if not path.exists() or path.read_text() != text:  # keep timestamps otherwise
            path.write_text(text)


def setup(app):
    app.connect("builder-inited", _write_tables)
    return {"parallel_read_safe": True}
