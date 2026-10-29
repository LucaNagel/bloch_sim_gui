import json

from blochsimulator.project_notebook import (
    DEFAULT_NOTEBOOK_NAME,
    write_project_loader_notebook,
)


def test_notebook_is_valid_json_and_never_overwrites(tmp_path):
    first = write_project_loader_notebook(tmp_path, loop_id="abc12345")
    first.write_text(first.read_text() + " ")  # simulate a user edit
    edited = first.read_text()
    second = write_project_loader_notebook(tmp_path)

    assert first.name == DEFAULT_NOTEBOOK_NAME
    assert second.name == "load_simulations_2.ipynb"
    assert first.read_text() == edited
    notebook = json.loads(edited)
    assert notebook["nbformat"] == 4
    code = "".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "LOOP_ID = 'abc12345'" in code
    assert "load_project" in code
    assert "infer_spiral_acquisition" in code
    assert "sampling_gap" in code
    assert "Cartesian FFT or non-Cartesian NUFFT" in edited
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "cell", "exec")
