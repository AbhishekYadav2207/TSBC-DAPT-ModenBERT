import os
import sys
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

def main():
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "runner (1).ipynb"

    print(f"Loading notebook: {nb_path}")
    nb = nbformat.read(str(nb_path), as_version=4)

    # We want to execute the analytical cells (from index 25 onwards)
    # The kernel needs the repo root in sys.path
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    
    # We can create a kernel and execute cells starting from 25
    print(f"Starting kernel to execute analytical cells 25 to {len(nb.cells)-1}...")
    with client.setup_kernel():
        for i in range(25, len(nb.cells)):
            cell = nb.cells[i]
            if cell.cell_type == "code":
                print(f"Executing cell {i}...")
                client.execute_cell(cell, i)
                print(f"  Outputs: {len(cell.outputs)}")

    nbformat.write(nb, str(nb_path))
    print(f"Successfully executed and updated {nb_path} with freshly rendered outputs.")

if __name__ == "__main__":
    main()
