import pathlib, sys  # DEBUG
print("DBG resolver package path:", pathlib.Path(__file__).resolve(), file=sys.stderr, flush=True)  # DEBUG
# empty file to mark this directory as a Python package
