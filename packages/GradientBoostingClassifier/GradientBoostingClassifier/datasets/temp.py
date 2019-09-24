import pathlib
import pandas

PACKAGE_ROOT = pathlib.Path(pandas.__file__).resolve().parent

print(PACKAGE_ROOT)