C++ implementation of Garch(1,1) and Python bindings.

To build,

```bash
# or use python virtualenv below instead of conda
conda create -n garch11 python numpy scikit-learn Cython
conda activate garch11
python -m pip install xalglib
python setup.py build_ext --inplace --force
```
