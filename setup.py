from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name='c_garch11',
        sources=[
            'src/_garch11.pyx',
            'src/garch11.cpp',
        ],
        include_dirs=[
            'src',
            np.get_include()
        ],
        language='c++',
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level='3',
    ),
    zip_safe=False,
)
