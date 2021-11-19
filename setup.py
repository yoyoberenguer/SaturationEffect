"""
Setup.py file

Configure the project, build the package and upload the package to PYPI
"""
import setuptools
from Cython.Build import cythonize
from setuptools import Extension

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SaturationEffect",
    version="1.0.7",
    author="Yoann Berenguer",
    author_email="yoyoberenguer@hotmail.com",
    description="Saturation effect (shader effect)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyoberenguer/SaturationEffect",
    # packages=setuptools.find_packages(),
    packages=['SaturationEffect'],
    ext_modules=cythonize([
        Extension("SaturationEffect.saturation", ["SaturationEffect/saturation.pyx"],
                  extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"], language="c")]),
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    license='MIT',

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Cython',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        # 'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],

    install_requires=[
        'setuptools>=49.2.1',
        'Cython>=0.28',
        'numpy>=1.18',
        'pygame>=2.0'
    ],
    python_requires='>=3.6',
    platforms=['any'],
    include_package_data=True,
    data_files=[
        ('./lib/site-packages/SaturationEffect',
         ['LICENSE',
          'MANIFEST.in',
          'pyproject.toml',
          'README.md',
          'requirements.txt',
          'SaturationEffect/__init__.py',
          'SaturationEffect/__init__.pxd',
          'SaturationEffect/saturation.pyx',
          'SaturationEffect/saturation.pxd',
          'SaturationEffect/setup_saturation.py',
          'SaturationEffect/example.py',
          'SaturationEffect/hsl_c.c'
          ]),
        ('./lib/site-packages/SaturationEffect/tests',
         ['SaturationEffect/tests/test_saturation.py',
          'SaturationEffect/tests/profiling.py'
          ]),
        ('./lib/site-packages/SaturationEffect/Assets',
         [
             'SaturationEffect/Assets/background_checker.png',
             'SaturationEffect/Assets/logo.png',
             'SaturationEffect/Assets/logo_alpha.png',
             'SaturationEffect/Assets/p1.png',
             'SaturationEffect/Assets/skull_alpha.png',
             'SaturationEffect/Assets/423px-Excitation_Purity.svg.png'
         ])
    ],

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/yoyoberenguer/SaturationEffect/issues',
        'Source': 'https://github.com/yoyoberenguer/SaturationEffect',
    },
)

