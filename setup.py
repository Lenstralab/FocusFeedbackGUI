import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="focusfeedbackgui",
    packages=["focusfeedbackgui"],
    version="2022.02.0",
    author="Wim Pomp @ Lenstra lab NKI",
    author_email="w.pomp@nki.nl",
    description="Live track particles.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['numpy>=1.16.5', 'scipy', 'matplotlib', 'parfor>=2021.7.1', 'scikit-image>=0.18.0', 'pandas',
                      'tqdm', 'trackpy', 'PyQt5', 'pyyaml', 'numba', 'tifffile', 'czifile', 'colorcet',
                      'SimpleITK-SimpleElastix', 'multipledispatch'],
    scripts=['bin/focusfeedbackgui'],
)
