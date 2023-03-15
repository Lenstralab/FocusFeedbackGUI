import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="focusfeedbackgui",
    packages=["focusfeedbackgui", "focusfeedbackgui.microscopes"],
    version="2023.3.0",
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
    python_requires='>=3.8, <3.11',
    install_requires=['numpy>=1.16.5', 'scipy', 'matplotlib', 'parfor>=2021.7.1', 'scikit-image>=0.18.0', 'pandas',
                      'PySide2', 'pyyaml', 'numba', 'icc_rt', 'pywin32; platform_system=="Windows"',
                      'tllab_common[transforms]>=2023.3.0'],
    entry_points={"console_scripts": ["focusfeedbackgui=focusfeedbackgui.__main__:main"]},
    package_data={'': ['conf.yml', 'stylesheet.qss']},
    include_package_data=True,
)
