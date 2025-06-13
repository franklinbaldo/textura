from setuptools import find_packages, setup

setup(
    name="textura",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "textura = textura.cli:textura_cli",
        ],
    },
)
