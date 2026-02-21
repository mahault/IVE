from setuptools import setup, find_packages

setup(
    name="ive",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "inferactively-pymdp",
    ],
)
