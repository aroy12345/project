from setuptools import setup, find_packages

setup(
    name="presto",
    version="0.1.0",
    packages=find_packages(where="presto/src"),
    package_dir={"": "presto/src"},
    install_requires=[
        "torch",
        "numpy",
        # Add other dependencies as needed
    ],
    python_requires=">=3.6",
)