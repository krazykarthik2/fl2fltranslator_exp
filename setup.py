from setuptools import setup, find_packages

setup(
    name="neural_compiler_loop",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pycparser>=2.21",
        "tqdm>=4.65.0",
    ],
)
