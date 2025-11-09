from setuptools import setup, find_packages

setup(
    name="enhanced_benchmark_tool",
    version="0.1.0",
    description="A Python package for dataset profiling and model benchmarking.",
    author="Yisong Chen",
    author_email="yisongchen97@gmail.com",
    url="https://github.com/Riley702/enhanced-benchmark-tool",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
 
