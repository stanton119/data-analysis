from setuptools import setup, find_packages

setup(
    name="ml-project",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning project template with MLflow integration.",
    packages=find_packages(),
    install_requires=[
        "mlflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",  # or "torch" depending on your model framework
        "matplotlib",
        "seaborn",
        "jupyter",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
