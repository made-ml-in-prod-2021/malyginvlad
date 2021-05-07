from setuptools import find_packages, setup

setup(
    name="ml_project",
    python_requires='>=3.4'
    packages=find_packages(),
    version='0.1.0',
    description="homework1",
    author="Vladislav Malygin",
    install_requires=[
        "click==7.1.2",
        "dataclasses==0.6",
        "Faker==8.1.2",
        "marshmallow-dataclass==8.3.0",
        "numpy==1.19.2",
        "pandas==1.1.5",
        "pytest==6.2.4",
        "python-dotenv>=0.5.1",
        "PyYAML==3.11",
        "scikit-learn==0.24.1",
        "seaborn==0.11.1",
        "matplotlib==3.4.1"
    ]
)