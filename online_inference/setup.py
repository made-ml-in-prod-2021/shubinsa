from setuptools import find_packages, setup

setup(
    name="online_inference",
    packages=find_packages(),
    version="0.1.0",
    description="online_inference",
    author="Sergey Shubin",
    install_requires=[
        "pandas==1.2.3",
        "numpy==1.20.2",
        "click==7.1.2",
        "fastapi~=0.65.1",
        "uvicorn==0.13.4",
        "requests==2.25.1",
        "pytest==6.2.4",
        "PyYAML==5.4.1",
        "scikit-learn==0.22.1",
        "setuptools~=45.2.0",
        "marshmallow-dataclass==8.4.1",
        "pydantic==1.8.2",
    ],
    license="MIT",
)
