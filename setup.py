from setuptools import setup


requirements = ["scipy>=0.17", "numpy>=1.10", "scikit-learn>=0.20.2"]

setup(
    name="bsz",
    version="0.0.1",
    packages=["bsz"],
    install_requires=requirements,
    description="Binary split with Zonotope",
    long_description="""
        bsz is good
        """,
    author="Peng Yu",
    author_email="yupbank@gmail.com",
    url="https://github.com/yupbank/tree_experiment",
    classifiers=["machine learning", "classification", "decision tree"],
)
