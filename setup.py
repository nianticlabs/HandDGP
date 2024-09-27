import setuptools

__version__ = "0.1.0"

setuptools.setup(
    name="handdgp",
    version=__version__,
    description="HandDGP: Camera-Space Hand Mesh Prediction with Differentiable Global Positioning",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    project_urls={"Source": "https://github.com/nianticlabs/handdgp"},
)
