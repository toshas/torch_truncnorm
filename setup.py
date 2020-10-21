import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torch_truncnorm',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='Truncated Normal distribution in PyTorch',
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    author='Anton Obukhov',
    license='BSD',
    url='https://www.github.com/toshas/torch_truncnorm',
)
