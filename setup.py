import setuptools

setuptools.setup(
	name="tnn",
	version="0.0.1",
	description="Deep triplet neural networks for integration of scRNAseq data",
	author='Lukas Simon, Yin-Ying Wang',
	author_email="lkmklsmn@gmail.com",
	packages=['tnn'],
	install_requires=['sklearn','scanpy','anndata','pandas','tensorflow','numpy', 'ivis', 'scipy', 'hnswlib']
)
