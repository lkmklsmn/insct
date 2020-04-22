import setuptools

setuptools.setup(
	name="tnn",version="0.0.1",description="Deep triplet neural networks overcome batch effects in single cell sequencing data",author='Lukas Simon, Yin-Ying Wang',author_email="Lukas.Simon@uth.tmc.edu",packages=['tnn'],install_requires=['sklearn','scanpy','anndata', 'bbknn','pandas','seaborn','matplotlib','tensorflow','numpy', 'ivis', 'scipy']
)
