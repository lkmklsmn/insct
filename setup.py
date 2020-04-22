import setuptools

setuptools.setup(
	name="bbtnn",version="0.0.1",description="Identify driving transcriptional regulator in single cell embeddings",author='Lukas Simon, Fangfang Yan',author_email="Lukas.Simon@uth.tmc.edu",packages=['bbtnn'],install_requires=['sklearn','scanpy','anndata', 'bbknn','pandas','seaborn','matplotlib','dca','tensorflow','numpy', 'ivis', 'scipy']
)
