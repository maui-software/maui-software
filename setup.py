from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='maui',
      version='0.1',
      description='Eco-acoustics data visualization and analysis',
      url='---',
      author='Caio Ferreira',
      author_email='caio.bernardo@usp.br',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'librosa',
          'audioread',
          'numpy',
          'pandas',
          'matplotlib',
          'plotly',
          'fpdf',
          'kaleido',
          'importlib'
      ],
      zip_safe=False,
      include_package_data=True,
      package_data={'': ['data/*.png']}

    )