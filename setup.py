import os
import setuptools 

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()

setuptools.setup(name='wavaimidiz',
      version='0.0',
      description='Pytorch music boundary detection',
      url='https://github.com/carlosholivan/music-boundaries-detection-cnn',
      author='Carlos Hernandez-Olivan',
      author_email='carloshero@unizar.es',
      license='https://github.com/carlosholivan/music-boundaries-detection-cnn/blob/master/LICENSE',
      packages=setuptools.find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3",
          ],
      install_requires=requirements,
      )