import sys
import os
from setuptools import setup, find_packages

CURRENT_PYTHON = sys.version_info[:2]
MIN_PYTHON = (3, 6)

if CURRENT_PYTHON < MIN_PYTHON:
    sys.stderr.write("""
        ============================
        Unsupported Python Version
        ============================

        Python {}.{} is unsupported. Please use a version newer than Python {}.{}.
    """.format(*CURRENT_PYTHON, *MIN_PYTHON))
    sys.exit(1)

with open('requirements.txt', 'r') as f:
    install_requires = f.readlines()

with open('README.rst') as f:
    README = f.read()

#trigger test
setup(name='gym-2048-extended',
      description='OpenAI Gym Environment for 2048 extended functionality',
      long_description=README,
      long_description_content_type='text/x-rst',
      version='1.7',
      url='https://www.github.com/geschnee/2048-gym',
      author='Georg Schneeberger',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      extras_require={},
)
