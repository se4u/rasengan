'''
| Filename    : setup.py
| Description : Setup Script.
| Author      : Pushpendre Rastogi
| Created     : Thu Oct 29 16:47:04 2015 (-0400)
| Last-Updated: Thu Oct 29 19:36:45 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 2
'''
from setuptools import setup

package = 'rasengan'
version = '0.1'

setup(name=package,
      version=version,
      description="A powerful mix of decorators and context managers.",
      url='https://github.com/se4u/rasengan',
      author='Pushpendre Rastogi',
      packages=['rasengan'])
