'''
Created on Jun 27, 2013

@author: schernikov
'''

from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

class cpp_build_ext(build_ext):
    def build_extension(self, ext):
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except:
            pass
        build_ext.build_extension(self, ext)

setup(ext_modules = cythonize(["receiver.pyx"]),
      cmdclass = {'build_ext': cpp_build_ext})
