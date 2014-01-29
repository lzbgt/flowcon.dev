'''
Created on Jun 27, 2013

@author: schernikov
'''

from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

#class cpp_build_ext(build_ext):
#    def build_extension(self, ext):
#        try:
#            self.compiler.compiler_so.remove('-Wstrict-prototypes')
#        except:
#            pass
#        build_ext.build_extension(self, ext)

res = setup(ext_modules = cythonize(["misc.pyx", "nreceiver.pyx", "collectors.pyx", "nquery.pyx"]),
            #cmdclass = {'build_ext': cpp_build_ext},
            script_args=['build_ext', '--inplace'])
