'''
Created on Jun 27, 2013

@author: schernikov
'''

from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

#class cpp_build_ext(build_ext):
#    def build_extension(self, ext):
#        try:
#            self.compiler.compiler_so.remove('-Wstrict-prototypes')
#        except:
#            pass
#        build_ext.build_extension(self, ext)

ext_modules = cythonize(["misc.pyx", "nreceiver.pyx", "collectors.pyx", 'timecollect.pyx', 
                         "nquery.pyx", "napps.pyx"])

ext_modules.extend([Extension('minutescoll',
                             sources=['../csrc/minutescoll.c'], 
                             include_dirs=['../includes']),
                    Extension('hourscoll',
                             sources=['../csrc/hourscoll.c'], 
                             include_dirs=['../includes']),
                    Extension('dayscoll',
                             sources=['../csrc/dayscoll.c'], 
                             include_dirs=['../includes'])])

res = setup(ext_modules = ext_modules,
            #cmdclass = {'build_ext': cpp_build_ext},
            script_args=['build_ext', '--inplace'])
