'''
Christian Bunker
M^2QM at UF
October 2021

transport module

transport calculations through magnetic molecules
'''
'''
import os

# Load modules which are developed as plugins of the namespace package
PYSCF_EXT_PATH = os.getenv('PYSCF_EXT_PATH')
if PYSCF_EXT_PATH:
    print(PYSCF_EXT_PATH);
else:
    print(__path__);
    __path__ = __import__('pkgutil').extend_path(__path__, __name__);
    print(__path__);
    if not all('/site-packages/' in p for p in __path__[1:]):
        sys.stderr.write('pyscf plugins found in \n%s\n'
                         'When PYTHONPATH is set, it is recommended to load '
                         'these plugins through the environment variable '
                         'PYSCF_EXT_PATH' % '\n'.join(__path__[1:]))
'''

