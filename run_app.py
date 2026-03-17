"""Launcher that fixes DLL paths before starting Streamlit."""
import os, sys, ctypes

# Pre-load torch DLLs on Windows
if sys.platform == 'win32':
    torch_lib = os.path.join(
        sys.prefix, 'Lib', 'site-packages', 'torch', 'lib'
    )
    if not os.path.isdir(torch_lib):
        # Try user-local install
        torch_lib = os.path.join(
            os.path.dirname(sys.executable),
            'Lib', 'site-packages', 'torch', 'lib'
        )
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        # Pre-load c10.dll
        try:
            ctypes.CDLL(os.path.join(torch_lib, 'c10.dll'))
        except Exception:
            pass

from streamlit.web import cli as stcli
sys.argv = [
    'streamlit', 'run', 'app.py',
    '--server.headless', 'true',
    '--server.port', '8501',
]
stcli.main()
