from .release import __version__

# the package is imported during installation
# however installation happens in an isolated build environment
# where no dependencies are installed.

# this means: no importing the following modules will fail
# during installation. This is OK (but only during installation)


try:
    from .core import *
except ImportError:
    import os
    if "PIP_BUILD_TRACKER" in os.environ:
        pass
    else:
        # raise the original exception
        raise

    # this might be relevant during the installation process
    # otherwise setup.py cannot be executed
    pass
