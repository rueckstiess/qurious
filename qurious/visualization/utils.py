import os


def clear_output():
    """
    Detects whether code is running in a Jupyter notebook or terminal
    and clears the output accordingly.

    Returns:
        bool: True if running in Jupyter, False if in terminal
    """
    # Try to detect if we're in a Jupyter environment
    try:
        # This will only work in IPython/Jupyter environments
        from IPython import get_ipython

        ipython = get_ipython()

        if ipython is not None and "IPKernelApp" in ipython.config:
            # We're in Jupyter notebook or qtconsole
            from IPython.display import clear_output as jupyter_clear

            jupyter_clear(wait=True)
            return True
        else:
            # We're in terminal IPython or standard Python
            os.system("cls" if os.name == "nt" else "clear")
            return False
    except (ImportError, NameError):
        # We're in standard Python
        os.system("cls" if os.name == "nt" else "clear")
        return False
