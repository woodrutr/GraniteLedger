from pathlib import Path
import importlib
import sys

# # __init__: Get models in folder
path = Path(__file__)
directory = path.parent
modules = [file.name.replace(".py", "") for file in directory.glob("*.py") if "__init__" not in file.name]

# # __init__: Unpack modules
__all__ = []
for module_name in modules:
    try:
        # Import the module dynamically
        module = importlib.import_module(f".{module_name}", __name__)

        # Get the model class definition
        model_class = getattr(module, module_name)

        # Add the module to the package's namespace
        setattr(sys.modules[__name__], module_name, model_class) 
        __all__.append(module_name)
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
