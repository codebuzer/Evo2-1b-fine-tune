import sys
import importlib
import pkgutil

def check_module(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"? {module_name} is available")
        
        # List submodules
        print(f"Submodules of {module_name}:")
        for _, name, ispkg in pkgutil.iter_modules(module.__path__):
            print(f"  - {name} ({'package' if ispkg else 'module'})")
        
        return module
    except ImportError as e:
        print(f"? {module_name} is not available: {e}")
        return None

def check_class(module, class_name):
    if module is None:
        return
    
    try:
        cls = getattr(module, class_name)
        print(f"? {class_name} class is available in {module.__name__}")
        
        # List methods
        print(f"Methods of {class_name}:")
        for method in dir(cls):
            if not method.startswith('_'):  # Skip private methods
                print(f"  - {method}")
    except AttributeError:
        print(f"? {class_name} class is not available in {module.__name__}")

# Check main modules
bionemo = check_module('bionemo')
if bionemo:
    # Check specific modules
    check_module('bionemo.evo2')
    evo2_model = check_module('bionemo.evo2.model')
    if evo2_model:
        check_class(evo2_model, 'Evo2Model')

print("\nPython path:")
for path in sys.path:
    print(f"  - {path}")

