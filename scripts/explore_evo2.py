import importlib
import pkgutil
import inspect

def explore_module(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"Module: {module_name}")
        
        # List functions and classes
        print("Functions and classes:")
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if not name.startswith('_'):  # Skip private items
                    print(f"  - {name} ({'class' if inspect.isclass(obj) else 'function'})")
        
        return module
    except ImportError as e:
        print(f"Cannot import {module_name}: {e}")
        return None

# Explore Evo2 modules
print("Exploring bionemo.evo2...")
explore_module('bionemo.evo2')

print("\nExploring bionemo.evo2.run...")
explore_module('bionemo.evo2.run')

print("\nExploring bionemo.evo2.data...")
explore_module('bionemo.evo2.data')

# Check for command-line tools
import subprocess
print("\nChecking for command-line tools...")
try:
    result = subprocess.run(['which', 'finetune_evo2'], capture_output=True, text=True)
    print(f"finetune_evo2: {'Found at ' + result.stdout.strip() if result.returncode == 0 else 'Not found'}")
    
    result = subprocess.run(['which', 'preprocess_evo2'], capture_output=True, text=True)
    print(f"preprocess_evo2: {'Found at ' + result.stdout.strip() if result.returncode == 0 else 'Not found'}")
    
    result = subprocess.run(['which', 'infer_evo2'], capture_output=True, text=True)
    print(f"infer_evo2: {'Found at ' + result.stdout.strip() if result.returncode == 0 else 'Not found'}")
except Exception as e:
    print(f"Error checking tools: {e}")

# Look for fine-tuning examples
print("\nLooking for fine-tuning examples...")
try:
    result = subprocess.run(['find', '/', '-name', '*evo2*fine*', '-type', 'f'], capture_output=True, text=True)
    examples = result.stdout.strip().split('\n')
    for example in examples:
        if example:
            print(f"  - {example}")
except Exception as e:
    print(f"Error finding examples: {e}")


