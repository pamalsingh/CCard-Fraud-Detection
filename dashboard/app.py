"""
Launcher for the dashboard that imports the project's root `app.py` by
file path to avoid a circular import when Streamlit executes this file
as the `app` module.

Run with:
    streamlit run dashboard/app.py
"""
import os
import importlib.util
import sys


def load_root_main():
    root_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app.py'))
    if not os.path.exists(root_app_path):
        raise FileNotFoundError(f"Root app.py not found at {root_app_path}")

    spec = importlib.util.spec_from_file_location("creditcard_root_app", root_app_path)
    module = importlib.util.module_from_spec(spec)
    # Ensure the module can import project packages relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec.loader.exec_module(module)
    if not hasattr(module, 'main'):
        raise AttributeError("Loaded root app.py does not define a 'main' function")
    return module.main


if __name__ == '__main__':
    main = load_root_main()
    main()
