import os

def setup_project_structure():
    """
    Programmatically ensure that data directory and required files exist.
    Person 1 is responsible for setup, .gitignore, and README.md.
    """
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('.gitignore'):
        with open('.gitignore', 'w') as f:
            f.write("__pycache__/\n.ipynb_checkpoints/\n*.pyc\n.DS_Store\n")
    print("Initialization complete: Checked project structure.")
