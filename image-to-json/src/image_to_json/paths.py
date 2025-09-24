from pathlib import Path

def get_path_to_configs() -> str:
    return str(Path(__file__).parent.parent.parent / "configs")

def get_path_to_evals() -> str:
    return str(Path(__file__).parent.parent.parent / "evals")