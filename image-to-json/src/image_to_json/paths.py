from pathlib import Path

def get_path_to_configs() -> str:
    path = str(Path(__file__).parent.parent.parent / "configs")

    # create path if it does not exist
    Path(path).mkdir(parents=True, exist_ok=True)

    return path

def get_path_to_evals() -> str:
    path = str(Path(__file__).parent.parent.parent / "evals")

    # create path if it does not exist
    Path(path).mkdir(parents=True, exist_ok=True)

    return path