from pathlib import Path


root: Path = Path(__file__).absolute().parent.parent.parent
data: Path = root / 'data'
plots: Path = root / 'plots'
