from pathlib import Path


root_dir = Path(__file__).parent.parent.resolve()

model_dir = str(Path(root_dir / 'models').resolve())

static_dir = root_dir / 'static'

templates_dir = root_dir / 'templates'
