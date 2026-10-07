"""Force stub embeddings before the app reads the environment."""

import os

os.environ["ML_STUB"] = "1"
