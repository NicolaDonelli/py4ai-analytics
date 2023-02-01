import os
import random

from py4ai.core.utils.fs import create_dir_if_not_exists

test_path = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = create_dir_if_not_exists(os.path.join(test_path, "resources", "data"))
TMP_FOLDER = create_dir_if_not_exists(
    os.path.join("/tmp", "%032x" % random.getrandbits(128))
)
