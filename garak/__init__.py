"""Top-level package for garak"""

__version__ = "0.10.3.post1"
__app__ = "garak"
__description__ = "LLM vulnerability scanner"

import logging
import os
from garak import _config

GARAK_LOG_PATH_VAR = "GARAK_LOG_PATH"

# allow for a file path configuration from the ENV and set for child processes
_log_filename = os.getenv(GARAK_LOG_PATH_VAR, default=None)
if _log_filename is None:
    _log_filename = _config.transient.data_dir / "garak.log"
    os.environ[GARAK_LOG_PATH_VAR] = str(_log_filename)

_config.transient.log_filename = _log_filename

logging.basicConfig(
    filename=_log_filename,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
