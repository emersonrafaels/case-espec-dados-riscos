"""Load src/config/settings.toml and expose a typed config dict.

Priority for credentials: settings.toml < environment variable.
Call get_config() anywhere in the project to get the merged config.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib          # built-in Python 3.11+
except ImportError:
    import tomli as tomllib  # pip install tomli  (Python < 3.11)

_SETTINGS_PATH = Path(__file__).parent / "settings.toml"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the merged configuration (file values overridden by env vars)."""
    with open(_SETTINGS_PATH, "rb") as fh:
        cfg = tomllib.load(fh)

    ia  = cfg["ia"]
    ath = cfg["athena"]

    # Env-var overrides (env wins over settings.toml)
    ia["api_key"]       = os.getenv("OPENAI_API_KEY",    ia.get("api_key", ""))
    ia["client_id"]     = os.getenv("IARA_CLIENT_ID",   ia.get("client_id", ""))
    ia["client_secret"] = os.getenv("IARA_CLIENT_SECRET", ia.get("client_secret", ""))

    ath["region"]                = os.getenv("AWS_DEFAULT_REGION",       ath.get("region", "sa-east-1"))
    ath["s3_output"]             = os.getenv("ATHENA_S3_OUTPUT",         ath.get("s3_output", ""))
    ath["workgroup"]             = os.getenv("ATHENA_WORKGROUP",         ath.get("workgroup", "primary"))
    ath["partition_col"]         = os.getenv("ATHENA_PARTITION_COL",    ath.get("partition_col", ""))
    ath["default_partition"]     = os.getenv("ATHENA_DEFAULT_PARTITION", ath.get("default_partition", "202604"))
    ath["aws_access_key_id"]     = os.getenv("AWS_ACCESS_KEY_ID",        ath.get("aws_access_key_id", ""))
    ath["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY",    ath.get("aws_secret_access_key", ""))
    ath["aws_session_token"]     = os.getenv("AWS_SESSION_TOKEN",        ath.get("aws_session_token", ""))

    def _int(val, default: int) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    ath["limit_min"]     = _int(ath.get("limit_min"),     10)
    ath["limit_max"]     = _int(ath.get("limit_max"),     10000)
    ath["limit_default"] = _int(ath.get("limit_default"), 100)

    return cfg
