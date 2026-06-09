"""
Configuração do Dynaconf para gerenciamento de settings.

Este módulo configura o Dynaconf para carregar configurações de settings.toml
e variáveis de ambiente.
"""

__author__ = "Emerson V. Rafael"
__copyright__ = "Copyright 2026, Verificador Inteligente de Orçamentos de Obras, DataCraft"
__credits__ = ["Emerson V. Rafael"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Emerson V. Rafael"

__status__ = "Development"

import sys
from functools import lru_cache
from pathlib import Path

from dynaconf import Dynaconf

# Adjust import path for data functions
sys.path.insert(0, str(Path(__file__).parents[2]))

# Get current directory
CONFIG_PATH = Path(__file__).parent.resolve()


@lru_cache()
def get_settings() -> Dynaconf:
    """Get IARA settings singleton instance."""

    settings = Dynaconf(
        settings_files=[
            Path(CONFIG_PATH, "settings.toml"),
            Path(CONFIG_PATH, ".secrets.toml"),
        ],
        environments=True,  # Enable multiple environments like development, production
        load_dotenv=True,  # Enable loading of .env files
    )

    return settings
