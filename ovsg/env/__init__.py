from ovsg.core.env import EnvBase
from .notiondb import NotionDB
from .notionovidb import NotionOVIDB

env_map = {
    "notiondb": NotionDB,
    "notionovidb": NotionOVIDB,
}


def env_builder(cfg, **kwargs) -> EnvBase:
    """Build env"""
    env = env_map[cfg.env_name](cfg, **kwargs)
    return env
