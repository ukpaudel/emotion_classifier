import yaml
import os

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.path = config_path

    def get(self, *keys, default=None):
        """
        Get nested config values using dot-like access.
        Example: config.get('model', 'hidden_dim')
        """
        value = self.cfg
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default

    def __getitem__(self, key):
        return self.cfg[key]

    def __repr__(self):
        return yaml.dump(self.cfg, default_flow_style=False)

    def __setitem__(self, key, value):
        self.cfg[key] = value


# ✅ Add this function to allow simple import in run_experiment.py
def load_config(config_path: str) -> Config:
    return Config(config_path)
