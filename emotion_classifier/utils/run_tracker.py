import yaml
import os
'''
This code logs model training  into a model_runs.yml file so that we can plot and compare training metrics between different models.

'''
def update_model_runs_yaml(config_path, log_dir, label):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    if 'runs' not in config:
        config['runs'] = []

    # Avoid duplicate entries
    if not any(run['log_dir'] == log_dir for run in config['runs']):
        config['runs'].append({'log_dir': log_dir, 'label': label})
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
