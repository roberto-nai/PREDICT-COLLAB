"""
Configuration loader for the application.
Reads settings from config.yml file.
"""

import yaml
import os


def load_config(config_path='config.yml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file is not found
        yaml.YAMLError: If configuration file has invalid YAML syntax
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_processing_dir():
    """
    Get the processing directory path from configuration.
    
    Returns:
        str: Absolute path to the processing directory
    """
    config = load_config()
    processing_dir_name = config['directories']['processing']
    
    # Get the base directory (parent of web directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    return os.path.join(base_dir, processing_dir_name)


def get_staging_dir():
    """
    Get the staging directory path from configuration.
    The staging directory is used for uploaded trace files.
    
    Returns:
        str: Absolute path to the staging directory
    """
    config = load_config()
    staging_dir_name = config['directories']['staging']
    
    # Get the base directory (parent of web directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    return os.path.join(base_dir, staging_dir_name)


def get_max_decimal_places():
    """
    Get the maximum decimal places for prediction metrics from configuration.
    
    Returns:
        int: Maximum decimal places to use for rounding metrics
    """
    config = load_config()
    return config.get('prediction', {}).get('max_decimal_places', 5)
