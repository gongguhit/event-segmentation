import os
import yaml
from pathlib import Path

def load_api_keys(config):
    """
    Load API keys from a separate configuration file and update the main config.
    This ensures sensitive information is not tracked in Git.
    
    Args:
        config: The main configuration dictionary
        
    Returns:
        Updated configuration with API keys
    """
    # Copy the config to avoid modifying the original
    updated_config = config.copy()
    
    # Define the path to the API keys file
    api_keys_path = Path(__file__).parents[1] / 'config' / 'api_keys.yaml'
    
    # Check if the file exists
    if not api_keys_path.exists():
        print(f"Warning: API keys file not found at {api_keys_path}")
        print("GPT-4o integration will not be available.")
        # Set use_gpt4o to False if API keys are not available
        if 'training' in updated_config and 'use_gpt4o' in updated_config['training']:
            updated_config['training']['use_gpt4o'] = False
        return updated_config
    
    try:
        # Load API keys
        with open(api_keys_path, 'r') as f:
            api_keys = yaml.safe_load(f)
        
        # Update the configuration with API keys
        if 'training' in updated_config and 'azure_openai' in updated_config['training']:
            updated_config['training']['azure_openai'] = api_keys['azure_openai']
            # Enable GPT-4o if API keys are available
            updated_config['training']['use_gpt4o'] = True
            print("GPT-4o integration enabled with Azure OpenAI API keys.")
        
        return updated_config
    except Exception as e:
        print(f"Error loading API keys: {e}")
        # Set use_gpt4o to False if there's an error
        if 'training' in updated_config and 'use_gpt4o' in updated_config['training']:
            updated_config['training']['use_gpt4o'] = False
        return updated_config

def create_empty_api_keys_file():
    """
    Create an empty API keys file with placeholders if it doesn't exist
    """
    api_keys_path = Path(__file__).parents[1] / 'config' / 'api_keys.yaml'
    
    if not api_keys_path.exists():
        with open(api_keys_path, 'w') as f:
            f.write("# Azure OpenAI API Keys - DO NOT COMMIT THIS FILE TO GIT\n")
            f.write("# This file contains sensitive information and is included in .gitignore\n\n")
            f.write("azure_openai:\n")
            f.write("  api_key: \"your-azure-openai-api-key\"  # Replace with your Azure OpenAI API key\n")
            f.write("  endpoint: \"https://your-endpoint.openai.azure.com\"  # Replace with your Azure endpoint\n")
            f.write("  deployment_name: \"gpt-4o\"  # Replace with your specific GPT-4o deployment name\n")
        
        print(f"Created empty API keys file at {api_keys_path}")
        print("Please edit this file with your actual API keys to enable GPT-4o integration.")
        return True
    return False 