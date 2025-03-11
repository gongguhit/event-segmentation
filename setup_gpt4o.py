#!/usr/bin/env python
"""
Helper script to set up GPT-4o integration for the event segmentation project.
This script will create a config/api_keys.yaml file with your Azure OpenAI API keys.
"""

import os
import yaml
from pathlib import Path
import getpass

def main():
    print("=== GPT-4o Integration Setup ===")
    print("This script will help you set up GPT-4o integration with Azure OpenAI.")
    print("Your API keys will be stored in config/api_keys.yaml, which is excluded from Git.")
    print("You can update your keys at any time by editing this file or running this script again.")
    print("\nIMPORTANT: Make sure you have an Azure OpenAI account with GPT-4o access.")
    print("If you don't have this, press Ctrl+C to exit now.")
    print("\nPress Enter to continue...")
    input()
    
    # Create the config directory if it doesn't exist
    config_dir = Path(__file__).parent / 'config'
    config_dir.mkdir(exist_ok=True)
    
    # Path to API keys file
    api_keys_path = config_dir / 'api_keys.yaml'
    
    # Check if file already exists
    existing_keys = {}
    if api_keys_path.exists():
        try:
            with open(api_keys_path, 'r') as f:
                existing_keys = yaml.safe_load(f) or {}
            print("Found existing API keys file. You can update the keys.")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            existing_keys = {}
    
    # Azure OpenAI settings
    azure_openai = existing_keys.get('azure_openai', {})
    
    # Get API key (with option to keep existing)
    default_api_key = azure_openai.get('api_key', '')
    if default_api_key and default_api_key != "your-azure-openai-api-key":
        default_api_key_display = default_api_key[:5] + '****' + default_api_key[-4:] if len(default_api_key) > 10 else '****'
        print(f"Current API key: {default_api_key_display}")
        update_api_key = input("Update API key? (y/N): ").lower() == 'y'
    else:
        update_api_key = True
    
    if update_api_key:
        api_key = getpass.getpass("Enter your Azure OpenAI API key: ")
    else:
        api_key = default_api_key
    
    # Get endpoint (with option to keep existing)
    default_endpoint = azure_openai.get('endpoint', '')
    if default_endpoint and default_endpoint != "https://your-endpoint.openai.azure.com":
        print(f"Current endpoint: {default_endpoint}")
        update_endpoint = input("Update endpoint? (y/N): ").lower() == 'y'
    else:
        update_endpoint = True
    
    if update_endpoint:
        endpoint = input("Enter your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com): ")
    else:
        endpoint = default_endpoint
    
    # Get deployment name (with option to keep existing)
    default_deployment = azure_openai.get('deployment_name', 'gpt-4o')
    print(f"Current deployment name: {default_deployment}")
    update_deployment = input("Update deployment name? (y/N): ").lower() == 'y'
    
    if update_deployment:
        deployment_name = input("Enter your GPT-4o deployment name (default is 'gpt-4o'): ") or 'gpt-4o'
    else:
        deployment_name = default_deployment
    
    # Prepare API keys dictionary
    api_keys = {
        'azure_openai': {
            'api_key': api_key,
            'endpoint': endpoint,
            'deployment_name': deployment_name
        }
    }
    
    # Save API keys
    with open(api_keys_path, 'w') as f:
        f.write("# Azure OpenAI API Keys - DO NOT COMMIT THIS FILE TO GIT\n")
        f.write("# This file contains sensitive information and is included in .gitignore\n\n")
        yaml.dump(api_keys, f, default_flow_style=False)
    
    print(f"\nAPI keys saved to {api_keys_path}")
    print("GPT-4o integration is now set up!")
    print("\nTo enable GPT-4o in your training, set use_gpt4o: true in config/default.yaml")
    print("Or just run your training normally - the system will automatically use your API keys.")
    
    # Verify .gitignore includes the API keys file
    gitignore_path = Path(__file__).parent / '.gitignore'
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        if 'config/api_keys.yaml' not in gitignore_content:
            print("\nWARNING: config/api_keys.yaml is not listed in .gitignore!")
            print("This means your API keys might be committed to Git.")
            add_to_gitignore = input("Add it to .gitignore now? (Y/n): ").lower() != 'n'
            
            if add_to_gitignore:
                with open(gitignore_path, 'a') as f:
                    f.write("\n# API Keys and Sensitive Information\nconfig/api_keys.yaml\n")
                print("Added config/api_keys.yaml to .gitignore")
    else:
        print("\nWARNING: No .gitignore file found in the project root!")
        print("Make sure to add config/api_keys.yaml to .gitignore before committing to Git.")

if __name__ == "__main__":
    main() 