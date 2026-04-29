#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Inform the user that the setup is complete
echo "Setup complete. Activate the virtual environment using 'source venv/bin/activate'."