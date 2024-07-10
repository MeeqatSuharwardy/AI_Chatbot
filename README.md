# Project README

This README provides instructions for setting up a virtual environment and running Python scripts for this project.

## Setting up a Virtual Environment

Create a virtual environment using Python's built-in venv module. Replace <env_name> with a name for your environment:

    python -m venv <env_name>

Activate the virtual environment:

On Windows:

    <env_name>\Scripts\activate

On macOS and Linux:
source <env_name>/bin/activate

# Installing Dependencies

    pip install -r requirements.txt

# Hugging Face token and CLI Login

    huggingface-cli login (requires password and email/username)

    OR

    huggingface-cli login --token $HUGGINGFACE_TOKEN  (create token from here "https://huggingface.co/settings/tokens")

# Running Python Scripts

1. To ingest data for the project and fine-tune/train on your data, run: `python ingest.py`
2. To ingest data for the project directly without fine-tuning or training, run: `python ingest_2.py`
3. To train the machine learning model, run: `python model.py`
4. To run ChainLit with the model.py script, use: `chainlit run model.py`
