# Machine Learning Experiment Template

[PROJECT RELATED TO PAPER ________ IMPLEMENTING MODEL ________ ]

## Folder Structure

The repository follows a specific folder structure to keep things organized:

- `src/`: Contains the python code for actually running scripts.
- `exps/`: Stores experimental results, including model outputs, logs, metrics, and visualizations.
- `data/`: Houses the dataset files and any data-related resources.

Within the `src/models` directory, each model we implemented is included in its own subfolder (e.g. src/models/SVM). We include a) base model class definition, implementing train and fit methods, and b) a script to run the model and training.
To train a particular model with a given configuration parameters, type "python -m src.models.MODEL_NAME.train_MODEL_NAME --config src/models/MODEL_NAME/MODEL_NAME_config"

All other folder names are hopefully sufficiently descriptive. General utils used for all components are included in the high-level /src model directory.

## Getting Started

Follow these steps to start a new machine learning project based on this template:

1. **Clone the Repository**: Begin by cloning this repository to your local machine:

   ```bash
   git clone https://github.com/hrna-ox/[$PROJECT_NAME].git
   cd $PROJECT_NAME

## Setup Virtual Environment

Create and activate a virtual environment to isolate project dependencies:

 ```bash
 python -m venv venv
 source venv/bin/activate
 pip install -r requirements.txt

## Data Preparation

If using MIMIC-IV-ED data, obtain data through the [Link Text](https://physionet.org/content/mimic-iv-ed/2.2/), and save all files under "./data/MIMIC/". Then run the MIMIC data processing script to pre-process the data:

 ```bash
 python -m src.data/preprocessing/MIMIC/run_processing

## Citing this Repository

If you find this repository and work helpful for your work, please cite us for any publications or submissions.

# Contributing Guidelines

Thank you for your interest in contributing to this project! We welcome contributions from the community. To ensure a smooth collaboration process, please review and follow these guidelines.

## Getting Started

1. Fork this repository and create a new branch for your contributions.
2. Make your changes, following the coding style and conventions used in the project.
3. Test your changes thoroughly.

## Submitting Contributions

1. Commit your changes with descriptive commit messages.
2. Push your changes to your forked repository.
3. Submit a pull request, explaining the purpose of your changes and providing any necessary context.

## Code of Conduct

Please note that we have a [Code of Conduct](CODE_OF_CONDUCT.md) in place to ensure a respectful and inclusive environment for all contributors.

## Contact

If you have any questions or need further assistance, feel free to reach out to us via [email](mailto:henrique.aguiar@eng.ox.ac.uk) or open an issue.

We appreciate your efforts in contributing to this project!
