Transformers-final-year-project
==============================

*Project done as a part of partial fulfillment of requirements of B.Tech in Computer Science and Engg. of [ASTU](https://astu.ac.in/) (Assam Science and Technology University)*

Final Year Project (TEXT GENERATION USING GENERATIVE AI) : Application to generate blogs and other text based content like email and social media posts using custom trained Transformer, LSTM models on various dataset of English and Assamese language. Option to generate blog with GPT-Neo is also available. Can be run as a standalone application without any need to call for external APIs. The project contains a flask API and the frontend to connect and communicate with it is available at [https://github.com/Transformers-G5/gen-front](https://github.com/Transformers-G5/gen-front)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Docker Installation
### 1. Clone models
1. cd /src/models
2. git lfs install (make sure git lfs is installed)
3. Model for text generation
    - git clone https://huggingface.co/EleutherAI/gpt-neo-125M (125M parameter model) 
    - git clone https://huggingface.co/EleutherAI/gpt-neo-1.3B (1.3B parameter model)
    - git clone https://huggingface.co/EleutherAI/gpt-neo-2.7B (2.7B parameter model)
4. Model for subprompts
    - git clone https://huggingface.co/mrm8488/t5-base-finetuned-common_gen
### 2. Create Docker Image
1. docker build . -t blog-generation (at root of the project)

### 3. Run Docker Image
2. docker run -p 4040:4040 blog-generation

# Conda Environment Initialisation
### 1. Create Conda environment
1. conda create --name g5-model python=3.8

### 2. Install dependencies
1. conda activate g5-model
2. pip install -r requirements.txt
3. python -m spacy download en_core_web_sm

### 3. Clone GPT-Neo model
1. cd /src/models
2. git lfs install (make sure git lfs is installed)
3. 
    - git clone https://huggingface.co/EleutherAI/gpt-neo-125M (125M parameter model) 
    - git clone https://huggingface.co/EleutherAI/gpt-neo-1.3B (1.3B parameter model)
    - git clone https://huggingface.co/EleutherAI/gpt-neo-2.7B (2.7B parameter model)
4. Model for subprompts
    - git clone https://huggingface.co/mrm8488/t5-base-finetuned-common_gen

** models not fine tuned

** Some of the custom trained models are not added to this repo and won't work without them.
    

