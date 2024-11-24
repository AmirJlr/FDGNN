# Thesis Project

[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
    
## Project Description
This project is core of **Molecular Property Prediction Pipeline** as thesis work, focusing on the analysis and modeling of molecular data. It includes data handling, molecular representation, and various modeling techniques.

## Table of Contents
- [Project Description](#project-description)
- [Folders Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Modules](#modules)
  - [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

## Folders Structure
```
Thesis/
├── modules/
│   ├── __init__.py
│   ├── data_handler.py
│   ├── fingerprints_descriptors.py
│   ├── models.py
│   ├── utils_classification.py
│   ├── utils_regression.py
│   └── ...
│
├── notebooks/
│   ├── data_handler.ipynb
│   ├── fingerprints_descriptors.ipynb
│   ├── molfeat.ipynb
│   ├── utils_classification.ipynb
│   └── utils_regression.ipynb
│  
├── data/
│   ├── ...
│   ├── README.md
│   └── datasets/
│       ├── delaney-processed.csv
│       ├── HIV.csv
│       ├── tox21.csv
│       └── ...
│
├── documents/
│   └── MainTemplate.docx
│
├── .gitignore
├── README.md
└── requirements.txt 
```

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/amirjlr/Thesis.git
    cd your-repo
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Modules
The `modules` directory contains the core code for manage **Molecular Property Prediction Pipeline.**

- `data_handler.py`: Functions and classes for data processing.
- `fingerprints_descriptors.py`: Code for calculating molecular fingerprints and descriptors.


### Notebooks
The `notebooks` directory contains Jupyter notebooks for data exploration, preprocessing, and model training.

- `data_handler.ipynb`: Process datasets in PyG graph data structure.
- `fingerprints_descriptors.ipynb`: Calculating 8 types molecular fingerprints and descriptors.
- `molfeat.ipynb`: Molecule feature extraction using molfeat.
- `utils_classification.ipynb`: Utilities for classification tasks.
- `utils_regression.ipynb`: Utilities for regression tasks.

### Running a Notebook
To run a notebook, navigate to the `notebooks` directory and open the desired notebook with Jupyter:

```bash
jupyter notebook notebooks/data_handler.ipynb
```


