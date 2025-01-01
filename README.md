# Capstone-1-Project

<p style="text-align: center">♟</p>

## Table of Contents

- [Introduction](#introduction)
- [Required Tools](#required-tools)
- [Live](#live)
- [Usage](#usage)
- [Features](#features)
- [Contact](#contact)

## Introduction

This little project is a test from the Machine Learning course curated by [DataTalks](https://datatalks.club/). The objective is create a model using Deep Learning methodologies to predict wildfires based on satellite images.

A Deep Learning model that predicts wildfires based on satellite images provides essential benefits by leveraging advanced image analysis to detect patterns and risks associated with wildfire occurrences. This predictive capability can assist in identifying areas prone to fires, enabling preventive measures to minimize damage. By processing satellite data, the model offers a cost-effective and scalable solution for analyzing large and remote areas, which traditional methods might struggle to monitor efficiently.

Though the model's scope is focused on prediction, it serves as a valuable tool for researchers, policymakers, and emergency responders. It provides actionable insights that can guide decisions, such as resource allocation or risk assessments, and enhance understanding of wildfire trends. While it doesn't directly combat wildfires, its predictions contribute to better preparedness and awareness, supporting efforts to protect lives, property, and the environment.

## Required Tools

- Docker
- Python 3.9 (it's needed for TL lite model loader)
- Poetry

## Live

1. [Deployed](https://supporting-gray-schneider-2a25633b.koyeb.app/)
2. Follow these [steps](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/tree/main/screenshots)
3. I recommend you to use this example (by the way, you can use any image's url):
   ```
   {"query": "https://www.copernicus.eu/system/files/styles/image_of_the_day/private/2023-03/image_day/20230327_IFVillanueva.jpg?itok=qqLCJirr"}
   ```
4. Check the biggest value from each class or label.

## Usage

To get started with the project, follow these steps:

1. **Fork the Repository**: Create your own copy of the repository to make changes.
2. **Build and Launch the Application**: Navigate to the project directory in your terminal and run the following command to build and launch the API:

```
docker-compose up --build
```

3. **Visit the url**: Access the Swagger interface by navigating to http://localhost:5000 in your web browser. Follow the prompts to use the predict route.

4. **Modify the query**: The api is going to download the image from the url and predict the results for each class or label.

5. **Install dependencies**: Navigte to the project directory and install dependencies using the following command:

```
poetry install
```

6. **Activate the virtual environment**: Only for Windows:

```
.\.venv\Scripts\activate.bat
```

macOs and Linux:

```
poetry shell
```

7. **Use the scripts**: You can use the train or predict script. Also you can train using your own model in the train script, remember to run these scripts with the following command:

```
python predict.py
```

## Features

- [x] [Notebook used for research](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/wildfire_prediction.ipynb)
- [x] [Data preparation](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/wildfire_prediction.ipynb)
- [x] [Data augmentation](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/wildfire_prediction.ipynb)
- [x] [Multiple models](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/wildfire_prediction.ipynb)
- [x] [Hyperparameter optimization](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/wildfire_prediction.ipynb)
- [x] [Script replication](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/train.py)
- [x] [Model exported](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/xception_v1_26_0.980.keras)
- [x] [Little model exported](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/model.tflite)
- [x] [Flask API](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/app.py)
- [x] [Dependency management with Poetry](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/pyproject.toml)
- [x] [Containerization with Docker](https://github.com/SchneiderSix/Capstone-1-Project-Zoomcamp/blob/main/Dockerfile)
- [x] [API Deployed on Koyeb](https://supporting-gray-schneider-2a25633b.koyeb.app/)

## Contact

Ask me anything, regards

[Juan Matias Rossi](https://www.linkedin.com/in/jmrossi6/)
