# implicit-recommender-system
A simple recommender system based on Collaborative Filtering for implicit feedback dataset that accepts a user id as an input and return top 10 recommended items for that user

## Installation
- First, clone this repo `git clone https://github.com/cruisybd/implicit-recommender-system/`
- Install all the requirments, `pip install -r requirements.txt` (you can do this in a virtual environment, just ensure you install the requirements after you have activated the virtual environment)

## How to Use
Please make sure you set up your environment correctly and install the requirements accordingly. Then, you can start using the app:
```
# on your terminal
cd ~/implicit-recommender-system
python item_recommender.py

# the program is building the recommender engine and once finished, will ask you to specify the user id you would like recommendations for
# if the user id you specify does not exist in the data, it will throw a warning and ask for another one
# once you are finished using it, exit the program via ctrl + c
```

## Basic Information
#### Sample Data
The file `data.csv` is a comma-separated file containing a random sample of user-item interactions. Both users and items are recorded with a unique id.

#### Exploratory Analysis
A series of exploratory analysis was conducted prior to building the model. Please see [this notebook](data_exploratory.ipynb)

#### Machine Learning Method Selection
An explanation of the model, assumptions and design decisions are outlined in [this notebook](machine_learning_approach.ipynb)

#### Model Testing
The testing method is described in [this notebook](machine_learning_approach.ipynb) and the code used is [test.py](test.py)
