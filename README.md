# MRI Classifier
Personal Experiemental Work. Brain MRI classification

## Installation
Use python 3.9

Install dependencies

`pip install -r requirements.txt`

## Dataset

Dataset can be downloaded from :https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Need to put the data in `data/`

##  Run

Run training with 20 epochs and force training on cpu

`python main.py --e 20 --device cpu`

## Output and Experiment Tracking

Each training is saved in `runs/`

## Personal Notes
todo
- add my notebooks
- implement resume training
- finalize inference 
- build web app 
- benchmark different architectures


