# Final ML/IDA Project
Final project of the Course Machine Learning and Intelligent Data Analysis at the University of Potsdam lead by Prof. T. Scheffer.

In this project I decided to work on project 5 (spam) since I am currently studying Cognitive Systems which has a focus on Language tasks.

## Requirements
The Python packages needed to run this project are listed in [requirements.txt](requirements.txt). The Python Version used is 3.8.8 on a windows 10 PC with 16GB of RAM and a RTX3070. If this is run on a GPU and there is insufficient VRAM available, please reduce the batch size in [config.yml](config.yml) and run the pytorch model directly instead of in the Jupyter Notebook.

## Reproduction
Everything can be directly run from the [Jupyter Notebook](final_project_WEISER.ipynb) and there are pre-trained model available [here](https://drive.google.com/drive/folders/1lZmJ6bpKlj2L15ByMNMiScPn6oZ7N2gn?usp=sharing).\
Alternatively models can be trained by running the training scripts in the notebook. The PyTorch model can be trained directly by running 

`$ python multi_layer_perceptron.py`

