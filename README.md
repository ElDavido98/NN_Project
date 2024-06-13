# NN_Project
For this project I was inspired by the paper [ClimateLearn - Benchmarking Machine Learning for Weather and Climate Modeling](https://arxiv.org/pdf/2307.01909.pdf#:~:text=ClimateLearn%20supports%20data%20pre%2D%20processing,weather%20forecasting%2C%20downscaling%2C%20and%20climate). I reimplemented the models used for forecasting; as networks, I have implemented: ResNet, UNet and ViT. After training, I used Root Mean Square Error (RMSE) and Anomaly Correlation Coefficient (ACC) to evaluate the performances of the trained networks.

## Usage
Running the code is very simple: for the colab notebook you have to run all the cells in sequence, while for the files here on github you have to download the 'Project' folder and run the file 'train.py' if you want to train the model or 'evaluate.py' if you want to evaluate it. I have also added a trained model so that you can test the 'evaluate.py' function.

The code has been trained and tested with the following dependencies:
* [**Python 3.10.2+**](https://www.python.org/)
* [climate-learn](https://climatelearn.readthedocs.io/en/latest/index.html)
* [Torch](https://pytorch.org/) - version 2.2.0
* [Numpy](https://scipy.org/install.html) - version 1.26.4
* [Pytorch Image Models (timm)](https://timm.fast.ai/) - version 0.9.16
* [NetCDF4](https://unidata.github.io/netcdf4-python/) - version 1.6.5
* [Scikit-Learn](https://scikit-learn.org/stable/install.html) - version 1.4.2
* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html#installation) - version 3.8.4

Since it is not possible to upload the data used for the experiment to github, I created an additional file for downloading. This file, called 'download.py', downloads the files to the directory defined by 'root_directory', so it may be necessary to change this directory to ensure that the data is saved correctly. The same change must be made in 'Project/Utils/utils' for the variable 'path', which is used to access the data.

### Installing Packages
```
pip install climate-learn
pip install torch==2.2.0
pip install numpy==1.26.4
pip install timm==0.9.16
pip install netCDF4==1.6.5
pip install scikit-learn==1.4.2
pip install matplotlib==3.8.4
```
