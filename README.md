# NN_Project
For this project I was inspired by the paper [ClimateLearn - Benchmarking Machine Learning for Weather and Climate Modeling](https://arxiv.org/pdf/2307.01909.pdf#:~:text=ClimateLearn%20supports%20data%20pre%2D%20processing,weather%20forecasting%2C%20downscaling%2C%20and%20climate). I reimplemented the models used for forecasting; as networks, I have implemented: ResNet, UNet and ViT. After training, I used Root Mean Square Error (RMSE) and Anomaly Correlation Coefficient (ACC) to evaluate the performances of the trained networks.

## Usage
The code has been tested with the following dependencies:
* [**Python 3.8+**](https://www.python.org/)
* Torch - version 
* [Numpy](https://scipy.org/install.html) - version 
* [Pytorch Image Models (timm)](https://timm.fast.ai/) - version 
* [NetCDF4](https://unidata.github.io/netcdf4-python/) - version 
* [Scikit-Learn](https://scikit-learn.org/stable/install.html) - version
* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html#installation) - version 
* argparse - version

### Installing Packages
```
pip install torch
pip install numpy 
pip install timm
pip install netCDF4
pip install scikit-learn
pip install matplotlib
pip install argparse
```
