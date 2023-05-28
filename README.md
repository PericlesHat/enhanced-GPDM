# enhanced-GPDM

We present enhanced Gaussian Process Dynamical Model, a Bayesian method used for capturing high-dimension data's dynamics and transfer learning ability. This repo is the official implementation of Enhanced Gaussian process dynamical models with knowledge transfer for long-term battery degradation forecasting. Our code is based on PyTorch with CUDA supported.

<img width="450" height="300" src="https://raw.githubusercontent.com/PericlesHat/modified-GPDM/main/assets/gpdm_nasa.png"/>

## TO-DO

- add mGPDM model

## Prerequisites

- NumPy
- Pandas
- Matplotlib
- SciPy
- scikit-learn
- PyTorch

## Data preparation

We provide several processed NASA batteries' data (B. Saha and K. Goebel, 2007) for demonstration. The original and whole datasets we used in the paper can be downloaded from: [NASA dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and [Oxford dataset](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac).

## Train GPDM

The script `train_tran.py` is a **ready-to-run** demo. We demonstrate the GPDM training and testing process along with the transfer learning ability. Please refer to the detailed comments in the code. 

Try to setup different hyperparameters to evaluate the model:
- `D`: observation space dimension (determined by observation data)
- `Q`: desired latent space dimension, empirically $Q << D$
- `dyn_target`: choose `full` or `delta`, `delta` models higher order feature by defining each latent point as $x_t - x_{t-1}$

## License

This project is licensed under the GNU General Public License v3.0.

## Acknowledgments

This work partly used the code from [CIGP](https://github.com/fabio-amadio/cgpdm_lib) and [CGPDM](https://github.com/fabio-amadio/cgpdm_lib).

## Cite

If you find our work useful, please cite our paper.
```
incoming...
```
