# enhanced-GPDM

We present enhanced Gaussian Process Dynamical Model (EGPDM), a Bayesian method used for capturing high-dimension data's dynamics and transfer learning ability. This repo is the official implementation of [Enhanced Gaussian Process Dynamical Models with Knowledge Transfer for Long-term Battery Degradation Forecasting](https://arxiv.org/abs/2212.01609). Our code is based on PyTorch with CUDA supported.

We also provide an original GPDM and its tutorial in [IceLab's repo](https://github.com/IceLab-X/Mini-GP).

<img width="700" height="300" src="https://raw.githubusercontent.com/PericlesHat/modified-GPDM/main/assets/egpdm_nasa.png"/>

## TO-DO

- [x] add EGPDM model
- [x] update: calculate the original RMSE instead of normalized
- [x] fix bugs: 3D-trajectory plot; kernel noises
- [ ] fix bugs: cycle limitation in LBFGS

## Prerequisites

- NumPy
- Pandas
- Matplotlib
- SciPy
- scikit-learn
- PyTorch

## Data preparation

We provide several processed NASA batteries' data (B. Saha and K. Goebel, 2007) for demonstration. The original and whole datasets we used in the paper can be downloaded from: [NASA dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and [Oxford dataset](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac).

## Train EGPDM

The script `train_tran.py` is a **ready-to-run** demo. We demonstrate the GPDM training and testing process along with the transfer learning ability. Please refer to the detailed comments in the code. 

Try to setup different hyperparameters to evaluate the model:
- **D:** observation space dimension (determined by observation data)
- **Q:** desired latent space dimension, empirically $Q << D$
- **dyn_target:** `full` or `delta`, `delta` models higher order feature by defining latent points as $x_t - x_{t-1}$

We offer linear, RBF, Matern3 and Matern5 kernel functions in the code. You can custom your desired kernels in `self.observationGP_kernel()` and `self.dynamicGP_kernel()` using linear combination of kernels.

Note that our model initializes most of the learnable parameters to $1$. If you want a more random initialization, set the parameters to `torch.randn()` and use a random seed to control.

## License

This project is licensed under the GNU General Public License v3.0.

## Acknowledgments

This work partly uses the code from [CIGP](https://github.com/IceLab-X/Mini-GP/blob/main/cigp_v14.py) and [CGPDM](https://github.com/fabio-amadio/cgpdm_lib).

## Cite

If you find our work useful, please cite our paper.
```
@article{xing2022enhanced,
  title={Enhanced Gaussian Process Dynamical Models with Knowledge Transfer for Long-term Battery Degradation Forecasting},
  author={Xing, Wei W and Zhang, Ziyang and Shah, Akeel A},
  journal={arXiv e-prints},
  pages={arXiv--2212},
  year={2022}
}
```
