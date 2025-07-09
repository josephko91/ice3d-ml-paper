# ice3d-ml-paper
Training and inference code for: A Machine Learning Framework for Predicting Microphysical Properties of Ice Crystals from Cloud Particle Imagery

# Requirements / Hardware / Environment
A copy of the conda environment that was used to train/eval the ML models can be found in `torch-env.yaml`. 

Training tested with following compute:
- 64-bit Ubuntu Linux OS, NVIDIA a100 GPU

Inference tested with following compute:
- 64-bit Ubuntu Linux OS, NVIDIA a100 GPU
- MacBook Pro (Apple Silicon), macOS Sonoma 14.5, Darwin 23.5.0 (arm64), CPU-only

# Description of contents
`data`: Contains python modules for pytorch lightning datamodules and torch datasets <br>
`models`: Contains python modules for torch models <br>
`train-skl.ipynb`: Code to train scikit-learn models from the paper <br>
`train-torch.ipynb`: Code to train torch models from the paper <br>
`eval-reproduce-results.ipynb`: Code to run models in inference on test data and to reproduce figures from paper <br>
