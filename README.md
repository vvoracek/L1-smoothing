# Improving $\ell_1$-Certified Robustness via Randomized Smoothing by Leveraging Box Constraints

Code for the paper "Improving $\ell_1$-Certified Robustness via Randomized Smoothing by Leveraging Box Constraints"

Most of the code is reused from (Levine 2021) https://github.com/alevine0/smoothingSplittingNoise (and thus from (Yang et al. 2020) available at https://github.com/tonyduan/rs4a)

Instructions for installing dependencies are reproduced here:

```
conda install numpy matplotlib pandas seaborn 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install torchnet tqdm statsmodels dfply
```

The train -> certification pipeline is  in  ```main.py``` 

For imagenet experiments, set the environment variables $IMAGENET_TRAIN_DIR and $IMAGENET_TEST_DIR.

