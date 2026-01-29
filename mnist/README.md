# Loss of plasticity in vision benchmarks


## Running the code

### Random MNIST

**Unregularized**

`python mnist_abrupt.py --config random_mnist_config.json`

**Interpolation**

`python mnist_gradual.py --config random_mnist_config_smooth.json`

**Other methods**

Update the `"baseline":''` field in abrupt `train_config*.json` with values `spectral_reg`, `l2`, `redo`, `shrink_perturb` to run baselines accordingly. 

`python baselines.py --config random_mnist_config.json`

### Permute EMNIST

Note that the keyword `'permute'` must be included in the `'exp_name'` in the config files to run pixel permutation tasks.

**Unregularized**

`python mnist_abrupt.py --config permute_emnist_config.json`

**Interpolation**

`python mnist_gradual.py --config permute_emnist_config_smooth.json`

**Other methods**

Update the `"baseline":''` field in abrupt `train_config*.json` with values `spectral_reg`, `l2`, `redo`, `shrink_perturb` to run baselines accordingly. 


`python baselines.py --config permute_emnist_config.json`
