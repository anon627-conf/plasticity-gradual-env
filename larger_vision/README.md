# Loss of plasticity in vision benchmarks

## Running the code

**Unregularized**

- Cifar10

`python train_abrupt.py --config random_imagenet_config.json`

- Imagenet
  
`python train_abrupt.py --config random_imagenet_config.json`

**Interpolation**

- Cifar10

`python train_gradual.py --config random_cifar_config_smooth.json`

- Imagenet
  
`python train_abrupt.py --config random_imagenet_config_smooth.json`

**Other methods**

Update the `"baseline":''` field in abrupt `train_config*.json` with values `spectral_reg`, `l2`, `redo`, `shrink_perturb` to run baselines accordingly. 

i.e. For Imagenet run:

`python baselines.py --config random_imagenet_config.json`
