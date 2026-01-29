# Loss of Plasticity in Language Model

## How to run the experiments?
`--config`: path to the config files containing basic hyperparameters such as learning rate

`--smooth`: boolean operator determining whether to run with Task Sampling

### Random Seq2seq

**Unregularized**

`python trainer.py --config train_config.json`

**Task Sampling**

`python trainer.py --config train_config_smooth.json --smooth`

**Other mitigation methods**

Spectral Regularization

`baselines.py --config train_config.json --baseline spectral_reg --reg_strength 0.001`

Shrink&Perturb

`baselines.py --config train_config.json --baseline shrink_perturb --shrink 0.5 --perturb 0.01`

L2 Regularization

`python baselines.py --config train_config.json --baseline l2_reg --reg_strength 0.0000001`

ReDO

`python baselines.py --config train_config.json --baseline redo`

### Bigram Cipher

Note that the keyword `'cipher'` must be included in the `'exp_name'` in the config files to run this task.

**Unregularized**

`python trainer.py --config train_config_cipher.json`

**Task Sampling**

`python trainer.py --config train_config_cipher_smooth.json --smooth`

**Other mitigation methods**

Spectral Regularization

`baselines.py --config train_config_cipher_baselines.json --baseline spectral_reg --reg_strength 0.001`

Shrink&Perturb

`baselines.py --config train_config_cipher_baselines.json --baseline shrink_perturb --shrink 0.5 --perturb 0.01`

L2 Regularization

`python baselines.py --config train_config_cipher_baselines.json --baseline l2_reg --reg_strength 0.0000001`

ReDO

`python baselines.py --config train_config_cipher_baselines.json --baseline redo`
