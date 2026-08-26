# TSFM-pretrain

This repo contains code to easily pretrain Time Series Foundation Models.

# Starting training.

> [!TIP]
> Before training a model, check the `config.yaml` file to specify the right device.
>  
> You will also need to generate synthetic data (See below on how to do so).

**You will get an error otherwise.**

To start the training, simply run :

```bash
python trainer.py
```

The script supports multi-gpu and DDP training, to do so, launch it with `torchrun` :

```bash
torchrun --n-proc-per-node=<N_GPUS> trainer.py
```

By default, the script trains a `Chronos-2` model on synthetic data.

# Synthetic data generation

To generate data, we use the `kernel_synth` introduced by the `Chronos` team. Run the following script to generate : 

```bash
python kernel_synth -N <N_SERIES> -J <N_KERNELS>
```

This will create a file `kernelsynth-data.arrow`.
