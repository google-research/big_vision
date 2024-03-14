# Our fork of Big vision

## Installation

[Download conda](https://docs.google.com/document/d/1rZF1h_kyURTS4H652jaMfRYcLcBTOzLm8NkK01RGNLw/edit?usp=sharing) and:
```
conda create --name bv python=3.10
conda activate bv


cd big_vision
pip install --upgrade pip
pip install -r big_vision/requirements.txt

pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Test if the setup works - this runs simple code on two nodes:

```
sbatch test.slurm
sleep 60
cat $(ls output/bv_test-*.out | sort -t '-' -k2,2n | tail -n 1)
# Make sure it does not error.
```