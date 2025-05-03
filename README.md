# Setup

```python
pip install -r requirements.txt
```

Based on platform you might have to follow the [PyTorch setup process](https://pytorch.org/get-started/locally/) to enable CUDA support.

Run scripts as modules:

```python
python -m optimization.optimization
```

# Training Loop

Use

```python
python -m optimization.optimization
```

to run the training loop. Configure run and system params in `optimization/optimization.py`.

Neptune logging is configured. To use it, set

```bash
export NEPTUNE_API_TOKEN="..."
```

in .env.
