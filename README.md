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

Neptune logging is configured. To use it, set

```bash
export NEPTUNE_API_TOKEN="..."
```

in .env.
