# pytorch_metrics
Custom implementation of some metrics I use with my PyTorch model

This is a repository for my personnal usage and you can use as it comes.

The functions returns the moving average of the given vaue. To access the raw value of the current mini-batch use the value attribute.

## How to use
```python
from pytorch_metrics.metrics import FScore
...

fscore_func = FScore(epsilon=1e-6)

...

fscore = fscore_func(pred, true)

# get the current value of the metric
fscore_value = fscore.value

# get the value of the running mean
fscore_running_mean = fscore.mean

# get the std of the metric at the current state
fscore_std = fscore.std

# reset all value (use at the begining of the epoch)
fscore_func.reset()

```

## Metrics available
It contain the following metrics:
- CategoricalAccuracy
- BinaryAccuracy
- Recall
- Precision
- FSCore (F1)
- Ratio (Evaluate the truth number of adversarial generated)

