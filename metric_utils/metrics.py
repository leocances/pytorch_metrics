import torch
import numpy as np


class Metrics:
    def __init__(self, epsilon=1e-10):
        self.value = None
        self.accumulate_value = 0
        self.count = 0
        self.epsilon = epsilon

    def reset(self):
        self.accumulate_value = 0
        self.count = 0

    def __call__(self, y_pred, y_true):
        self.count += 1


class BinaryAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            y_pred = (y_pred > 0.5).float()
            correct = (y_pred == y_true).float().sum()
            self.value = correct / np.prod(y_true.shape)

            self.accumulate_value += self.value
            return self.accumulate_value / self.count

    @property
    def accuracy(self):
        return self.value


class CategoricalAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            self.value = torch.mean((y_true == y_pred).float())
            self.accumulate_value += self.value

            return self.accumulate_value / self.count

    @property
    def accuracy(self):
        return self.value


class Ratio(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_adv_pred):
        super().__call__(y_pred, y_adv_pred)

        results = zip(y_pred, y_adv_pred)
        results_bool = [int(r[0] != r[1]) for r in results]
        self.value = sum(results_bool) / len(results_bool)
        self.accumulate_value += self.value

        return self.accumulate_value / self.count

    @property
    def ratio(self):
        return self.value


class Precision(Metrics):
    def __init__(self, dim = None, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        self.dim = dim

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            dim = () if self.dim is None else self.dim

            true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)), dim=dim)
            predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)), dim=dim)

            if self.dim is None and predicted_positives == 0:
                self.value = torch.as_tensor(0.0)
            else:
                self.value = true_positives / (predicted_positives + self.epsilon)
                
            self.accumulate_value += self.value
            return self.accumulate_value / self.value


class Recall(Metrics):
    def __init__(self, dim = None, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        self.dim = dim

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            dim = () if self.dim is None else self.dim            
            
            true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0.0, 1.0)), dim=dim)
            possible_positives = torch.sum(torch.clamp(y_true, 0.0, 1.0), dim=dim)
            
            if self.dim is None and possible_positives == 0:
                self.value = torch.as_tensor(0.0)
            else:
                self.value = true_positives / (possible_positives + self.epsilon)
                
            self.accumulate_value += self.value
            return self.accumulate_value / self.count


class FScore(Metrics):
    def __init__(self, dim = None, epsilon=np.spacing(1)):
        Metrics.__init__(self, epsilon)
        self.dim = dim
        
        self.precision_func = Precision(dim, epsilon)
        self.recall_func = Recall(dim, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            dim = () if self.dim is None else self.dim
            
            self.precision = self.precision_func(y_pred, y_true)
            self.recall = self.recall_func(y_pred, y_true)

            if self.dim is None and (self.precision == 0 and self.recall == 0):
                self.value = torch.as_tensor(0.0)
            else:
                self.value = 2 * ((self.precision_func.value * self.recall_func.value) / (self.precision_func.value + self.recall_func.value + self.epsilon))
                
            self.accumulate_value += self.value
            return self.accumulate_value / self.count