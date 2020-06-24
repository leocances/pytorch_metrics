import torch
import numpy as np


class Metrics:
    def __init__(self, epsilon=1e-10):
        self.value_ = None
        self.accumulate_value = 0
        self.count = 0
        self.epsilon = epsilon

    def reset(self):
        self.accumulate_value = 0
        self.count = 0

    def __call__(self, y_pred, y_true):
        self.count += 1
        
    @property
    def value(self):
        return self.value_

    @property
    def mean(self):
        return self.accumulate_value / self.count


class FuncContinueAverage(Metrics):
    def __init__(self, func, epsilon=1e-10):
        super().__init__(epsilon)
        self.func = func

    def __call__(self, *args, **kwargs):
        super().__call__(None, None)
        self.value_ = self.func(*args, **kwargs)
        self.accumulate_value += self.value_

        return self


class ContinueAverage(Metrics):
    def __init__(self, epsilon=1e-10):
        super().__init__(epsilon)

    def __call__(self, value):
        super().__call__(None, None)
        self.value_ = value
        self.accumulate_value += self.value_

        return self.accumulate_value / self.count



class BinaryAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            y_pred = (y_pred > 0.5).float()
            correct = (y_pred == y_true).float().sum()
            self.value_ = correct / np.prod(y_true.shape)

            self.accumulate_value += self.value_
            return self.accumulate_value / self.count

    @property
    def accuracy(self):
        return self.value_


class CategoricalAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            self.value_ = torch.mean((y_true == y_pred).float())
            self.accumulate_value += self.value_

            return self.accumulate_value / self.count

    @property
    def accuracy(self):
        return self.value_


class Ratio(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_adv_pred):
        super().__call__(y_pred, y_adv_pred)

        results = zip(y_pred, y_adv_pred)
        results_bool = [int(r[0] != r[1]) for r in results]
        self.value_ = sum(results_bool) / len(results_bool)
        self.accumulate_value += self.value_

        return self.accumulate_value / self.count

    @property
    def ratio(self):
        return self.value_


class BinaryRatio(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        
    def __call__(self, y_pred, y_true):
        nb_pred = torch.sum(y_pred)
        nb_true = torch.sum(y_true)
        
        self.value_ = nb_pred / nb_true
        self.accumulate_value += self.value_
        
        return self.accumulate_value / self.count
    
    
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
                self.value_ = torch.as_tensor(0.0)
            else:
                self.value_ = true_positives / (predicted_positives + self.epsilon)
                
            self.accumulate_value += self.value_
            return self.accumulate_value / self.count

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
                self.value_ = torch.as_tensor(0.0)
            else:
                self.value_ = true_positives / (possible_positives + self.epsilon)
                
            self.accumulate_value += self.value_
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
                self.value_ = torch.as_tensor(0.0)
            else:
                self.value_ = 2 * ((self.precision_func.value * self.recall_func.value) / (self.precision_func.value + self.recall_func.value + self.epsilon))
                
            self.accumulate_value += self.value_
            return self.accumulate_value / self.count


if __name__ == '__main__':
    def test(i):
        return i

    ca_func = ContinueAverage(test)

    for i in range(100):
        ca = ca_func(i).mean

    print(ca)
    print(sum(range(100)) / 100)
