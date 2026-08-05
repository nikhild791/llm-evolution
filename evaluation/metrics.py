from collections import defaultdict
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get(self, key):
        return self.metrics[key]

    def plot(self, train_metric, val_metric=None):

        plt.figure(figsize=(7,5))

        plt.plot(
            self.metrics[train_metric],
            label=train_metric.replace("_", " ").title(),
            linewidth=2
        )

        if val_metric is not None:
            plt.plot(
                self.metrics[val_metric],
                label=val_metric.replace("_", " ").title(),
                linestyle="--",
                linewidth=2
            )

        plt.title(train_metric.replace("train_", "").replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()