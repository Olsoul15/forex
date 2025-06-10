"""Dummy Optimizer for import resolution."""


class Optimizer:
    def optimize_strategy(self, *args, **kwargs):
        print("DummyOptimizer: optimize_strategy called")
        return {"best_params": {}, "best_performance": 0}

    # Add other methods if StrategyTool calls them on _optimizer


def get_optimizer():
    print("Dummy get_optimizer called")
    return Optimizer()
