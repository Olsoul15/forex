"""Dummy Backtest Manager for import resolution."""


class BacktestManager:
    def run_backtest(self, *args, **kwargs):
        print("DummyBacktestManager: run_backtest called")
        return {"net_profit": 0, "summary": "Dummy backtest complete"}

    # Add other methods if StrategyTool calls them on _backtest_manager


def get_backtest_manager():
    print("Dummy get_backtest_manager called")
    return BacktestManager()
