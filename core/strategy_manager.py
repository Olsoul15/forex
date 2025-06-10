"""Dummy Strategy Manager for import resolution."""


class StrategyManager:
    def list_strategies(self, *args, **kwargs):
        print("StrategyManager: list_strategies called")
        return []

    def get_strategy(self, strategy_id: str, *args, **kwargs):
        print(f"StrategyManager: get_strategy called for {strategy_id}")
        return {"id": strategy_id, "name": "Dummy Strategy"}

    def get_strategy_performance(self, strategy_id: str, *args, **kwargs):
        print(
            f"StrategyManager: get_strategy_performance called for {strategy_id}"
        )
        return {"pnl": 0, "sharpe": 0}

    def get_recent_signals(self, strategy_id: str, *args, **kwargs):
        print(f"StrategyManager: get_recent_signals called for {strategy_id}")
        return []

    def get_compatible_pairs(self, strategy_id: str, *args, **kwargs):
        print(f"StrategyManager: get_compatible_pairs called for {strategy_id}")
        return []


def get_strategy_manager():
    print("Dummy get_strategy_manager called")
    return StrategyManager()
