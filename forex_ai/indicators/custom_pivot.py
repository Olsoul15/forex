import backtrader as bt


class CustomPivotPoint(bt.ind.PivotPoint):
    """Custom PivotPoint indicator that explicitly sets a minimal _minperiod
    to potentially avoid issues with backtrader's internal processing.

    Access the pivot lines directly via the lines attribute:
    - instance.lines.p
    - instance.lines.s1
    - etc.
    """

    _minperiod = 1  # Set a minimal period

    def __init__(self):
        # Call the parent init to perform the actual calculations
        super().__init__()
        # Optional: Log that the custom indicator is being used
        # print(f"CustomPivotPoint Initialized with minperiod: {self._minperiod}")

    # Unlike the property approach, here we simply let users access the lines directly
    # This is the normal way to access indicator lines in backtrader
