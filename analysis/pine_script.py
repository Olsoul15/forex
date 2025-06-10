"""
Pine Script Integration Module for the AI Forex Trading System.

This module implements functionality to parse, extract parameters from, and execute
Pine Script strategies. It provides a bridge between the AI system and existing
TradingView Pine Script strategies.
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime
import uuid
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import PineScriptError

logger = get_logger(__name__)


class PineScriptVersion(Enum):
    """Enumeration of supported Pine Script versions."""

    V2 = "2"
    V3 = "3"
    V4 = "4"
    V5 = "5"


@dataclass
class PineScriptParameter:
    """Data class for Pine Script input parameters."""

    name: str
    type: str
    default_value: Any
    display_name: Optional[str] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    options: List[Any] = field(default_factory=list)
    group: Optional[str] = None
    tooltip: Optional[str] = None


@dataclass
class PineScriptStrategy:
    """Data class for Pine Script strategy metadata and content."""

    name: str
    content: str
    version: PineScriptVersion
    parameters: List[PineScriptParameter] = field(default_factory=list)
    unique_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[Path] = None
    author: Optional[str] = None
    is_active: bool = True

    def __post_init__(self):
        """Generate a hash of the content if unique_id is not provided."""
        if self.unique_id == str(uuid.uuid4()):
            self.unique_id = hashlib.md5(self.content.encode()).hexdigest()


class PineScriptParser:
    """Parser for Pine Script strategies to extract metadata and parameters."""

    def __init__(self):
        """Initialize the Pine Script parser."""
        self.version_pattern = re.compile(r"\/\/@version=(\d+)")
        self.strategy_name_pattern = re.compile(r'strategy\((?:title=)?"([^"]+)"')
        self.study_name_pattern = re.compile(r'study\((?:title=)?"([^"]+)"')
        self.indicator_name_pattern = re.compile(r'indicator\((?:title=)?"([^"]+)"')
        self.input_pattern = re.compile(
            r'(input(?:\.\w+)?)\s*\((?:title=)?"([^"]+)",\s*(?:type=)?(\w+)(?:\.(\w+))?(?:,\s*defval=)?([^,\)]+)(?:.*?)(?:\)|,\s*minval=([\d\.]+))?(?:.*?)(?:\)|,\s*maxval=([\d\.]+))?(?:.*?)(?:\)|,\s*options=(\[[^\]]+\]))?(?:.*?)(?:\)|,\s*group=?\"([^\"]+)\")?(?:.*?)(?:\)|,\s*tooltip=?\"([^\"]+)\")?'
        )

    def parse_from_file(self, file_path: Union[str, Path]) -> PineScriptStrategy:
        """
        Parse a Pine Script strategy from a file.

        Args:
            file_path: Path to the Pine Script file

        Returns:
            PineScriptStrategy object containing parsed strategy

        Raises:
            PineScriptError: If parsing fails
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise PineScriptError(f"Pine Script file not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            strategy = self.parse_from_string(content)
            strategy.source_file = file_path

            return strategy
        except Exception as e:
            logger.error(f"Error parsing Pine Script file: {str(e)}")
            raise PineScriptError(f"Failed to parse Pine Script file: {str(e)}") from e

    def parse_from_string(self, content: str) -> PineScriptStrategy:
        """
        Parse a Pine Script strategy from a string.

        Args:
            content: String containing Pine Script code

        Returns:
            PineScriptStrategy object containing parsed strategy

        Raises:
            PineScriptError: If parsing fails
        """
        try:
            # Extract version
            version_match = self.version_pattern.search(content)
            version = PineScriptVersion.V4  # Default to v4 if not specified
            if version_match:
                version_str = version_match.group(1)
                try:
                    version = PineScriptVersion(version_str)
                except ValueError:
                    logger.warning(
                        f"Unsupported Pine Script version: {version_str}, defaulting to v4"
                    )

            # Extract name
            name = None
            for pattern in [
                self.strategy_name_pattern,
                self.study_name_pattern,
                self.indicator_name_pattern,
            ]:
                name_match = pattern.search(content)
                if name_match:
                    name = name_match.group(1)
                    break

            if not name:
                name = "Unnamed Pine Script Strategy"
                logger.warning(
                    f"Could not extract strategy name, using default: {name}"
                )

            # Extract parameters
            parameters = self._extract_parameters(content)

            # Create strategy object
            strategy = PineScriptStrategy(
                name=name, content=content, version=version, parameters=parameters
            )

            # Extract description from comments at the top
            description_lines = []
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("//") and not line.startswith("//@"):
                    description_lines.append(line[2:].strip())
                elif not line.startswith("//") and description_lines:
                    break

            if description_lines:
                strategy.description = "\n".join(description_lines)

            # Extract author if available
            author_match = re.search(r"\/\/\s*©\s*([^\n]+)", content)
            if author_match:
                strategy.author = author_match.group(1).strip()

            return strategy
        except Exception as e:
            logger.error(f"Error parsing Pine Script content: {str(e)}")
            raise PineScriptError(
                f"Failed to parse Pine Script content: {str(e)}"
            ) from e

    def _extract_parameters(self, content: str) -> List[PineScriptParameter]:
        """
        Extract input parameters from Pine Script content.

        Args:
            content: String containing Pine Script code

        Returns:
            List of PineScriptParameter objects
        """
        parameters = []

        # Find all input declarations
        input_matches = self.input_pattern.finditer(content)

        for match in input_matches:
            (
                input_func,
                display_name,
                param_type,
                param_subtype,
                default_value,
                min_value,
                max_value,
                options_str,
                group,
                tooltip,
            ) = match.groups()

            # Clean up default value
            if default_value:
                default_value = default_value.strip()

                # Handle string values
                if default_value.startswith('"') and default_value.endswith('"'):
                    default_value = default_value[1:-1]
                # Handle numeric values
                elif default_value.replace(".", "", 1).isdigit():
                    if "." in default_value:
                        default_value = float(default_value)
                    else:
                        default_value = int(default_value)
                # Handle boolean values
                elif default_value.lower() in ["true", "false"]:
                    default_value = default_value.lower() == "true"

            # Parse options if available
            options = []
            if options_str:
                try:
                    # Simple cleaning to make it JSON-compatible
                    options_str = options_str.replace("'", '"')
                    options = json.loads(options_str)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse options string: {options_str}")

            # Create parameter
            parameter = PineScriptParameter(
                name=display_name.lower().replace(" ", "_"),
                display_name=display_name,
                type=param_type,
                default_value=default_value,
                min_value=float(min_value) if min_value else None,
                max_value=float(max_value) if max_value else None,
                options=options,
                group=group,
                tooltip=tooltip,
            )

            parameters.append(parameter)

        return parameters


class StrategyRepository:
    """Repository for storing and retrieving Pine Script strategies."""

    def __init__(self, storage_dir: Union[str, Path]):
        """
        Initialize the strategy repository.

        Args:
            storage_dir: Directory to store strategy files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PineScriptParser()
        self._strategies = {}
        self._load_strategies()

    def _load_strategies(self):
        """Load all strategies from the storage directory."""
        try:
            for file_path in self.storage_dir.glob("*.pine"):
                strategy = self.parser.parse_from_file(file_path)
                self._strategies[strategy.unique_id] = strategy

            logger.info(f"Loaded {len(self._strategies)} Pine Script strategies")
        except Exception as e:
            logger.error(f"Error loading strategies: {str(e)}")

    def add_strategy(
        self, strategy: Union[PineScriptStrategy, str, Path]
    ) -> PineScriptStrategy:
        """
        Add a strategy to the repository.

        Args:
            strategy: Either a PineScriptStrategy object, a path to a .pine file, or a string with Pine Script code

        Returns:
            The added PineScriptStrategy

        Raises:
            PineScriptError: If strategy addition fails
        """
        try:
            # Parse strategy if it's a string or path
            if isinstance(strategy, (str, Path)):
                if (
                    isinstance(strategy, str)
                    and not strategy.endswith(".pine")
                    and "\n" in strategy
                ):
                    # It's a content string
                    strategy = self.parser.parse_from_string(strategy)
                else:
                    # It's a file path
                    strategy = self.parser.parse_from_file(strategy)

            # Save strategy to disk
            file_path = self.storage_dir / f"{strategy.unique_id}.pine"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(strategy.content)

            # Update source file in strategy
            strategy.source_file = file_path

            # Add to in-memory cache
            self._strategies[strategy.unique_id] = strategy

            return strategy
        except Exception as e:
            logger.error(f"Error adding strategy: {str(e)}")
            raise PineScriptError(f"Failed to add strategy: {str(e)}") from e

    def get_strategy(self, strategy_id: str) -> Optional[PineScriptStrategy]:
        """
        Get a strategy by its ID.

        Args:
            strategy_id: Unique ID of the strategy

        Returns:
            PineScriptStrategy if found, None otherwise
        """
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> List[PineScriptStrategy]:
        """
        Get all strategies in the repository.

        Returns:
            List of all PineScriptStrategy objects
        """
        return list(self._strategies.values())

    def search_strategies(self, query: str) -> List[PineScriptStrategy]:
        """
        Search for strategies matching the query.

        Args:
            query: Search query string (matches against name, description, author)

        Returns:
            List of matching PineScriptStrategy objects
        """
        query = query.lower()
        results = []

        for strategy in self._strategies.values():
            # Check name
            if query in strategy.name.lower():
                results.append(strategy)
                continue

            # Check description
            if strategy.description and query in strategy.description.lower():
                results.append(strategy)
                continue

            # Check author
            if strategy.author and query in strategy.author.lower():
                results.append(strategy)
                continue

        return results

    def update_strategy(self, strategy_id: str, content: str) -> PineScriptStrategy:
        """
        Update an existing strategy.

        Args:
            strategy_id: Unique ID of the strategy to update
            content: New Pine Script content

        Returns:
            Updated PineScriptStrategy

        Raises:
            PineScriptError: If strategy not found or update fails
        """
        try:
            if strategy_id not in self._strategies:
                raise PineScriptError(f"Strategy not found: {strategy_id}")

            # Parse new content
            new_strategy = self.parser.parse_from_string(content)

            # Preserve ID and metadata
            old_strategy = self._strategies[strategy_id]
            new_strategy.unique_id = strategy_id
            new_strategy.created_at = old_strategy.created_at
            new_strategy.updated_at = datetime.now()
            new_strategy.source_file = old_strategy.source_file

            # Save updated strategy to disk
            file_path = self.storage_dir / f"{strategy_id}.pine"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Update in-memory cache
            self._strategies[strategy_id] = new_strategy

            return new_strategy
        except Exception as e:
            logger.error(f"Error updating strategy: {str(e)}")
            raise PineScriptError(f"Failed to update strategy: {str(e)}") from e

    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Delete a strategy from the repository.

        Args:
            strategy_id: Unique ID of the strategy to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if strategy_id not in self._strategies:
                logger.warning(f"Strategy not found for deletion: {strategy_id}")
                return False

            # Delete file from disk
            strategy = self._strategies[strategy_id]
            if strategy.source_file and strategy.source_file.exists():
                strategy.source_file.unlink()

            # Remove from in-memory cache
            del self._strategies[strategy_id]

            return True
        except Exception as e:
            logger.error(f"Error deleting strategy: {str(e)}")
            return False


class PineScriptExecutor:
    """Executor for running Pine Script strategies on OHLCV data."""

    def __init__(self, strategy: PineScriptStrategy, parameters: Dict[str, Any] = None):
        """
        Initialize the Pine Script executor.

        Args:
            strategy: PineScriptStrategy to execute
            parameters: Optional dictionary of parameter overrides
        """
        self.strategy = strategy
        self.parameters = parameters or {}
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate that all provided parameters exist in the strategy."""
        for param_name, param_value in self.parameters.items():
            param_exists = any(p.name == param_name for p in self.strategy.parameters)
            if not param_exists:
                logger.warning(
                    f"Parameter '{param_name}' not found in strategy '{self.strategy.name}'"
                )

    def generate_parameter_set(self) -> Dict[str, Any]:
        """
        Generate a complete parameter set with defaults and overrides.

        Returns:
            Dictionary mapping parameter names to values
        """
        param_set = {}

        # Start with default values
        for param in self.strategy.parameters:
            param_set[param.name] = param.default_value

        # Apply overrides
        for param_name, param_value in self.parameters.items():
            if param_name in param_set:
                param_set[param_name] = param_value

        return param_set

    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the strategy on the provided data.

        This is a placeholder that simulates executing the Pine Script.
        In a real implementation, this would either:
        1. Use a Pine Script runtime (if available)
        2. Translate the Pine Script to Python (advanced)
        3. Send the script to TradingView for execution via API (if available)

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary containing execution results

        Raises:
            PineScriptError: If execution fails
        """
        try:
            # This is a simplified placeholder implementation
            # In a real system, you'd need actual Pine Script execution

            logger.info(
                f"Executing strategy '{self.strategy.name}' with {len(data)} rows of data"
            )

            # Generate signals based on pattern detection
            # This is a very basic approximation - real execution would run the actual Pine Script
            from .technical.patterns import detect_candlestick_patterns, PatternType

            # Get parameters with overrides
            params = self.generate_parameter_set()

            # Extract any relevant parameters for pattern detection
            fib_level = params.get("fib_level", 0.333)
            atr_min = params.get("atr_min_filter_size", 0.0)
            atr_max = params.get("atr_max_filter_size", 3.0)

            # Detect patterns that might be used in the strategy
            patterns = detect_candlestick_patterns(
                data,
                patterns=[
                    PatternType.HAMMER,
                    PatternType.SHOOTING_STAR,
                    PatternType.ENGULFING,
                ],
                fib_level=fib_level,
                atr_min_filter=atr_min,
                atr_max_filter=atr_max,
            )

            # Generate mock execution results
            signals = []
            for p in patterns:
                signals.append(
                    {
                        "index": p.index,
                        "date": (
                            data.index[p.index]
                            if hasattr(data.index, "__getitem__")
                            else None
                        ),
                        "pattern": p.pattern_type.value,
                        "direction": p.direction.value,
                        "confidence": p.confidence,
                        "price": data.iloc[p.index]["close"],
                    }
                )

            # Calculate simple performance metrics
            entry_signals = [s for s in signals if s["confidence"] > 0.6]
            win_rate = 0.55  # Placeholder win rate
            avg_profit = 0.8  # Placeholder average profit percentage

            return {
                "strategy_name": self.strategy.name,
                "parameters": params,
                "signals": signals,
                "entry_signals": entry_signals,
                "performance": {
                    "signal_count": len(entry_signals),
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                },
            }
        except Exception as e:
            logger.error(f"Error executing Pine Script strategy: {str(e)}")
            raise PineScriptError(f"Failed to execute strategy: {str(e)}") from e
