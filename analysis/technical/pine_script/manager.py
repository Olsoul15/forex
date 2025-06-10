"""
Pine Script strategy management system.
Handles versioning, storage, and deployment of Pine Script strategies.
"""

import os
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from forex_ai.exceptions import (
    StrategyError,
    StrategyNotFoundError,
    StrategyValidationError,
    DatabaseError,
)
from forex_ai.custom_types import CurrencyPair, TimeFrame, MarketCondition

logger = logging.getLogger(__name__)


class PineScriptStrategy:
    """
    Represents a Pine Script trading strategy with metadata.

    Attributes:
        id: Unique identifier for the strategy
        name: Human-readable name
        description: Detailed description
        author: Strategy author
        version: Strategy version (major.minor.patch)
        code: Pine Script code
        metadata: Additional metadata about the strategy
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        author: str,
        version: str,
        code: str,
        metadata: Dict[str, Any],
        created_at: datetime = None,
        updated_at: datetime = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.author = author
        self.version = version
        self.code = code
        self.metadata = metadata
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    @property
    def supported_pairs(self) -> List[CurrencyPair]:
        """Get currency pairs supported by this strategy."""
        return [
            CurrencyPair(base=p["base"], quote=p["quote"])
            for p in self.metadata.get("supported_pairs", [])
        ]

    @property
    def supported_timeframes(self) -> List[TimeFrame]:
        """Get timeframes supported by this strategy."""
        return [TimeFrame(tf) for tf in self.metadata.get("supported_timeframes", [])]

    def is_compatible_with(
        self, currency_pair: CurrencyPair, timeframe: TimeFrame
    ) -> bool:
        """Check if strategy is compatible with given currency pair and timeframe."""
        return (
            any(
                cp.base == currency_pair.base and cp.quote == currency_pair.quote
                for cp in self.supported_pairs
            )
            and timeframe in self.supported_timeframes
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "code": self.code,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PineScriptStrategy":
        """Create strategy from dictionary."""
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None
        )
        updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None
        )

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            author=data["author"],
            version=data["version"],
            code=data["code"],
            metadata=data["metadata"],
            created_at=created_at,
            updated_at=updated_at,
        )


class PineScriptStrategyManager:
    """
    Manages a repository of Pine Script strategies with version control
    and deployment capabilities.
    """

    def __init__(self, db_connection=None, git_mcp_client=None):
        self.db = db_connection  # PostgreSQL MCP connection
        self.git = git_mcp_client  # Git MCP connection
        self.strategies_dir = os.path.join(os.path.dirname(__file__), "strategies")

        # Create strategies directory if it doesn't exist
        os.makedirs(self.strategies_dir, exist_ok=True)

    def get_strategy(self, strategy_id: str) -> PineScriptStrategy:
        """
        Retrieve a strategy from the repository.

        Args:
            strategy_id: Unique identifier of the strategy.

        Returns:
            PineScriptStrategy object.

        Raises:
            StrategyNotFoundError: If strategy not found.
        """
        try:
            # Try to load from database if connection available
            if self.db:
                strategy_data = self._load_strategy_from_db(strategy_id)
                if strategy_data:
                    return PineScriptStrategy.from_dict(strategy_data)

            # Fall back to file system
            strategy_path = os.path.join(self.strategies_dir, f"{strategy_id}.json")
            if os.path.exists(strategy_path):
                with open(strategy_path, "r") as f:
                    strategy_data = json.load(f)
                return PineScriptStrategy.from_dict(strategy_data)

            # Check for Pine Script file
            pine_path = os.path.join(self.strategies_dir, f"{strategy_id}.pine")
            if os.path.exists(pine_path):
                with open(pine_path, "r") as f:
                    pine_code = f.read()

                # Extract metadata from Pine Script code
                metadata = self._extract_metadata_from_pine(pine_code)

                return PineScriptStrategy(
                    id=strategy_id,
                    name=metadata.get("name", strategy_id),
                    description=metadata.get("description", ""),
                    author=metadata.get("author", "Unknown"),
                    version=metadata.get("version", "1.0.0"),
                    code=pine_code,
                    metadata=metadata,
                )

            raise StrategyNotFoundError(f"Strategy '{strategy_id}' not found")

        except Exception as e:
            if isinstance(e, StrategyNotFoundError):
                raise
            logger.error(f"Error retrieving strategy '{strategy_id}': {str(e)}")
            raise StrategyError(
                f"Failed to retrieve strategy '{strategy_id}': {str(e)}"
            )

    def list_strategies(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[PineScriptStrategy]:
        """
        List available strategies with optional filtering.

        Args:
            filters: Optional dictionary of filter criteria.
                    Supported filters: author, timeframe, currency_pair, market_condition

        Returns:
            List of PineScriptStrategy objects matching filters.
        """
        try:
            strategies = []

            # Try to load from database if connection available
            if self.db:
                db_strategies = self._list_strategies_from_db(filters)
                strategies.extend(
                    [PineScriptStrategy.from_dict(s) for s in db_strategies]
                )

            # Also load from file system
            file_strategies = self._list_strategies_from_files(filters)

            # Merge strategies, prioritizing database versions
            strategy_map = {s.id: s for s in strategies}
            for strategy in file_strategies:
                if strategy.id not in strategy_map:
                    strategies.append(strategy)

            # Apply additional filtering
            if filters:
                strategies = self._apply_filters(strategies, filters)

            return strategies

        except Exception as e:
            logger.error(f"Error listing strategies: {str(e)}")
            raise StrategyError(f"Failed to list strategies: {str(e)}")

    def add_strategy(
        self,
        name: str,
        description: str,
        pine_script_code: str,
        metadata: Dict[str, Any],
        author: str = "Forex AI System",
    ) -> PineScriptStrategy:
        """
        Add a new strategy to the repository.

        Args:
            name: Human-readable name for the strategy.
            description: Detailed description of the strategy.
            pine_script_code: Pine Script code for the strategy.
            metadata: Additional metadata about the strategy.
            author: Strategy author.

        Returns:
            Newly created PineScriptStrategy object.

        Raises:
            StrategyValidationError: If validation fails.
        """
        try:
            # Validate the Pine Script code
            self._validate_pine_script(pine_script_code)

            # Generate ID from name
            strategy_id = self._generate_strategy_id(name)

            # Create version (start with 1.0.0 for new strategies)
            version = "1.0.0"

            # Create strategy object
            strategy = PineScriptStrategy(
                id=strategy_id,
                name=name,
                description=description,
                author=author,
                version=version,
                code=pine_script_code,
                metadata=metadata,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            # Save to database if available
            if self.db:
                self._save_strategy_to_db(strategy)

            # Save to file system (as both JSON and Pine Script file)
            self._save_strategy_to_files(strategy)

            # Optionally, commit to Git repository if available
            if self.git:
                self._commit_strategy_to_git(strategy)

            return strategy

        except Exception as e:
            logger.error(f"Error adding strategy: {str(e)}")
            if isinstance(e, StrategyValidationError):
                raise
            raise StrategyError(f"Failed to add strategy: {str(e)}")

    def update_strategy(
        self,
        strategy_id: str,
        pine_script_code: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PineScriptStrategy:
        """
        Update an existing strategy.

        Args:
            strategy_id: Unique identifier of the strategy to update.
            pine_script_code: Updated Pine Script code.
            metadata: Updated metadata (optional).

        Returns:
            Updated PineScriptStrategy object.

        Raises:
            StrategyNotFoundError: If strategy not found.
            StrategyValidationError: If validation fails.
        """
        try:
            # Retrieve existing strategy
            strategy = self.get_strategy(strategy_id)

            # Validate the updated Pine Script code
            self._validate_pine_script(pine_script_code)

            # Update version (increment patch version)
            major, minor, patch = map(int, strategy.version.split("."))
            new_version = f"{major}.{minor}.{patch + 1}"

            # Update strategy
            strategy.code = pine_script_code
            strategy.version = new_version
            strategy.updated_at = datetime.now()

            if metadata:
                strategy.metadata.update(metadata)

            # Save to database if available
            if self.db:
                self._save_strategy_to_db(strategy)

            # Save to file system
            self._save_strategy_to_files(strategy)

            # Optionally, commit to Git repository if available
            if self.git:
                self._commit_strategy_to_git(strategy)

            return strategy

        except Exception as e:
            logger.error(f"Error updating strategy '{strategy_id}': {str(e)}")
            if isinstance(e, (StrategyNotFoundError, StrategyValidationError)):
                raise
            raise StrategyError(f"Failed to update strategy '{strategy_id}': {str(e)}")

    def deploy_to_trading_view(self, strategy_id: str) -> Dict[str, Any]:
        """
        Deploy a strategy to TradingView via available methods.

        Args:
            strategy_id: Unique identifier of the strategy to deploy.

        Returns:
            Dictionary with deployment details.

        Raises:
            StrategyNotFoundError: If strategy not found.
            StrategyError: If deployment fails.
        """
        try:
            # Retrieve the strategy
            strategy = self.get_strategy(strategy_id)

            # Implementation would depend on TradingView API or other deployment method
            # For now, just return deployment details
            return {
                "status": "simulated",
                "message": "TradingView deployment simulated (actual API not implemented)",
                "strategy_id": strategy.id,
                "version": strategy.version,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error deploying strategy '{strategy_id}': {str(e)}")
            if isinstance(e, StrategyNotFoundError):
                raise
            raise StrategyError(f"Failed to deploy strategy '{strategy_id}': {str(e)}")

    # Helper methods

    def _generate_strategy_id(self, name: str) -> str:
        """Generate a unique ID from strategy name."""
        base_id = re.sub(r"[^a-z0-9_]", "_", name.lower())
        base_id = re.sub(
            r"_{2,}", "_", base_id
        )  # Replace multiple underscores with single
        base_id = base_id.strip("_")

        # Add hash suffix if needed to ensure uniqueness
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]

        return f"{base_id}_{hash_suffix}"

    def _validate_pine_script(self, code: str) -> bool:
        """
        Validate Pine Script code.

        Args:
            code: Pine Script code to validate.

        Returns:
            True if valid.

        Raises:
            StrategyValidationError: If validation fails.
        """
        # This is a basic validation, a real implementation would parse the Pine Script
        # and check for syntax errors, required elements, etc.

        if not code or len(code) < 10:
            raise StrategyValidationError("Pine Script code is too short or empty")

        # Check for basic Pine Script elements
        required_elements = [
            "//@version=",
            "strategy",
        ]

        for element in required_elements:
            if element not in code:
                raise StrategyValidationError(
                    f"Pine Script code missing required element: {element}"
                )

        return True

    def _extract_metadata_from_pine(self, code: str) -> Dict[str, Any]:
        """Extract metadata from Pine Script code comments."""
        metadata = {
            "supported_pairs": [],
            "supported_timeframes": [],
            "market_conditions": [],
        }

        # Parse metadata from comments
        comment_pattern = r"//\s*@([a-zA-Z_]+)\s*:\s*(.+)"
        for line in code.split("\n"):
            match = re.search(comment_pattern, line)
            if match:
                key, value = match.groups()
                if key in ["name", "description", "author", "version"]:
                    metadata[key] = value.strip()
                elif key == "pairs":
                    # Format: @pairs: EUR/USD, GBP/JPY, USD/CAD
                    pairs = [p.strip() for p in value.split(",")]
                    for pair in pairs:
                        if "/" in pair:
                            base, quote = pair.split("/")
                            metadata["supported_pairs"].append(
                                {"base": base.strip(), "quote": quote.strip()}
                            )
                elif key == "timeframes":
                    # Format: @timeframes: 1h, 4h, D
                    metadata["supported_timeframes"] = [
                        tf.strip() for tf in value.split(",")
                    ]
                elif key == "market_conditions":
                    # Format: @market_conditions: trending, ranging, volatile
                    metadata["market_conditions"] = [
                        mc.strip() for mc in value.split(",")
                    ]

        return metadata

    def _save_strategy_to_files(self, strategy: PineScriptStrategy) -> None:
        """Save strategy to file system."""
        # Save as JSON
        json_path = os.path.join(self.strategies_dir, f"{strategy.id}.json")
        with open(json_path, "w") as f:
            json.dump(strategy.to_dict(), f, indent=2)

        # Save as Pine Script
        pine_path = os.path.join(self.strategies_dir, f"{strategy.id}.pine")
        with open(pine_path, "w") as f:
            f.write(strategy.code)

    def _list_strategies_from_files(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[PineScriptStrategy]:
        """List strategies from file system."""
        strategies = []

        # Look for JSON strategy files
        for filename in os.listdir(self.strategies_dir):
            if filename.endswith(".json"):
                try:
                    strategy_id = filename[:-5]  # Remove .json
                    strategy_path = os.path.join(self.strategies_dir, filename)

                    with open(strategy_path, "r") as f:
                        strategy_data = json.load(f)

                    strategies.append(PineScriptStrategy.from_dict(strategy_data))
                except Exception as e:
                    logger.warning(f"Error loading strategy from {filename}: {str(e)}")

        # Also look for Pine Script files without JSON
        for filename in os.listdir(self.strategies_dir):
            if filename.endswith(".pine") and not os.path.exists(
                os.path.join(self.strategies_dir, filename[:-5] + ".json")
            ):
                try:
                    strategy_id = filename[:-5]  # Remove .pine
                    pine_path = os.path.join(self.strategies_dir, filename)

                    with open(pine_path, "r") as f:
                        pine_code = f.read()

                    # Extract metadata from Pine Script code
                    metadata = self._extract_metadata_from_pine(pine_code)

                    strategies.append(
                        PineScriptStrategy(
                            id=strategy_id,
                            name=metadata.get("name", strategy_id),
                            description=metadata.get("description", ""),
                            author=metadata.get("author", "Unknown"),
                            version=metadata.get("version", "1.0.0"),
                            code=pine_code,
                            metadata=metadata,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error loading strategy from {filename}: {str(e)}")

        return strategies

    def _apply_filters(
        self, strategies: List[PineScriptStrategy], filters: Dict[str, Any]
    ) -> List[PineScriptStrategy]:
        """Apply filters to a list of strategies."""
        filtered_strategies = strategies.copy()

        if "author" in filters:
            filtered_strategies = [
                s for s in filtered_strategies if s.author == filters["author"]
            ]

        if "timeframe" in filters:
            timeframe = TimeFrame(filters["timeframe"])
            filtered_strategies = [
                s for s in filtered_strategies if timeframe in s.supported_timeframes
            ]

        if "currency_pair" in filters:
            cp = filters["currency_pair"]
            if not isinstance(cp, CurrencyPair):
                cp = CurrencyPair(base=cp["base"], quote=cp["quote"])
            filtered_strategies = [
                s
                for s in filtered_strategies
                if any(
                    p.base == cp.base and p.quote == cp.quote for p in s.supported_pairs
                )
            ]

        if "market_condition" in filters:
            mc = filters["market_condition"]
            if not isinstance(mc, MarketCondition):
                mc = MarketCondition(mc)
            filtered_strategies = [
                s
                for s in filtered_strategies
                if "market_conditions" in s.metadata
                and mc.value in s.metadata["market_conditions"]
            ]

        return filtered_strategies

    # Database methods (placeholders, would be implemented with actual DB connection)

    def _load_strategy_from_db(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Load strategy from database."""
        if not self.db:
            return None

        # This is a placeholder - would be implemented with actual DB connection
        logger.info(
            f"DB connection available but _load_strategy_from_db not implemented for {strategy_id}"
        )
        return None

    def _list_strategies_from_db(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List strategies from database."""
        if not self.db:
            return []

        # This is a placeholder - would be implemented with actual DB connection
        logger.info(
            "DB connection available but _list_strategies_from_db not implemented"
        )
        return []

    def _save_strategy_to_db(self, strategy: PineScriptStrategy) -> None:
        """Save strategy to database."""
        if not self.db:
            return

        # This is a placeholder - would be implemented with actual DB connection
        logger.info(
            f"DB connection available but _save_strategy_to_db not implemented for {strategy.id}"
        )

    def _commit_strategy_to_git(self, strategy: PineScriptStrategy) -> None:
        """Commit strategy to Git repository."""
        if not self.git:
            return

        # This is a placeholder - would be implemented with actual Git client
        logger.info(
            f"Git client available but _commit_strategy_to_git not implemented for {strategy.id}"
        )
