"""
Market State Workflow for the Advanced Orchestrator

This module defines workflows for integrating market state detection with the
Advanced Orchestrator architecture, allowing market state analysis to influence
other agents and decision-making processes.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio

# Try to import from both possible locations
try:
    from auto_agent.app.agent.manager_agents.advanced_orchestrator import (
        AdvancedOrchestrator,
        Task,
        TaskPriority,
        TaskStatus,
    )
except ImportError:
    try:
        from forex_ai.agents.advanced_orchestrator import (
            AdvancedOrchestrator,
            Task,
            TaskPriority,
            TaskStatus,
        )
    except ImportError:
        # Define minimal versions if neither import works
        class TaskPriority:
            HIGH = 1
            MEDIUM = 2
            LOW = 3

        class TaskStatus:
            PENDING = "pending"
            IN_PROGRESS = "in_progress"
            COMPLETED = "completed"
            FAILED = "failed"

        class Task:
            def __init__(self, **kwargs):
                self.id = kwargs.get("id", "task_id")
                self.title = kwargs.get("title", "Task Title")
                self.description = kwargs.get("description", "Task Description")
                self.priority = kwargs.get("priority", TaskPriority.MEDIUM)
                self.status = kwargs.get("status", TaskStatus.PENDING)

        class AdvancedOrchestrator:
            async def add_task(self, *args, **kwargs):
                pass


from forex_ai.agents.market_state_analysis_agent import (
    create_market_state_analysis_agent,
)

logger = logging.getLogger(__name__)


class MarketStateWorkflow:
    """
    Workflow for integrating market state detection into the Advanced Orchestrator.

    This class provides methods to register market state analysis tasks, establish
    dependencies between market state analysis and other tasks, and use market
    state information to guide the orchestrator's decision-making process.
    """

    def __init__(self, orchestrator: AdvancedOrchestrator):
        """
        Initialize the market state workflow.

        Args:
            orchestrator: The AdvancedOrchestrator instance to integrate with
        """
        self.orchestrator = orchestrator
        self.market_state_agent_id = None

        # Cache of recent market states for quick access
        self.recent_states: Dict[str, Dict[str, Any]] = {}

        logger.info("Market State Workflow initialized")

    async def register_market_state_agent(self) -> str:
        """
        Register the MarketStateAnalysisAgent with the orchestrator.

        Returns:
            Agent ID of the registered agent
        """
        # Create the agent
        agent = create_market_state_analysis_agent()

        # Register the agent with the orchestrator
        try:
            # Method might vary depending on orchestrator implementation
            if hasattr(self.orchestrator, "register_agent"):
                agent_id = await self.orchestrator.register_agent(agent)
            elif hasattr(self.orchestrator, "mas") and hasattr(
                self.orchestrator.mas, "register_agent"
            ):
                agent_id = await self.orchestrator.mas.register_agent(agent)
            else:
                # Fallback with a default ID
                agent_id = "market_state_analysis_agent"
                logger.warning(
                    "Could not register agent with orchestrator. Using default ID."
                )
        except Exception as e:
            logger.error(f"Error registering MarketStateAnalysisAgent: {e}")
            agent_id = "market_state_analysis_agent"

        # Store the agent ID for future use
        self.market_state_agent_id = agent_id

        # Register the agent's capabilities
        if hasattr(self.orchestrator, "register_agent_capabilities"):
            await self.orchestrator.register_agent_capabilities(
                agent_id,
                ["market_state_detection", "market_analysis", "technical_analysis"],
            )

        logger.info(f"Registered MarketStateAnalysisAgent with ID: {agent_id}")
        return agent_id

    async def create_market_state_analysis_task(
        self,
        pair: str,
        timeframe: str,
        features: Dict[str, Any],
        priority: int = TaskPriority.MEDIUM,
        dependencies: List[str] = None,
    ) -> str:
        """
        Create a task to analyze the market state for a specific pair and timeframe.

        Args:
            pair: Currency pair to analyze
            timeframe: Timeframe to analyze
            features: Pre-calculated features to use for analysis
            priority: Task priority
            dependencies: List of task IDs that must complete before this task

        Returns:
            Task ID of the created task
        """
        if not self.market_state_agent_id:
            await self.register_market_state_agent()

        # Create the task
        task_id = await self.orchestrator.add_task(
            title=f"Market State Analysis: {pair}/{timeframe}",
            description=f"Analyze the market state for {pair} on {timeframe} timeframe",
            priority=priority,
            assigned_agent=self.market_state_agent_id,
            prerequisites=dependencies or [],
            required_capabilities=["market_state_detection"],
            context={
                "task": "analyze_market_state",
                "params": {"pair": pair, "timeframe": timeframe, "features": features},
            },
        )

        logger.info(
            f"Created market state analysis task {task_id} for {pair}/{timeframe}"
        )
        return task_id

    async def create_market_state_dependency_chain(
        self,
        pairs: List[str],
        timeframes: List[str],
        features: Dict[str, Dict[str, Dict[str, Any]]],  # pair -> timeframe -> features
        dependent_task_generator: callable,
        priority: int = TaskPriority.MEDIUM,
    ) -> List[str]:
        """
        Create a chain of dependent tasks where subsequent tasks depend on market state analysis.

        Args:
            pairs: List of currency pairs to analyze
            timeframes: List of timeframes to analyze
            features: Dictionary of pre-calculated features
            dependent_task_generator: Function that creates dependent tasks based on market state
            priority: Priority for the market state analysis tasks

        Returns:
            List of all created task IDs
        """
        all_task_ids = []
        market_state_task_ids = []

        # First, create all market state analysis tasks
        for pair in pairs:
            for timeframe in timeframes:
                # Skip if we don't have features for this pair/timeframe
                if (
                    pair not in features
                    or timeframe not in features[pair]
                    or not features[pair][timeframe]
                ):
                    logger.warning(
                        f"No features available for {pair}/{timeframe}, skipping"
                    )
                    continue

                # Create the market state analysis task
                task_id = await self.create_market_state_analysis_task(
                    pair=pair,
                    timeframe=timeframe,
                    features=features[pair][timeframe],
                    priority=priority,
                )

                market_state_task_ids.append(task_id)
                all_task_ids.append(task_id)

        # Then create dependent tasks that depend on the market state tasks
        dependent_task_ids = await dependent_task_generator(market_state_task_ids)
        all_task_ids.extend(dependent_task_ids)

        return all_task_ids

    async def register_market_state_callback(self, task_id: str):
        """
        Register a callback to process market state analysis results.

        Args:
            task_id: Task ID of the market state analysis task
        """
        await self.orchestrator.add_task_callback(
            task_id, self._process_market_state_result
        )

    async def _process_market_state_result(
        self, task_id: str, result: Any, error: Optional[str]
    ):
        """
        Process the result of a market state analysis task.

        Args:
            task_id: Task ID of the completed task
            result: Task result (market state analysis)
            error: Error message if the task failed
        """
        if error:
            logger.error(f"Market state analysis task {task_id} failed: {error}")
            return

        if (
            not result
            or not isinstance(result, dict)
            or not result.get("success", False)
        ):
            logger.warning(f"Invalid result from market state analysis task {task_id}")
            return

        # Extract relevant information
        market_state = result.get("market_state", {})
        pair = result.get("market_state", {}).get("context", {}).get("pair")
        timeframe = result.get("market_state", {}).get("context", {}).get("timeframe")

        if not pair or not timeframe:
            logger.warning(
                f"Missing pair or timeframe in market state result: {result}"
            )
            return

        # Store in cache for quick access
        cache_key = f"{pair}_{timeframe}"
        self.recent_states[cache_key] = {
            "timestamp": asyncio.get_event_loop().time(),
            "market_state": market_state,
            "summary": result.get("market_state_summary", ""),
            "implications": result.get("trading_implications", {}),
        }

        logger.info(
            f"Processed market state for {pair}/{timeframe}: {result.get('market_state_summary', '')}"
        )

    async def get_recent_market_state(
        self, pair: str, timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a recently detected market state from the cache.

        Args:
            pair: Currency pair
            timeframe: Timeframe

        Returns:
            Recent market state or None if not available
        """
        cache_key = f"{pair}_{timeframe}"
        if cache_key in self.recent_states:
            # Check if it's still recent (within 5 minutes)
            cache_age = (
                asyncio.get_event_loop().time()
                - self.recent_states[cache_key]["timestamp"]
            )
            if cache_age < 300:  # 5 minutes
                return self.recent_states[cache_key]

        return None


# Function to create and initialize a MarketStateWorkflow
async def create_market_state_workflow(
    orchestrator: AdvancedOrchestrator,
) -> MarketStateWorkflow:
    """
    Create and initialize a MarketStateWorkflow.

    Args:
        orchestrator: AdvancedOrchestrator instance

    Returns:
        Initialized MarketStateWorkflow
    """
    workflow = MarketStateWorkflow(orchestrator)

    # Register the market state agent
    await workflow.register_market_state_agent()

    return workflow
