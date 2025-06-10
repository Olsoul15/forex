"""
Trading workflow using LangGraph for the Forex AI Trading System.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from langchain.graphs import StateGraph
from langchain.graphs.state_graph import END
from pydantic import BaseModel, Field

from forex_ai.agents.chains.trading_chains import (
    create_market_analysis_chain,
    create_risk_management_chain,
    TradingDecision,
)


class WorkflowState(BaseModel):
    """State for the trading workflow."""

    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Analysis timeframe")
    account_balance: float = Field(..., description="Current account balance")
    open_positions: List[Dict[str, Any]] = Field(default_factory=list)
    trading_decision: Optional[TradingDecision] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    execution_status: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


class TradingWorkflow:
    """
    Trading workflow using LangGraph.

    This workflow coordinates:
    1. Market Analysis
    2. Risk Management
    3. Trade Execution
    4. Position Monitoring
    """

    def __init__(self):
        """Initialize the trading workflow."""
        # Create the state graph
        self.graph = StateGraph(WorkflowState)

        # Initialize chains
        self.market_analysis = create_market_analysis_chain()
        self.risk_management = create_risk_management_chain()

        # Build the workflow
        self._build_workflow()

    def _build_workflow(self):
        """Build the trading workflow graph."""
        # Add nodes
        self.graph.add_node("market_analysis", self._run_market_analysis)
        self.graph.add_node("risk_assessment", self._run_risk_assessment)
        self.graph.add_node("trade_execution", self._run_trade_execution)
        self.graph.add_node("position_monitoring", self._run_position_monitoring)

        # Add edges with conditions
        self.graph.add_edge(
            "market_analysis", "risk_assessment", self._should_assess_risk
        )
        self.graph.add_edge(
            "risk_assessment", "trade_execution", self._should_execute_trade
        )
        self.graph.add_edge(
            "trade_execution", "position_monitoring", self._should_monitor_position
        )
        self.graph.add_edge("position_monitoring", END, self._should_end_workflow)

        # Add error handling edges
        self.graph.add_edge("market_analysis", END, self._handle_analysis_error)
        self.graph.add_edge("risk_assessment", END, self._handle_risk_error)
        self.graph.add_edge("trade_execution", END, self._handle_execution_error)

    async def _run_market_analysis(self, state: WorkflowState) -> WorkflowState:
        """Run market analysis node."""
        try:
            trading_decision = await self.market_analysis.analyze(
                symbol=state.symbol, timeframe=state.timeframe
            )
            state.trading_decision = trading_decision
            return state
        except Exception as e:
            state.errors.append(f"Market analysis error: {str(e)}")
            return state

    async def _run_risk_assessment(self, state: WorkflowState) -> WorkflowState:
        """Run risk assessment node."""
        try:
            if state.trading_decision:
                risk_assessment = await self.risk_management.assess_risk(
                    trading_decision=state.trading_decision,
                    account_balance=state.account_balance,
                    open_positions=state.open_positions,
                )
                state.risk_assessment = risk_assessment
            return state
        except Exception as e:
            state.errors.append(f"Risk assessment error: {str(e)}")
            return state

    async def _run_trade_execution(self, state: WorkflowState) -> WorkflowState:
        """Run trade execution node."""
        try:
            if state.trading_decision and state.risk_assessment:
                # Execute trade logic would go here
                state.execution_status = {
                    "status": "simulated",
                    "timestamp": datetime.now().isoformat(),
                    "details": {
                        "action": state.trading_decision.action,
                        "symbol": state.trading_decision.symbol,
                        "entry_price": state.trading_decision.entry_price,
                    },
                }
            return state
        except Exception as e:
            state.errors.append(f"Trade execution error: {str(e)}")
            return state

    async def _run_position_monitoring(self, state: WorkflowState) -> WorkflowState:
        """Run position monitoring node."""
        try:
            if state.execution_status:
                # Position monitoring logic would go here
                pass
            return state
        except Exception as e:
            state.errors.append(f"Position monitoring error: {str(e)}")
            return state

    def _should_assess_risk(self, state: WorkflowState) -> bool:
        """Determine if we should proceed to risk assessment."""
        return (
            state.trading_decision is not None
            and state.trading_decision.action != "hold"
            and not state.errors
        )

    def _should_execute_trade(self, state: WorkflowState) -> bool:
        """Determine if we should proceed to trade execution."""
        return state.risk_assessment is not None and not state.errors

    def _should_monitor_position(self, state: WorkflowState) -> bool:
        """Determine if we should proceed to position monitoring."""
        return (
            state.execution_status is not None
            and state.execution_status["status"] == "executed"
            and not state.errors
        )

    def _should_end_workflow(self, state: WorkflowState) -> bool:
        """Determine if we should end the workflow."""
        return True  # Always end after monitoring

    def _handle_analysis_error(self, state: WorkflowState) -> bool:
        """Handle market analysis errors."""
        return bool(state.errors)

    def _handle_risk_error(self, state: WorkflowState) -> bool:
        """Handle risk assessment errors."""
        return bool(state.errors)

    def _handle_execution_error(self, state: WorkflowState) -> bool:
        """Handle trade execution errors."""
        return bool(state.errors)

    async def run(
        self,
        symbol: str,
        timeframe: str,
        account_balance: float,
        open_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> WorkflowState:
        """
        Run the trading workflow.

        Args:
            symbol: Trading pair symbol
            timeframe: Analysis timeframe
            account_balance: Current account balance
            open_positions: List of current open positions

        Returns:
            Final workflow state
        """
        # Initialize workflow state
        initial_state = WorkflowState(
            symbol=symbol,
            timeframe=timeframe,
            account_balance=account_balance,
            open_positions=open_positions or [],
        )

        # Run the workflow
        try:
            final_state = await self.graph.arun(initial_state)
            return final_state
        except Exception as e:
            initial_state.errors.append(f"Workflow error: {str(e)}")
            return initial_state


# Export workflow factory
def create_trading_workflow() -> TradingWorkflow:
    """Create a new trading workflow."""
    return TradingWorkflow()
