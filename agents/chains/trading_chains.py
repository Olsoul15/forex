"""
Trading chains for the Forex AI Trading System.
"""

from typing import Dict, List, Any
from datetime import datetime

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field

from forex_ai.models.llm_controller import get_llm_controller
from forex_ai.agents.tools.market_tools import (
    TechnicalAnalysisTool,
    SentimentAnalysisTool,
    MarketImpactTool,
)


class TradingDecision(BaseModel):
    """Output schema for trading decisions."""

    action: str = Field(..., description="Trading action (buy, sell, hold)")
    symbol: str = Field(..., description="Trading pair symbol")
    entry_price: float = Field(..., description="Suggested entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    take_profit: float = Field(..., description="Take profit price")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    reasoning: str = Field(..., description="Reasoning behind the decision")


class TradingDecisionParser(PydanticOutputParser):
    """Parser for trading decisions."""

    def __init__(self):
        super().__init__(pydantic_object=TradingDecision)


class MarketAnalysisChain:
    """Chain for comprehensive market analysis."""

    def __init__(self):
        self.llm_controller = get_llm_controller()
        self.llm = self.llm_controller.get_model("gpt4")  # Using GPT-4 for analysis

        # Initialize tools
        self.technical_tool = TechnicalAnalysisTool()
        self.sentiment_tool = SentimentAnalysisTool()
        self.impact_tool = MarketImpactTool()

        # Initialize output parser
        self.output_parser = TradingDecisionParser()

        # Create the analysis chain
        self._create_chain()

    def _create_chain(self):
        """Create the sequential analysis chain."""
        # Technical Analysis Chain
        technical_prompt = PromptTemplate(
            input_variables=["symbol", "timeframe"],
            template="""
            Analyze the technical indicators for {symbol} on {timeframe} timeframe.
            Consider trends, support/resistance levels, and key patterns.
            
            {format_instructions}
            """,
        )

        technical_chain = LLMChain(
            llm=self.llm, prompt=technical_prompt, output_parser=self.output_parser
        )

        # Sentiment Analysis Chain
        sentiment_prompt = PromptTemplate(
            input_variables=["symbol", "timeframe", "technical_analysis"],
            template="""
            Given the technical analysis:
            {technical_analysis}
            
            Analyze market sentiment for {symbol} on {timeframe} timeframe.
            Consider news sentiment and social media indicators.
            
            {format_instructions}
            """,
        )

        sentiment_chain = LLMChain(
            llm=self.llm, prompt=sentiment_prompt, output_parser=self.output_parser
        )

        # Impact Analysis Chain
        impact_prompt = PromptTemplate(
            input_variables=[
                "symbol",
                "timeframe",
                "technical_analysis",
                "sentiment_analysis",
            ],
            template="""
            Given the technical analysis:
            {technical_analysis}
            
            And sentiment analysis:
            {sentiment_analysis}
            
            Predict potential market impact for {symbol} on {timeframe} timeframe.
            Consider event impact probabilities and price movement ranges.
            
            {format_instructions}
            """,
        )

        impact_chain = LLMChain(
            llm=self.llm, prompt=impact_prompt, output_parser=self.output_parser
        )

        # Final Decision Chain
        decision_prompt = PromptTemplate(
            input_variables=[
                "symbol",
                "timeframe",
                "technical_analysis",
                "sentiment_analysis",
                "impact_analysis",
            ],
            template="""
            Based on the following analyses for {symbol} on {timeframe} timeframe:
            
            Technical Analysis:
            {technical_analysis}
            
            Sentiment Analysis:
            {sentiment_analysis}
            
            Impact Analysis:
            {impact_analysis}
            
            Make a trading decision that includes:
            1. Action (buy, sell, or hold)
            2. Entry price
            3. Stop loss level
            4. Take profit level
            5. Confidence score
            6. Detailed reasoning
            
            {format_instructions}
            """,
        )

        decision_chain = LLMChain(
            llm=self.llm, prompt=decision_prompt, output_parser=self.output_parser
        )

        # Combine into sequential chain
        self.chain = SequentialChain(
            chains=[technical_chain, sentiment_chain, impact_chain, decision_chain],
            input_variables=["symbol", "timeframe"],
            output_variables=["trading_decision"],
        )

    async def analyze(self, symbol: str, timeframe: str) -> TradingDecision:
        """
        Perform comprehensive market analysis and make trading decision.

        Args:
            symbol: Trading pair symbol
            timeframe: Analysis timeframe

        Returns:
            Trading decision with analysis
        """
        try:
            # Run technical analysis
            technical_result = await self.technical_tool._arun(
                symbol=symbol, timeframe=timeframe
            )

            # Run sentiment analysis
            sentiment_result = await self.sentiment_tool._arun(
                symbol=symbol, timeframe=timeframe
            )

            # Run impact analysis
            impact_result = await self.impact_tool._arun(
                symbol=symbol, timeframe=timeframe
            )

            # Make final decision
            result = await self.chain.arun(
                symbol=symbol,
                timeframe=timeframe,
                technical_analysis=technical_result,
                sentiment_analysis=sentiment_result,
                impact_analysis=impact_result,
            )

            return self.output_parser.parse(result)

        except Exception as e:
            raise Exception(f"Error in market analysis chain: {str(e)}")


class RiskManagementChain:
    """Chain for risk assessment and position sizing."""

    def __init__(self):
        self.llm_controller = get_llm_controller()
        self.llm = self.llm_controller.get_model("gpt4")

        # Create the risk management chain
        self._create_chain()

    def _create_chain(self):
        """Create the risk management chain."""
        risk_prompt = PromptTemplate(
            input_variables=["trading_decision", "account_balance", "open_positions"],
            template="""
            Given the trading decision:
            {trading_decision}
            
            Account balance: {account_balance}
            Current open positions: {open_positions}
            
            Assess risk and determine:
            1. Position size
            2. Risk per trade
            3. Portfolio exposure
            4. Risk/reward ratio
            5. Maximum drawdown impact
            
            {format_instructions}
            """,
        )

        self.chain = LLMChain(llm=self.llm, prompt=risk_prompt)

    async def assess_risk(
        self,
        trading_decision: TradingDecision,
        account_balance: float,
        open_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assess risk and determine position sizing.

        Args:
            trading_decision: Trading decision to assess
            account_balance: Current account balance
            open_positions: List of open positions

        Returns:
            Risk assessment and position sizing details
        """
        try:
            result = await self.chain.arun(
                trading_decision=trading_decision.dict(),
                account_balance=account_balance,
                open_positions=open_positions,
            )

            return result

        except Exception as e:
            raise Exception(f"Error in risk management chain: {str(e)}")


# Export chain factories
def create_market_analysis_chain() -> MarketAnalysisChain:
    """Create a new market analysis chain."""
    return MarketAnalysisChain()


def create_risk_management_chain() -> RiskManagementChain:
    """Create a new risk management chain."""
    return RiskManagementChain()
