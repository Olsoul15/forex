"""
LLM Strategy Integrator for Forex AI

This module integrates LLM capabilities into the strategy system of Forex AI trading platform.
It provides functionality for natural language strategy definition, code generation, optimization,
and other AI-powered features to enhance strategy creation and management.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field

# Core dependencies
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Google Vertex AI dependencies
from langchain_google_vertexai import VertexAI

# Forex AI dependencies
from forex_ai.custom_types import (
    Strategy,
    CandlestickStrategy,
    IndicatorStrategy,
    PineScriptStrategy,
)
from forex_ai.exceptions import StrategyError, ValidationError
from forex_ai.models.llm_controller import LLMController

logger = logging.getLogger(__name__)


class StrategyTranslationResult(BaseModel):
    """Schema for strategy translation results"""

    strategy_type: str = Field(
        description="Type of strategy (e.g., 'candlestick_pattern', 'indicator_based', 'pinescript')"
    )
    parameters: Dict[str, Any] = Field(description="Parameters for the strategy")
    entry_conditions: List[Dict[str, Any]] = Field(
        description="List of entry conditions"
    )
    exit_conditions: List[Dict[str, Any]] = Field(description="List of exit conditions")
    risk_management: Dict[str, Any] = Field(description="Risk management settings")
    timeframes: List[str] = Field(description="Timeframes to use")
    pairs: List[str] = Field(description="Currency pairs to apply strategy to")
    python_code: Optional[str] = Field(
        None, description="Generated Python code for the strategy"
    )
    pine_script: Optional[str] = Field(
        None, description="Generated PineScript code for the strategy"
    )


class StrategyOptimizationSuggestion(BaseModel):
    """Schema for strategy optimization suggestions"""

    parameter_changes: Dict[str, Any] = Field(description="Suggested parameter changes")
    reasoning: str = Field(description="Reasoning behind the suggestions")
    expected_improvement: str = Field(description="Expected improvement from changes")


class StrategyValidationResult(BaseModel):
    """Schema for strategy validation results"""

    is_valid: bool = Field(description="Whether the strategy is valid")
    logical_issues: List[Dict[str, str]] = Field(
        description="Logical issues found in the strategy"
    )
    suggestions: List[Dict[str, str]] = Field(description="Suggestions for improvement")


class LLMStrategyIntegrator:
    """
    Integrates LLM capabilities into Forex AI strategy system.

    This class provides methods for:
    1. Natural language strategy definition
    2. Code generation and translation
    3. Strategy optimization
    4. Intelligent validation
    5. Market context integration
    6. Documentation generation
    7. Collaborative features
    8. User experience enhancement
    9. Performance analysis
    10. Risk management
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM Strategy Integrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_provider = config.get("provider", "vertex")
        self.model_name = config.get("model_name", "gemini-1.5-pro")
        self.temperature = config.get("temperature", 0.2)

        # Initialize LLM
        self._initialize_llm()

        logger.info(
            f"LLM Strategy Integrator initialized with provider: {self.llm_provider}, model: {self.model_name}"
        )

    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        try:
            # Use Google Vertex AI
            self.llm = VertexAI(
                model_name=self.model_name,
                temperature=self.temperature,
                project=os.getenv("GCP_PROJECT_ID"),
                location=os.getenv("GCP_LOCATION", "us-central1"),
                max_output_tokens=4000,
            )
            logger.info(f"Initialized Google Vertex AI with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Google Vertex AI: {str(e)}")
            # Fallback to MCP agent
            try:
                llm_controller = LLMController()
                self.llm = llm_controller.get_client("mcp_agent")
                if not self.llm:
                    raise ValueError("MCP agent not available")
                logger.info("Initialized MCP agent as fallback")
            except Exception as mcp_error:
                logger.error(f"Failed to initialize MCP agent: {str(mcp_error)}")
                raise ValueError("Failed to initialize any LLM provider") from e

    async def natural_language_to_strategy(
        self, description: str
    ) -> StrategyTranslationResult:
        """
        Convert natural language description to a structured strategy.

        Args:
            description: Natural language description of the strategy

        Returns:
            StrategyTranslationResult: Structured strategy information
        """
        try:
            parser = PydanticOutputParser(pydantic_object=StrategyTranslationResult)
            format_instructions = parser.get_format_instructions()

            prompt = PromptTemplate(
                template="""
                Translate the following forex trading strategy description into a structured format.
                
                Description:
                {description}
                
                {format_instructions}
                """,
                input_variables=["description"],
                partial_variables={"format_instructions": format_instructions},
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(description=description)

            # Parse the result
            parsed_result = parser.parse(result)
            return parsed_result

        except Exception as e:
            logger.error(f"Error translating natural language to strategy: {str(e)}")
            raise StrategyError(
                f"Failed to translate natural language to strategy: {str(e)}"
            )

    async def generate_strategy_code(
        self, strategy: StrategyTranslationResult
    ) -> str:
        """
        Generate Python code for a strategy.

        Args:
            strategy: Strategy translation result

        Returns:
            str: Generated Python code
        """
        try:
            prompt = PromptTemplate(
                template="""
                Generate Python code for the following forex trading strategy:
                
                Strategy Type: {strategy_type}
                Parameters: {parameters}
                Entry Conditions: {entry_conditions}
                Exit Conditions: {exit_conditions}
                Risk Management: {risk_management}
                Timeframes: {timeframes}
                Currency Pairs: {pairs}
                
                The code should:
                1. Be compatible with the Forex AI trading system
                2. Use pandas and numpy for data manipulation
                3. Include proper error handling
                4. Be well-documented with comments
                5. Follow PEP 8 style guidelines
                
                Return only the Python code without any additional text.
                """,
                input_variables=[
                    "strategy_type",
                    "parameters",
                    "entry_conditions",
                    "exit_conditions",
                    "risk_management",
                    "timeframes",
                    "pairs",
                ],
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(
                strategy_type=strategy.strategy_type,
                parameters=json.dumps(strategy.parameters),
                entry_conditions=json.dumps(strategy.entry_conditions),
                exit_conditions=json.dumps(strategy.exit_conditions),
                risk_management=json.dumps(strategy.risk_management),
                timeframes=json.dumps(strategy.timeframes),
                pairs=json.dumps(strategy.pairs),
            )

            return result

        except Exception as e:
            logger.error(f"Error generating strategy code: {str(e)}")
            raise StrategyError(f"Failed to generate strategy code: {str(e)}")

    async def translate_pinescript_to_python(self, pinescript: str) -> str:
        """
        Translate TradingView PineScript to Python code.

        Args:
            pinescript: PineScript code

        Returns:
            str: Python code
        """
        try:
            prompt = PromptTemplate(
                template="""
                Translate the following TradingView PineScript code to Python code for the Forex AI trading system:
                
                ```pinescript
                {pinescript}
                ```
                
                The Python code should:
                1. Use pandas and numpy for data manipulation
                2. Be compatible with the Forex AI trading system
                3. Include proper error handling
                4. Be well-documented with comments
                5. Follow PEP 8 style guidelines
                
                Return only the Python code without any additional text.
                """,
                input_variables=["pinescript"],
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(pinescript=pinescript)

            return result

        except Exception as e:
            logger.error(f"Error translating PineScript to Python: {str(e)}")
            raise StrategyError(f"Failed to translate PineScript to Python: {str(e)}")

    async def translate_python_to_pinescript(self, python_code: str) -> str:
        """
        Translate Python code to TradingView PineScript.

        Args:
            python_code: Python code

        Returns:
            str: PineScript code
        """
        try:
            prompt = PromptTemplate(
                template="""
                Translate the following Python code to TradingView PineScript:
                
                ```python
                {python_code}
                ```
                
                The PineScript code should:
                1. Be compatible with TradingView
                2. Include proper error handling
                3. Be well-documented with comments
                4. Follow PineScript best practices
                
                Return only the PineScript code without any additional text.
                """,
                input_variables=["python_code"],
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(python_code=python_code)

            return result

        except Exception as e:
            logger.error(f"Error translating Python to PineScript: {str(e)}")
            raise StrategyError(f"Failed to translate Python to PineScript: {str(e)}")

    async def optimize_strategy(
        self, strategy: Dict[str, Any], performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize a strategy based on performance data.

        Args:
            strategy: Strategy definition
            performance_data: Performance metrics

        Returns:
            Dict[str, Any]: Optimized strategy
        """
        try:
            prompt = PromptTemplate(
                template="""
                Optimize the following forex trading strategy based on its performance data:
                
                Strategy:
                {strategy}
                
                Performance Data:
                {performance_data}
                
                Suggest specific improvements to the strategy parameters, entry/exit conditions,
                or risk management settings to improve its performance.
                
                Return the optimized strategy as a JSON object.
                """,
                input_variables=["strategy", "performance_data"],
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(
                strategy=json.dumps(strategy),
                performance_data=json.dumps(performance_data),
            )

            # Parse the result as JSON
            try:
                optimized_strategy = json.loads(result)
                return optimized_strategy
            except json.JSONDecodeError:
                logger.error("Failed to parse optimization result as JSON")
                raise StrategyError("Failed to parse optimization result as JSON")

        except Exception as e:
            logger.error(f"Error optimizing strategy: {str(e)}")
            raise StrategyError(f"Failed to optimize strategy: {str(e)}")

    async def validate_strategy(
        self, strategy: Dict[str, Any]
    ) -> StrategyValidationResult:
        """
        Perform intelligent validation on a strategy, checking for logical errors and contradictions.

        Args:
            strategy: Strategy configuration to validate

        Returns:
            StrategyValidationResult with validation results and suggestions
        """
        try:
            template = """
            You are an expert forex trading strategy validator.
            Analyze the following strategy for logical issues, contradictions, and potential improvements.
            
            Strategy configuration:
            {strategy}
            
            Check for:
            1. Contradictory entry and exit conditions
            2. Unrealistic parameter values
            3. Missing risk management settings
            4. Potential overfitting signals
            5. Strategy logic that might not work in all market conditions
            6. Computational efficiency issues
            
            {format_instructions}
            """

            parser = PydanticOutputParser(pydantic_object=StrategyValidationResult)
            prompt = PromptTemplate(
                template=template,
                input_variables=["strategy"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(strategy=json.dumps(strategy, indent=2))

            parsed_result = parser.parse(result)

            logger.info(
                f"Successfully validated strategy: valid={parsed_result.is_valid}"
            )
            return parsed_result

        except Exception as e:
            logger.error(f"Error validating strategy: {str(e)}")
            raise StrategyError(f"Failed to validate strategy: {str(e)}")

    async def generate_strategy_documentation(
        self, strategy: Dict[str, Any], include_examples: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive documentation for a strategy.

        Args:
            strategy: Strategy configuration
            include_examples: Whether to include example scenarios

        Returns:
            Dictionary containing documentation sections
        """
        try:
            template = """
            You are an expert forex trading documentation writer.
            Create comprehensive documentation for the following trading strategy:
            
            Strategy configuration:
            {strategy}
            
            Your documentation should include:
            1. Strategy overview and purpose
            2. Detailed explanation of entry and exit conditions
            3. Parameter descriptions with recommended ranges
            4. Risk management guidelines
            5. Suitable market conditions
            6. Known limitations
            {examples_request}
            
            Format the documentation as a JSON with sections as keys and markdown text as values.
            """

            examples_request = (
                "7. Example scenarios with visualizations described in markdown"
                if include_examples
                else ""
            )

            prompt = PromptTemplate(
                template=template,
                input_variables=["strategy"],
                partial_variables={"examples_request": examples_request},
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(strategy=json.dumps(strategy, indent=2))

            # Parse the JSON output
            documentation = json.loads(result)

            logger.info(f"Successfully generated strategy documentation")
            return documentation

        except Exception as e:
            logger.error(f"Error generating strategy documentation: {str(e)}")
            raise StrategyError(f"Failed to generate strategy documentation: {str(e)}")

    async def analyze_market_context(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze current market conditions and suggest appropriate strategy configurations.

        Args:
            market_data: Current market data including price, indicators, and news

        Returns:
            Dictionary with market analysis and strategy recommendations
        """
        try:
            template = """
            You are an expert forex market analyst.
            Analyze the following market data and suggest appropriate trading strategies.
            
            Market data:
            {market_data}
            
            Provide:
            1. Current market condition analysis (trend, volatility, key levels)
            2. Suitable strategy types for the current market
            3. Recommended parameters for those strategies
            4. Risk management suggestions based on current volatility
            5. Key news events to watch
            
            Format your analysis as a JSON with appropriate sections.
            """

            prompt = PromptTemplate(template=template, input_variables=["market_data"])

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(market_data=json.dumps(market_data, indent=2))

            # Parse the JSON output
            analysis = json.loads(result)

            logger.info(f"Successfully generated market context analysis")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing market context: {str(e)}")
            raise StrategyError(f"Failed to analyze market context: {str(e)}")

    async def explain_performance(
        self, strategy: Dict[str, Any], performance_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate natural language explanation of strategy performance.

        Args:
            strategy: Strategy configuration
            performance_data: Performance metrics

        Returns:
            Dictionary with performance explanations and insights
        """
        try:
            template = """
            You are an expert forex strategy performance analyst.
            Create a detailed explanation of the following strategy's performance.
            
            Strategy configuration:
            {strategy}
            
            Performance data:
            {performance_data}
            
            Your analysis should include:
            1. Summary of overall performance (profit, win rate, drawdown)
            2. Root causes of successful and unsuccessful trades
            3. How the strategy performs in different market conditions
            4. Performance compared to benchmark
            5. Actionable insights for improvement
            
            Format your analysis as a JSON with section keys and markdown text values.
            """

            prompt = PromptTemplate(
                template=template, input_variables=["strategy", "performance_data"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(
                strategy=json.dumps(strategy, indent=2),
                performance_data=json.dumps(performance_data, indent=2),
            )

            # Parse the JSON output
            analysis = json.loads(result)

            logger.info(f"Successfully generated performance explanation")
            return analysis

        except Exception as e:
            logger.error(f"Error explaining performance: {str(e)}")
            raise StrategyError(f"Failed to explain performance: {str(e)}")

    async def suggest_risk_parameters(
        self,
        strategy: Dict[str, Any],
        account_info: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Suggest appropriate risk management parameters for a strategy.

        Args:
            strategy: Strategy configuration
            account_info: Trading account information
            market_data: Current market data

        Returns:
            Dictionary with risk management recommendations
        """
        try:
            template = """
            You are an expert forex risk management specialist.
            Suggest appropriate risk parameters for the following strategy.
            
            Strategy configuration:
            {strategy}
            
            Account information:
            {account_info}
            
            Market data:
            {market_data}
            
            Provide recommendations for:
            1. Position sizing (percentage of account)
            2. Stop-loss placement (fixed, ATR-based, or structure-based)
            3. Take-profit targets
            4. Risk-reward ratio
            5. Maximum open positions
            6. Maximum daily/weekly drawdown limits
            
            Format your recommendations as a JSON with clear explanations.
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["strategy", "account_info", "market_data"],
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(
                strategy=json.dumps(strategy, indent=2),
                account_info=json.dumps(account_info, indent=2),
                market_data=json.dumps(market_data, indent=2),
            )

            # Parse the JSON output
            recommendations = json.loads(result)

            logger.info(f"Successfully generated risk management recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error suggesting risk parameters: {str(e)}")
            raise StrategyError(f"Failed to suggest risk parameters: {str(e)}")

    async def process_user_feedback(
        self, strategy: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """
        Process user feedback on a strategy and suggest improvements.

        Args:
            strategy: Strategy configuration
            feedback: User feedback in natural language

        Returns:
            Dictionary with suggested improvements based on feedback
        """
        try:
            template = """
            You are an expert forex strategy designer.
            A user has provided feedback on the following strategy:
            
            Strategy configuration:
            {strategy}
            
            User feedback:
            {feedback}
            
            Based on this feedback:
            1. Identify the key issues or requests mentioned
            2. Suggest specific changes to the strategy
            3. Explain the expected impact of these changes
            4. Provide any additional recommendations
            
            Format your response as a JSON with clear sections.
            """

            prompt = PromptTemplate(
                template=template, input_variables=["strategy", "feedback"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(
                strategy=json.dumps(strategy, indent=2), feedback=feedback
            )

            # Parse the JSON output
            suggestions = json.loads(result)

            logger.info(f"Successfully processed user feedback")
            return suggestions

        except Exception as e:
            logger.error(f"Error processing user feedback: {str(e)}")
            raise StrategyError(f"Failed to process user feedback: {str(e)}")
