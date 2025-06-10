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
from langchain.chat_models import ChatOpenAI

# Forex AI dependencies
from forex_ai.custom_types import (
    Strategy,
    CandlestickStrategy,
    IndicatorStrategy,
    PineScriptStrategy,
)
from forex_ai.exceptions import StrategyError, ValidationError

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
        self.llm_provider = config.get("llm_provider", "openai")
        self.model_name = config.get("model_name", "gpt-4")

        # Initialize LLM
        self._initialize_llm()

        logger.info(
            f"LLM Strategy Integrator initialized with provider: {self.llm_provider}, model: {self.model_name}"
        )

    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        if self.llm_provider == "azure":
            # Use Azure OpenAI credentials from environment
            from langchain.chat_models import AzureChatOpenAI

            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                temperature=self.config.get("temperature", 0.2),
            )
            logger.info(
                f"Initialized Azure OpenAI with deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}"
            )
        elif self.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.config.get("temperature", 0.2),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logger.info(f"Initialized OpenAI with model: {self.model_name}")
        elif self.llm_provider == "anthropic":
            from langchain.chat_models import ChatAnthropic

            self.llm = ChatAnthropic(
                model=self.model_name,
                temperature=self.config.get("temperature", 0.2),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            logger.info(f"Initialized Anthropic with model: {self.model_name}")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    async def natural_language_to_strategy(
        self, description: str
    ) -> StrategyTranslationResult:
        """
        Convert a natural language strategy description to a structured strategy.

        Args:
            description: Natural language description of trading strategy

        Returns:
            StrategyTranslationResult object containing structured strategy information
        """
        try:
            template = """
            You are an expert in forex trading strategies and technical analysis.
            Convert the following natural language description of a trading strategy into a structured format.
            
            Natural language description:
            {description}
            
            Analyze the description and extract:
            1. Strategy type (candlestick pattern, indicator-based, etc.)
            2. Entry and exit conditions with precise parameters
            3. Risk management settings (if any)
            4. Suitable timeframes and currency pairs
            
            {format_instructions}
            """

            parser = PydanticOutputParser(pydantic_object=StrategyTranslationResult)
            prompt = PromptTemplate(
                template=template,
                input_variables=["description"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(description=description)

            parsed_result = parser.parse(result)

            logger.info(
                f"Successfully parsed natural language strategy: {parsed_result.strategy_type}"
            )
            return parsed_result

        except Exception as e:
            logger.error(f"Error parsing natural language strategy: {str(e)}")
            raise StrategyError(f"Failed to parse natural language strategy: {str(e)}")

    async def generate_strategy_code(
        self, strategy: Union[Dict[str, Any], BaseModel]
    ) -> str:
        """
        Generate Python code for a strategy based on its structured definition.

        Args:
            strategy: Structured strategy definition

        Returns:
            String containing Python code implementing the strategy
        """
        try:
            if isinstance(strategy, BaseModel):
                strategy_dict = strategy.dict()
            else:
                strategy_dict = strategy

            template = """
            You are a Python expert specializing in algorithmic trading implementations.
            Generate clean, efficient, and well-commented Python code for the following trading strategy:
            
            Strategy definition:
            {strategy}
            
            The code should:
            1. Follow best practices for Python (PEP 8)
            2. Include proper error handling
            3. Be compatible with the Forex AI system architecture
            4. Include docstrings and type hints
            5. Be optimized for performance
            
            Return only the Python code with no additional explanation.
            """

            prompt = PromptTemplate(template=template, input_variables=["strategy"])

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(strategy=json.dumps(strategy_dict, indent=2))

            logger.info(f"Successfully generated strategy code")
            return result.strip()

        except Exception as e:
            logger.error(f"Error generating strategy code: {str(e)}")
            raise StrategyError(f"Failed to generate strategy code: {str(e)}")

    async def translate_pinescript_to_python(self, pinescript_code: str) -> str:
        """
        Translate TradingView PineScript code to Python code.

        Args:
            pinescript_code: PineScript code to translate

        Returns:
            Equivalent Python code
        """
        try:
            template = """
            You are an expert in both TradingView PineScript and Python for algorithmic trading.
            Translate the following PineScript code to equivalent Python code that can run in the Forex AI system.
            
            PineScript code:
            ```
            {pinescript_code}
            ```
            
            Your translation should:
            1. Accurately preserve the strategy logic
            2. Use appropriate Python libraries (pandas, numpy, ta-lib)
            3. Follow Python best practices
            4. Include detailed comments explaining the translation
            5. Handle edge cases properly
            
            Return only the Python code with no additional explanation.
            """

            prompt = PromptTemplate(
                template=template, input_variables=["pinescript_code"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(pinescript_code=pinescript_code)

            logger.info(f"Successfully translated PineScript to Python")
            return result.strip()

        except Exception as e:
            logger.error(f"Error translating PineScript to Python: {str(e)}")
            raise StrategyError(f"Failed to translate PineScript to Python: {str(e)}")

    async def translate_python_to_pinescript(self, python_code: str) -> str:
        """
        Translate Python trading code to TradingView PineScript.

        Args:
            python_code: Python code to translate

        Returns:
            Equivalent PineScript code
        """
        try:
            template = """
            You are an expert in both Python algorithmic trading and TradingView PineScript.
            Translate the following Python trading code to equivalent PineScript code.
            
            Python code:
            ```
            {python_code}
            ```
            
            Your translation should:
            1. Accurately preserve the strategy logic
            2. Use appropriate PineScript functions and syntax
            3. Follow PineScript best practices
            4. Include detailed comments explaining the translation
            5. Be compatible with TradingView
            
            Return only the PineScript code with no additional explanation.
            """

            prompt = PromptTemplate(template=template, input_variables=["python_code"])

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(python_code=python_code)

            logger.info(f"Successfully translated Python to PineScript")
            return result.strip()

        except Exception as e:
            logger.error(f"Error translating Python to PineScript: {str(e)}")
            raise StrategyError(f"Failed to translate Python to PineScript: {str(e)}")

    async def optimize_strategy(
        self, strategy: Dict[str, Any], performance_data: Dict[str, Any]
    ) -> StrategyOptimizationSuggestion:
        """
        Analyze strategy performance and suggest optimizations.

        Args:
            strategy: Strategy configuration
            performance_data: Historical performance metrics

        Returns:
            StrategyOptimizationSuggestion with recommendations
        """
        try:
            template = """
            You are an expert forex trading strategy optimizer.
            Analyze the following strategy and its performance data to suggest improvements.
            
            Strategy configuration:
            {strategy}
            
            Performance data:
            {performance_data}
            
            Provide specific parameter changes that could improve performance.
            Consider factors like:
            1. Win rate and profit factor
            2. Drawdown management
            3. Entry and exit timing
            4. Risk-reward ratio
            5. Market condition adaptability
            
            {format_instructions}
            """

            parser = PydanticOutputParser(
                pydantic_object=StrategyOptimizationSuggestion
            )
            prompt = PromptTemplate(
                template=template,
                input_variables=["strategy", "performance_data"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = await chain.arun(
                strategy=json.dumps(strategy, indent=2),
                performance_data=json.dumps(performance_data, indent=2),
            )

            parsed_result = parser.parse(result)

            logger.info(f"Successfully generated optimization suggestions")
            return parsed_result

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
