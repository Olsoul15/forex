"""
Strategy API Endpoints for Forex AI

This module contains FastAPI endpoints for strategy management with LLM-enhanced capabilities.
"""

from typing import Dict, List, Any, Optional
import logging
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    File,
    UploadFile,
    Form,
    Query,
    status,
)
from fastapi.responses import JSONResponse

from forex_ai.data.storage.supabase_strategy_repository import (
    SupabaseStrategyRepository,
)
from forex_ai.custom_types import (
    Strategy,
    CandlestickStrategy,
    IndicatorStrategy,
    PineScriptStrategy,
)
from forex_ai.exceptions import StrategyNotFoundError, StrategyRepositoryError
from forex_ai.agents.framework.llm_strategy_integrator import LLMStrategyIntegrator
from forex_ai.api.auth import get_current_user

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/strategy", tags=["strategy"])

# Initialize LLM strategy integrator (will be properly initialized when app starts)
llm_strategy_integrator = None
# Initialize the strategy repository
strategy_repository = None


def init_llm_strategy_integrator(config: Dict[str, Any]):
    """Initialize the LLM strategy integrator with config"""
    global llm_strategy_integrator
    llm_strategy_integrator = LLMStrategyIntegrator(config)
    logger.info("LLM strategy integrator initialized")


def init_strategy_repository():
    """Initialize the strategy repository"""
    global strategy_repository
    from forex_ai.data.storage.supabase_strategy_repository import (
        SupabaseStrategyRepository,
    )

    try:
        strategy_repository = SupabaseStrategyRepository()
        logger.info("Strategy repository initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Supabase strategy repository: {str(e)}")
        logger.warning("Using mock strategy repository instead")
        
        # Create a simple mock repository with basic functionality
        class MockStrategyRepository:
            async def save_strategy(self, strategy_dict):
                logger.info(f"Mock: Saving strategy {strategy_dict.get('name', 'unnamed')}")
                return {"id": "mock-strategy-id", **strategy_dict}
                
            async def get_strategy(self, strategy_id):
                logger.info(f"Mock: Getting strategy {strategy_id}")
                return {"id": strategy_id, "name": "Mock Strategy"}
                
            async def list_strategies(self):
                logger.info("Mock: Listing strategies")
                return []
                
            async def delete_strategy(self, strategy_id):
                logger.info(f"Mock: Deleting strategy {strategy_id}")
                return True
                
            async def update_strategy(self, strategy_id, strategy_dict):
                logger.info(f"Mock: Updating strategy {strategy_id}")
                return {"id": strategy_id, **strategy_dict}
        
        strategy_repository = MockStrategyRepository()
        logger.info("Mock strategy repository initialized")
    
    return strategy_repository


# --- Natural Language Strategy Endpoints --- #


@router.post("/natural-language", response_model=Dict[str, Any])
async def create_strategy_from_natural_language(
    description: str = Form(...),
    name: str = Form(...),
    auto_generate_code: bool = Form(False),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Create a new strategy from a natural language description.

    The LLM will parse the description and extract strategy parameters,
    entry/exit conditions, and other relevant settings.
    """
    try:
        logger.info(f"Creating strategy from natural language: {name}")

        # Translate natural language to strategy
        strategy_result = await llm_strategy_integrator.natural_language_to_strategy(
            description
        )

        # Generate code if requested
        if auto_generate_code:
            python_code = await llm_strategy_integrator.generate_strategy_code(
                strategy_result
            )
            strategy_result.python_code = python_code

        # Prepare the strategy for saving
        strategy_dict = strategy_result.dict()
        strategy_dict["name"] = name
        strategy_dict["original_description"] = description
        strategy_dict["created_by"] = (
            current_user["id"] if "id" in current_user else None
        )

        # Save to database
        saved_strategy = await strategy_repository.save_strategy(strategy_dict)

        # Record the LLM interaction
        await _record_llm_interaction(
            strategy_id=saved_strategy["id"],
            user_id=current_user["id"] if "id" in current_user else None,
            interaction_type="creation",
            input_text=description,
            output_json=strategy_dict,
        )

        return {
            "success": True,
            "message": "Strategy created successfully",
            "strategy": saved_strategy,
        }
    except Exception as e:
        logger.error(f"Error creating strategy from natural language: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create strategy: {str(e)}",
        )


# --- Code Generation and Translation Endpoints --- #


@router.post("/pinescript-to-python", response_model=Dict[str, Any])
async def translate_pinescript_to_python(
    pinescript: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Translate TradingView PineScript code to Python code.
    """
    try:
        logger.info("Translating PineScript to Python")

        python_code = await llm_strategy_integrator.translate_pinescript_to_python(
            pinescript
        )

        return {"success": True, "python_code": python_code}
    except Exception as e:
        logger.error(f"Error translating PineScript to Python: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}",
        )


@router.post("/python-to-pinescript", response_model=Dict[str, Any])
async def translate_python_to_pinescript(
    python_code: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Translate Python code to TradingView PineScript.
    """
    try:
        logger.info("Translating Python to PineScript")

        pinescript = await llm_strategy_integrator.translate_python_to_pinescript(
            python_code
        )

        return {"success": True, "pinescript": pinescript}
    except Exception as e:
        logger.error(f"Error translating Python to PineScript: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}",
        )


# --- Strategy Optimization Endpoints --- #


@router.post("/{strategy_id}/optimize", response_model=Dict[str, Any])
async def optimize_strategy(
    strategy_id: str,
    performance_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Analyze strategy performance and suggest optimizations.
    """
    try:
        logger.info(f"Optimizing strategy: {strategy_id}")

        # TODO: Get strategy from database via strategy_component
        strategy = {"id": strategy_id}  # This is a placeholder

        optimization = await llm_strategy_integrator.optimize_strategy(
            strategy, performance_data
        )

        return {"success": True, "optimization": optimization.dict()}
    except Exception as e:
        logger.error(f"Error optimizing strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}",
        )


# --- Validation Endpoints --- #


@router.post("/validate", response_model=Dict[str, Any])
async def validate_strategy(
    strategy: Dict[str, Any], current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Perform intelligent validation on a strategy, checking for logical errors and contradictions.
    """
    try:
        logger.info("Validating strategy")

        validation_result = await llm_strategy_integrator.validate_strategy(strategy)

        return {"success": True, "validation": validation_result.dict()}
    except Exception as e:
        logger.error(f"Error validating strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


# --- Documentation Endpoints --- #


@router.get("/{strategy_id}/documentation", response_model=Dict[str, Any])
async def get_strategy_documentation(
    strategy_id: str,
    include_examples: bool = Query(True),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Generate comprehensive documentation for a strategy.
    """
    try:
        logger.info(f"Generating documentation for strategy: {strategy_id}")

        # TODO: Get strategy from database via strategy_component
        strategy = {"id": strategy_id}  # This is a placeholder

        documentation = await llm_strategy_integrator.generate_strategy_documentation(
            strategy, include_examples
        )

        return {"success": True, "documentation": documentation}
    except Exception as e:
        logger.error(f"Error generating documentation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Documentation generation failed: {str(e)}",
        )


# --- Market Context Integration Endpoints --- #


@router.post("/market-analysis", response_model=Dict[str, Any])
async def analyze_market_context(
    market_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Analyze current market conditions and suggest appropriate strategy configurations.
    """
    try:
        logger.info("Analyzing market context")

        analysis = await llm_strategy_integrator.analyze_market_context(market_data)

        return {"success": True, "analysis": analysis}
    except Exception as e:
        logger.error(f"Error analyzing market context: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Market analysis failed: {str(e)}",
        )


# --- Performance Analysis Endpoints --- #


@router.post("/{strategy_id}/explain-performance", response_model=Dict[str, Any])
async def explain_strategy_performance(
    strategy_id: str,
    performance_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Generate natural language explanation of strategy performance.
    """
    try:
        logger.info(f"Explaining performance for strategy: {strategy_id}")

        # TODO: Get strategy from database via strategy_component
        strategy = {"id": strategy_id}  # This is a placeholder

        explanation = await llm_strategy_integrator.explain_performance(
            strategy, performance_data
        )

        return {"success": True, "explanation": explanation}
    except Exception as e:
        logger.error(f"Error explaining performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance explanation failed: {str(e)}",
        )


# --- Risk Management Endpoints --- #


@router.post("/{strategy_id}/risk-parameters", response_model=Dict[str, Any])
async def suggest_risk_parameters(
    strategy_id: str,
    account_info: Dict[str, Any],
    market_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Suggest appropriate risk management parameters for a strategy.
    """
    try:
        logger.info(f"Suggesting risk parameters for strategy: {strategy_id}")

        # TODO: Get strategy from database via strategy_component
        strategy = {"id": strategy_id}  # This is a placeholder

        recommendations = await llm_strategy_integrator.suggest_risk_parameters(
            strategy, account_info, market_data
        )

        return {"success": True, "risk_recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error suggesting risk parameters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk parameter suggestion failed: {str(e)}",
        )


# --- Collaborative Features Endpoints --- #


@router.post("/{strategy_id}/feedback", response_model=Dict[str, Any])
async def process_strategy_feedback(
    strategy_id: str,
    feedback: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Process user feedback on a strategy and suggest improvements.
    """
    try:
        logger.info(f"Processing feedback for strategy: {strategy_id}")

        # TODO: Get strategy from database via strategy_component
        strategy = {"id": strategy_id}  # This is a placeholder

        suggestions = await llm_strategy_integrator.process_user_feedback(
            strategy, feedback
        )

        return {"success": True, "suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback processing failed: {str(e)}",
        )


# Helper function to record LLM interactions
async def _record_llm_interaction(
    strategy_id: str,
    user_id: Optional[str],
    interaction_type: str,
    input_text: str,
    output_json: Dict[str, Any],
    model_used: Optional[str] = None,
    execution_time_ms: Optional[int] = None,
):
    """Record an LLM interaction in the database"""
    try:
        if not strategy_repository:
            logger.warning(
                "Strategy repository not initialized, skipping interaction recording"
            )
            return

        # Create the interaction record
        interaction = {
            "strategy_id": strategy_id,
            "user_id": user_id,
            "interaction_type": interaction_type,
            "input_text": input_text,
            "output_json": output_json,
            "model_used": model_used or llm_strategy_integrator.model_name,
            "execution_time_ms": execution_time_ms,
        }

        # Save to database using Supabase directly
        response = (
            strategy_repository.supabase.table("llm_strategy_interactions")
            .insert(interaction)
            .execute()
        )

        if hasattr(response, "error") and response.error:
            logger.error(f"Error recording LLM interaction: {response.error}")

    except Exception as e:
        # Just log the error, don't fail the main operation
        logger.error(f"Error recording LLM interaction: {str(e)}")
