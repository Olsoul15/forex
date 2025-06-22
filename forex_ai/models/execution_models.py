from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class ExecutionMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"

class PositionSizeType(str, Enum):
    FIXED = "fixed"
    RISK = "risk"
    RISK_PERCENT = "risk_percent"

class OrderTriggerType(str, Enum):
    PRICE = "price"
    TIME = "time"

class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"

class MarketOrderRequest(BaseModel):
    account_id: str = ""
    instrument: str = ""
    direction: str = ""
    position_sizing: dict = {}
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    time_in_force: Optional[TimeInForce] = None
    client_request_id: Optional[str] = None
    strategy_id: Optional[str] = None

class MarketOrderResponse(BaseModel):
    success: bool = True
    message: str = ""
    timestamp: datetime = datetime.now()

class LimitOrderRequest(BaseModel):
    pass

class LimitOrderResponse(BaseModel):
    pass

class StopOrderRequest(BaseModel):
    pass

class StopOrderResponse(BaseModel):
    pass

class OrderTriggerRequest(BaseModel):
    pass

class OrderTriggerResponse(BaseModel):
    pass

class PositionSizeRequest(BaseModel):
    account_id: str = ""
    instrument: str = ""
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    risk_amount: Optional[float] = None
    risk_percent: Optional[float] = None

class PositionSizeResponse(BaseModel):
    success: bool = True
    calculation: dict = {}
    timestamp: datetime = datetime.now()

class RiskAnalysisRequest(BaseModel):
    account_id: str = ""
    instrument: str = ""
    direction: str = ""
    units: int = 0
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0

class RiskAnalysisResponse(BaseModel):
    success: bool = True
    risk_analysis: dict = {}
    timestamp: datetime = datetime.now()

class ExecutionPreferencesUpdate(BaseModel):
    preferences: dict = {}

class ExecutionPreferencesResponse(BaseModel):
    account_id: str = ""
    preferences: dict = {}
    timestamp: datetime = datetime.now()

class OrderCancelRequest(BaseModel):
    pass

class OrderCancelResponse(BaseModel):
    pass

class PositionCloseRequest(BaseModel):
    pass

class PositionCloseResponse(BaseModel):
    pass

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class PositionSizing(BaseModel):
    type: str = ""
    value: float = 0.0

class ExecutionResult(BaseModel):
    success: bool = True
    message: str = ""

class RiskAnalysis(BaseModel):
    risk: float = 0.0
    details: dict = {}

class PositionSizeCalculation(BaseModel):
    size: float = 0.0
    details: dict = {} 