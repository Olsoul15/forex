from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class TimeFrame(str, Enum):
    M1 = "M1"
    H1 = "H1"
    D1 = "D1"

class PriceType(str, Enum):
    BID = "bid"
    ASK = "ask"
    MID = "mid"

class InstrumentListRequest(BaseModel):
    pass

class PriceHistoryRequest(BaseModel):
    instrument: str
    timeframe: TimeFrame
    from_time: Optional[str] = None
    to_time: Optional[str] = None
    count: Optional[int] = None

class TechnicalIndicatorRequest(BaseModel):
    pass

class PatternDetectionRequest(BaseModel):
    pass

class MarketAnalysisRequest(BaseModel):
    pass

class PriceHistoryResponse(BaseModel):
    history: list = []
    timestamp: datetime = datetime.now()

class CurrentPriceResponse(BaseModel):
    prices: list = []
    timestamp: datetime = datetime.now()

class InstrumentListResponse(BaseModel):
    instruments: list = []
    count: int = 0
    timestamp: datetime = datetime.now()

class InstrumentDetailResponse(BaseModel):
    instrument: str = ""
    timestamp: datetime = datetime.now()
    trading_hours: Optional[dict] = None
    typical_spread: Optional[float] = None
    margin_requirement: Optional[float] = None
    related_instruments: Optional[list] = None

class TechnicalIndicatorResponse(BaseModel):
    pass

class PatternDetectionResponse(BaseModel):
    pass

class MarketAnalysisResponse(BaseModel):
    pass

class TechnicalLevelResponse(BaseModel):
    pass

class IndicatorType(str, Enum):
    SMA = "SMA"
    EMA = "EMA"

class ChartPatternType(str, Enum):
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"

class StreamingEventType(str, Enum):
    PRICE_UPDATE = "price_update"
    HEARTBEAT = "heartbeat"

class StreamingPriceUpdate(BaseModel):
    pass

class StreamingHeartbeat(BaseModel):
    pass

class StreamingSessionChange(BaseModel):
    pass

class InstrumentInfo(BaseModel):
    instrument: str = ""
    name: str = ""
    display_name: str = ""
    pip_location: int = -4
    trade_units_precision: int = 0
    margin_rate: float = 0.02
    max_leverage: float = 50.0
    bid: float = 0.0
    ask: float = 0.0
    base_currency: str = ""
    quote_currency: str = ""
    type: str = "FOREX"
    tradeable: bool = True
    trading_hours: Optional[dict] = None
    typical_spread: Optional[float] = None
    margin_requirement: Optional[float] = None
    related_instruments: Optional[list] = None
    
    @property
    def id(self) -> str:
        """Return instrument as id for test compatibility."""
        return self.instrument

class CandleData(BaseModel):
    time: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

class PriceHistory(BaseModel):
    candles: list = []

class CurrentPrice(BaseModel):
    instrument: str = ""
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    timestamp: datetime = datetime.now()
    id: str = ""  # Add this field for test compatibility

class TechnicalLevel(BaseModel):
    level: float = 0.0
    type: str = ""

class ChartPattern(BaseModel):
    pattern: str = ""
    confidence: float = 0.0

class TrendType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class VolatilityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class MarketAnalysis(BaseModel):
    summary: str = ""
    details: dict = {} 