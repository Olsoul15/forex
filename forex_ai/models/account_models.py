from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class AccountType(str, Enum):
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class AccountSummary(BaseModel):
    id: str = ""
    balance: float = 0.0

class AccountMetrics(BaseModel):
    win_rate: float = 0.0
    profit_factor: float = 0.0

class Order(BaseModel):
    id: str = ""
    type: OrderType = OrderType.MARKET

class Position(BaseModel):
    id: str = ""
    instrument: str = ""

class Trade(BaseModel):
    id: str = ""
    instrument: str = ""

class Transaction(BaseModel):
    id: str = ""
    type: str = ""

class BalanceEntry(BaseModel):
    amount: float = 0.0
    timestamp: datetime = datetime.now()

class OrderCreate(BaseModel):
    pass

class OrderUpdate(BaseModel):
    pass

class PositionClose(BaseModel):
    pass

class AccountListResponse(BaseModel):
    success: bool = True
    accounts: List[Dict[str, Any]] = []
    count: int = 0
    timestamp: str = ""

class AccountDetailResponse(BaseModel):
    account: Optional[AccountSummary] = None
    metrics: Optional[AccountMetrics] = None

class OrderListResponse(BaseModel):
    orders: list = []
    count: int = 0

class OrderDetailResponse(BaseModel):
    order: Optional[Order] = None

class PositionListResponse(BaseModel):
    positions: list = []
    count: int = 0

class PositionDetailResponse(BaseModel):
    position: Optional[Position] = None

class TradeListResponse(BaseModel):
    trades: list = []
    count: int = 0

class TradeDetailResponse(BaseModel):
    trade: Optional[Trade] = None

class TransactionListResponse(BaseModel):
    transactions: list = []
    count: int = 0

class BalanceHistoryResponse(BaseModel):
    entries: list = []
    count: int = 0

class PerformanceReportResponse(BaseModel):
    report: dict = {}

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

class OrderDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED" 