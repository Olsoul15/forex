import pytest
from unittest.mock import MagicMock, patch, ANY
from forex_ai.data.connectors.realtime_data import RealtimeDataConnector


@pytest.fixture
def mock_redis_password(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", "testpassword")


def test_realtime_data_connector_initialization(mock_redis_password):
    """Test that the RealtimeDataConnector initializes correctly."""
    connector = RealtimeDataConnector(api_key="test_api_key")
    assert connector.api_key == "test_api_key"
    assert isinstance(connector.active_connections, dict)
    assert len(connector.active_connections) == 0


def test_initialize_websocket(mock_redis_password):
    """Test that the initialize_websocket method correctly creates a WebSocket connection."""
    mock_websocket_module = MagicMock()
    mock_ws_app = MagicMock()
    mock_websocket_module.WebSocketApp.return_value = mock_ws_app

    with patch.dict('sys.modules', {'websocket': mock_websocket_module}):
        connector = RealtimeDataConnector(api_key="test_api_key")
        connection_info = connector.initialize_websocket(
            currency_pairs=["EUR/USD"],
            provider="default",
        )

        assert connection_info["status"] == "initializing"
        mock_websocket_module.WebSocketApp.assert_called_once_with(
            ANY,
            on_message=ANY,
            on_error=ANY,
            on_close=ANY,
            on_open=ANY
        )


def test_initialize_websocket_missing_client(mock_redis_password):
    """Test that initialize_websocket handles the case where websocket-client is not installed."""
    with patch.dict("sys.modules", {"websocket": None}):
        connector = RealtimeDataConnector(api_key="test_api_key")
        result = connector.initialize_websocket(currency_pairs=["EUR/USD"])
        assert result["status"] == "error"
        assert "Missing websocket-client package" in result["message"]


@patch("forex_ai.data.connectors.realtime_data.threading.Thread")
def test_start_and_stop_websocket(mock_thread, mock_redis_password):
    """Test that the start and stop methods work correctly."""
    mock_websocket_module = MagicMock()
    mock_ws_app = MagicMock()
    mock_websocket_module.WebSocketApp.return_value = mock_ws_app

    with patch.dict('sys.modules', {'websocket': mock_websocket_module}):
        connector = RealtimeDataConnector(api_key="test_api_key")
        connection_info = connector.initialize_websocket(currency_pairs=["EUR/USD"])

        # Start the websocket
        start_info = connector.start_websocket(connection_info)
        assert start_info["status"] == "running"
        mock_thread.assert_called_once_with(target=mock_ws_app.run_forever)
        mock_thread.return_value.start.assert_called_once()

        # Stop the websocket
        stop_info = connector.stop_websocket(start_info)
        assert stop_info["status"] == "closed"
        mock_ws_app.close.assert_called_once() 