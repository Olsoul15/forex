import json
from unittest.mock import patch

@patch('forex_ai.backend_api.server.OHLCService')
@patch('forex_ai.backend_api.server.IndicatorService')
def test_health_check(mock_indicator_service, mock_ohlc_service, test_client):
    """
    Test the /health endpoint to ensure it returns a successful response.
    """
    response = test_client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'
    assert 'timestamp' in data 