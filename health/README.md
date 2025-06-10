# Health Module

This module provides health monitoring, diagnostics, and operational metrics for the Forex AI Trading System.

## Overview

The health module serves as the central monitoring infrastructure for the entire Forex AI Trading System. It collects, processes, and reports health metrics, enabling early detection of issues, performance optimization, and operational insights.

## Key Components

### Health Controller

- **health_controller.py**: Central health monitoring service
  - Manages system-wide health checks
  - Collects metrics from all subsystems
  - Provides alerting and notification
  - Implements health status API endpoints

### Metrics Collection

- **metrics_collector.py**: Core metrics collection infrastructure
  - Implements metric collection across services
  - Provides standardized metrics APIs
  - Supports various metric types (counters, gauges, histograms)
  - Handles metric aggregation and processing

### Diagnostics

- **diagnostic_tools.py**: System diagnostics utilities
  - Provides deeper diagnostic capabilities
  - Supports problem analysis and troubleshooting
  - Enables performance profiling
  - Implements self-healing mechanisms

### Alerting

- **alert_manager.py**: Alert management system
  - Defines alert rules and thresholds
  - Manages alert state and escalation
  - Supports multiple notification channels
  - Provides alert muting and suppression

## Monitored Areas

The health module monitors various aspects of the system:

1. **Service Health**: Status of individual services and components
2. **Performance Metrics**: Response times, throughput, and resource usage
3. **Error Rates**: Frequency and patterns of errors
4. **System Resources**: CPU, memory, disk, and network utilization
5. **Integration Health**: Status of external integrations and APIs
6. **Business Metrics**: Trading volumes, order execution, and P&L

## Usage Examples

### Basic Health Checking

```python
from forex_ai.health.health_controller import HealthController

# Get the health controller
health_controller = HealthController()

# Check overall system health
system_health = health_controller.check_system_health()
print(f"System Health: {system_health.status}")

# Get detailed component health
component_health = health_controller.check_component_health("execution_service")
print(f"Execution Service Status: {component_health.status}")
print(f"Execution Service Details: {component_health.details}")
```

### Registering Custom Health Checks

```python
from forex_ai.health.health_controller import HealthController
from forex_ai.health.models import HealthCheckResult, HealthStatus

# Define a custom health check function
def check_database_connection():
    try:
        # Perform actual check logic here
        connection_ok = True
        
        if connection_ok:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                details="Database connection established successfully"
            )
        else:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                details="Failed to connect to database"
            )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            details=f"Error checking database connection: {str(e)}"
        )

# Register the custom health check
health_controller = HealthController()
health_controller.register_health_check(
    "database_connection", 
    check_database_connection,
    category="infrastructure"
)
```

### Working with Metrics

```python
from forex_ai.health.metrics_collector import MetricsCollector

# Get the metrics collector
metrics = MetricsCollector()

# Record a simple counter metric
metrics.increment_counter("api.requests", tags={"endpoint": "/orders"})

# Record a gauge metric
metrics.record_gauge("system.memory.usage", 1024, tags={"unit": "MB"})

# Record a timing metric
with metrics.measure_time("order_execution_time"):
    # Code to execute an order
    pass

# Get recorded metrics
recent_metrics = metrics.get_recent_metrics(last_minutes=5)
for metric in recent_metrics:
    print(f"{metric.name}: {metric.value} {metric.tags}")
```

## Integration with Monitoring Tools

The health module supports integration with various monitoring tools and platforms:

- **Prometheus**: For metrics collection and alerting
- **Grafana**: For metrics visualization and dashboards
- **ELK Stack**: For log aggregation and analysis
- **CloudWatch**: For AWS-based deployments
- **DataDog**: For comprehensive monitoring

## Dependencies

- **Core Module**: For system infrastructure and data models
- **Config Module**: For monitoring configuration settings
- **Services Module**: For API and integration support 