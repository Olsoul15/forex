# Automation Module

This module provides workflow automation, scheduling, and task orchestration for the Forex AI Trading System.

## Overview

The automation module enables the creation, scheduling, and execution of automated workflows within the Forex AI Trading System. It provides infrastructure for task orchestration, dependency management, error handling, and monitoring of automated processes.

## Key Components

### Workflow Engine

- **workflow_engine.py**: Core workflow orchestration
  - Manages workflow definitions and execution
  - Handles workflow state and persistence
  - Provides workflow templating and reuse
  - Implements error handling and retries

### Task Scheduler

- **scheduler.py**: Task scheduling service
  - Manages time-based and event-based scheduling
  - Supports cron-like schedule definitions
  - Handles timezone awareness and DST
  - Provides schedule management APIs

### Task Runner

- **task_runner.py**: Task execution framework
  - Executes individual workflow tasks
  - Manages task dependencies and ordering
  - Handles task isolation and resource allocation
  - Implements task logging and monitoring

### Workflow Templates

- **workflows/**: Directory containing predefined workflow templates
  - Data ingestion workflows
  - Market monitoring workflows
  - Model training workflows
  - Trading strategy workflows
  - Reporting and analysis workflows

## Workflow Concepts

Workflows in the automation module follow these core concepts:

1. **Tasks**: Individual units of work with defined inputs and outputs
2. **Workflows**: Directed acyclic graphs (DAGs) of tasks with dependencies
3. **Triggers**: Events or schedules that initiate workflow execution
4. **State**: Workflow execution state and history
5. **Artifacts**: Data produced or consumed by workflow tasks

## Usage Examples

### Defining a Simple Workflow

```python
from forex_ai.automation.workflow_engine import Workflow, Task

# Define individual tasks
data_fetch_task = Task(
    name="fetch_market_data",
    handler="forex_ai.tasks.market_data.fetch_ohlc_data",
    params={"instrument": "EUR_USD", "timeframe": "H1", "count": 1000}
)

analysis_task = Task(
    name="analyze_market_data",
    handler="forex_ai.tasks.analysis.technical_analysis",
    params={"indicators": ["RSI", "MACD", "MA"]}
)

signal_task = Task(
    name="generate_signals",
    handler="forex_ai.tasks.signals.generate_trading_signals",
    params={"strategy": "trend_following"}
)

# Create a workflow with dependencies
workflow = Workflow(
    name="market_analysis_workflow",
    description="Analyzes market data and generates trading signals",
    tasks=[data_fetch_task, analysis_task, signal_task],
    dependencies={
        analysis_task.name: [data_fetch_task.name],
        signal_task.name: [analysis_task.name]
    }
)

# Save the workflow
from forex_ai.automation.workflow_engine import WorkflowManager
manager = WorkflowManager()
manager.save_workflow(workflow)
```

### Scheduling a Workflow

```python
from forex_ai.automation.scheduler import Scheduler
from datetime import timedelta

# Initialize the scheduler
scheduler = Scheduler()

# Schedule the workflow to run every hour
scheduler.schedule_workflow(
    workflow_name="market_analysis_workflow",
    schedule_type="interval",
    interval=timedelta(hours=1),
    start_time="2023-01-01T00:00:00Z"
)

# Schedule a workflow with cron expression (every weekday at 8:00 AM)
scheduler.schedule_workflow(
    workflow_name="daily_market_report",
    schedule_type="cron",
    cron_expression="0 8 * * 1-5",
    timezone="UTC"
)

# Start the scheduler
scheduler.start()
```

### Running a Workflow Manually

```python
from forex_ai.automation.workflow_engine import WorkflowManager

# Initialize the workflow manager
manager = WorkflowManager()

# Run a workflow with default parameters
execution_id = manager.run_workflow("market_analysis_workflow")

# Run a workflow with custom parameters
execution_id = manager.run_workflow(
    workflow_name="market_analysis_workflow",
    parameters={
        "fetch_market_data": {
            "instrument": "USD_JPY",
            "timeframe": "M15",
            "count": 500
        }
    }
)

# Check workflow execution status
status = manager.get_execution_status(execution_id)
print(f"Execution Status: {status.state}")
print(f"Start Time: {status.start_time}")
print(f"Duration: {status.duration}")
```

## Dependencies

- **Core Module**: For system infrastructure and data models
- **Config Module**: For workflow configuration settings
- **Services Module**: For API and integration support
- **Health Module**: For workflow monitoring integration
- **APScheduler**: For schedule management
- **asyncio**: For asynchronous task execution 