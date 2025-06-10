# N8N Guide for Forex AI

## Introduction to N8N

N8N is a workflow automation tool that enables you to connect different services and automate workflows without coding. The Forex AI Trading System uses N8N for automating various tasks, such as data collection, alerts, and reporting.

## N8N in Forex AI

In the Forex AI system, N8N is used for:

1. **Data Collection** - Scheduled fetching of market data from external sources
2. **Alert System** - Sending notifications based on market conditions or trading signals
3. **Reporting** - Generating and distributing performance reports
4. **YouTube Processing** - Extracting and analyzing financial content from YouTube

## Accessing N8N

When running the Forex AI system with Docker Compose, N8N is available at:

```
http://localhost:5678
```

Default credentials:
- Username: `admin@example.com`
- Password: `password`

(These should be changed in a production environment)

## Setting up N8N for Forex AI

### 1. Set Environment Variables

Edit the `.env` file to configure N8N:

```
N8N_HOST=localhost
N8N_PORT=5678
N8N_ENCRYPTION_KEY=your_encryption_key
```

### 2. Start N8N with Docker Compose

```bash
docker-compose up -d n8n
```

### 3. Access the N8N Dashboard

Open your browser and navigate to `http://localhost:5678`

## Key Concepts in N8N

### Workflows

A workflow in N8N is a collection of nodes that process data. Each workflow typically consists of:

- **Trigger Node** - Starts the workflow (e.g., a schedule)
- **Processing Nodes** - Transform and process data
- **Action Nodes** - Perform actions based on processed data

### Nodes

Nodes are the building blocks of workflows. N8N offers hundreds of pre-built nodes for various services and functions. Some nodes commonly used in Forex AI include:

- **HTTP Request** - For API calls to external services
- **Postgres** - For database operations
- **Function** - For custom JavaScript functions
- **Webhook** - For receiving external triggers
- **Schedule** - For time-based triggers
- **Email** - For sending notifications

### Credentials

Credentials in N8N store authentication information for various services. They are encrypted and securely stored.

## Sample Workflows

### 1. Market Data Collection Workflow

This workflow fetches market data from AlphaVantage at regular intervals.

```
[Schedule Node] --> [HTTP Request Node] --> [Function Node] --> [Postgres Node]
```

- **Schedule Node**: Runs every hour
- **HTTP Request Node**: Calls AlphaVantage API to fetch forex data
- **Function Node**: Formats the data for storage
- **Postgres Node**: Saves data to the database

### 2. Trading Alert Workflow

This workflow monitors trading signals and sends alerts.

```
[Webhook Node] --> [Function Node] --> [Split Node] --> [IF Node] --> [Email Node]
                                                     --> [Telegram Node]
```

- **Webhook Node**: Receives trading signals from the Forex AI system
- **Function Node**: Processes and formats the signal data
- **Split Node**: Duplicates the data for different notification channels
- **IF Node**: Determines if alert criteria are met
- **Email/Telegram Nodes**: Send notifications through different channels

### 3. Performance Reporting Workflow

This workflow generates and sends performance reports.

```
[Schedule Node] --> [Postgres Node] --> [Function Node] --> [PDF Node] --> [Email Node]
```

- **Schedule Node**: Runs weekly on Sunday
- **Postgres Node**: Fetches performance data from the database
- **Function Node**: Generates report content
- **PDF Node**: Creates a PDF report
- **Email Node**: Sends the report to specified recipients

## Creating Custom Workflows

### Basic Steps to Create a Workflow

1. Go to the N8N dashboard and click "Create new workflow"
2. Add a trigger node (e.g., Schedule, Webhook)
3. Add processing nodes to handle your data
4. Add action nodes to perform the desired actions
5. Connect the nodes by dragging from the output of one node to the input of another
6. Configure each node with the necessary settings
7. Save and activate the workflow

### Example: Creating a Simple Alert Workflow

1. **Add a Webhook Node**: 
   - Set to `POST` method
   - Enable "Respond to Webhook"
   - Copy the webhook URL for later use

2. **Add a Function Node**:
   ```javascript
   // Process the webhook data
   const data = items[0].json;
   const pair = data.pair;
   const price = data.price;
   const signal = data.signal;
   
   return {
     json: {
       subject: `Forex Alert: ${signal} signal for ${pair}`,
       body: `A ${signal} signal has been detected for ${pair} at price ${price}.`,
       signal: signal,
       pair: pair,
       price: price
     }
   };
   ```

3. **Add an IF Node**:
   - Add a condition: `data.signal === "buy" || data.signal === "sell"`
   - Connect the Function node to the IF node

4. **Add an Email Node** (connected to the "true" output of the IF node):
   - Set recipient(s)
   - Use expressions for subject: `{{$node["Function"].json["subject"]}}`
   - Use expressions for body: `{{$node["Function"].json["body"]}}`

5. **Save and activate the workflow**

### Testing Workflows

You can test workflows using the "Execute Workflow" button in the N8N editor. For webhook triggers, you can send a test request using tools like curl or Postman:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"pair":"EUR/USD","price":1.1234,"signal":"buy"}' http://localhost:5678/webhook/path
```

## Importing and Exporting Workflows

### Exporting Workflows

1. Open the workflow in the N8N editor
2. Click on the menu button in the top right
3. Select "Export"
4. Choose "Download" to save as a JSON file

### Importing Workflows

1. In the workflows list, click "Import from file"
2. Select the JSON file to import
3. Review and confirm the import

## The Forex AI Predefined Workflows

The Forex AI system includes several predefined workflows in the `forex_ai/automation/workflows` directory:

1. `data_collection.json` - For market data collection
2. `alert_system.json` - For trading alerts
3. `reporting.json` - For performance reporting
4. `youtube_processing.json` - For YouTube video analysis

These workflows can be imported into N8N for immediate use.

## Extending Workflows

### Custom Functions

You can extend workflows using the Function node with custom JavaScript:

```javascript
// Custom indicator calculation
const prices = items.map(item => item.json.close);
const sma = prices.reduce((sum, price) => sum + price, 0) / prices.length;
const rsi = calculateRSI(prices); // Custom function

return {
  json: {
    sma: sma,
    rsi: rsi,
    signal: rsi > 70 ? "overbought" : rsi < 30 ? "oversold" : "neutral"
  }
};
```

### Custom Nodes

For advanced use cases, you can develop custom nodes for N8N using the N8N Community Nodes feature.

## Resources

- [N8N Documentation](https://docs.n8n.io/)
- [N8N Node Reference](https://docs.n8n.io/integrations/builtin/app-nodes/)
- [N8N Community](https://community.n8n.io/) 