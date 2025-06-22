#!/bin/bash
# Run storage tests with coverage

echo "Running storage tests..."
pytest tests/storage \
    --cov=forex_ai.data.storage \
    --cov-report=term-missing \
    --cov-report=html:coverage_report \
    -v \
    "$@"

if [ $? -eq 0 ]; then
    echo
    echo "Tests completed successfully!"
    echo "Coverage report available in coverage_report/index.html"
else
    echo
    echo "Tests failed with exit code $?"
fi 