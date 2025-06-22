"""
API Endpoint Testing Script

This script tests all available API endpoints in the Forex AI Trading System.
It uses the OpenAPI schema to discover endpoints and tests them with appropriate methods.
"""

import argparse
import json
import sys
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

def get_openapi_schema(base_url: str) -> Dict[str, Any]:
    """
    Fetch the OpenAPI schema from the API server.
    
    Args:
        base_url: Base URL of the API server
        
    Returns:
        Dict containing the OpenAPI schema
    """
    try:
        response = requests.get(f"{base_url}/api/openapi.json", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OpenAPI schema: {str(e)}")
        print(f"Make sure the API server is running at {base_url}")
        sys.exit(1)

def extract_endpoints_from_schema(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract endpoints from the OpenAPI schema.
    
    Args:
        schema: OpenAPI schema
        
    Returns:
        List of endpoint information dictionaries
    """
    endpoints = []
    
    for path, path_item in schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue
                
            tags = operation.get("tags", ["other"])
            summary = operation.get("summary", "")
            description = operation.get("description", "")
            
            # Extract parameters
            parameters = []
            for param in operation.get("parameters", []):
                parameters.append({
                    "name": param.get("name", ""),
                    "in": param.get("in", ""),
                    "required": param.get("required", False),
                    "schema": param.get("schema", {})
                })
            
            endpoints.append({
                "path": path,
                "method": method.upper(),
                "tags": tags,
                "summary": summary,
                "description": description,
                "parameters": parameters
            })
    
    return endpoints

def test_endpoint(base_url: str, endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a single API endpoint.
    
    Args:
        base_url: Base URL of the API server
        endpoint: Endpoint information
        
    Returns:
        Dictionary with test results
    """
    path = endpoint["path"]
    method = endpoint["method"].lower()
    url = f"{base_url}{path}"
    
    # Replace path parameters with sample values
    for param in endpoint["parameters"]:
        if param["in"] == "path" and "{" + param["name"] + "}" in url:
            # Use appropriate sample values based on parameter name
            if "id" in param["name"].lower():
                sample_value = "1"
            elif "account" in param["name"].lower():
                sample_value = "primary"
            elif "instrument" in param["name"].lower() or "pair" in param["name"].lower():
                sample_value = "EUR_USD"
            elif "timeframe" in param["name"].lower():
                sample_value = "H1"
            elif "strategy" in param["name"].lower():
                sample_value = "sma_crossover"
            else:
                sample_value = "sample"
                
            url = url.replace("{" + param["name"] + "}", sample_value)
    
    # Prepare query parameters
    query_params = {}
    for param in endpoint["parameters"]:
        if param["in"] == "query":
            # Use appropriate sample values based on parameter name
            if "limit" in param["name"].lower():
                query_params[param["name"]] = 5
            elif "offset" in param["name"].lower():
                query_params[param["name"]] = 0
            elif "start" in param["name"].lower():
                query_params[param["name"]] = "2023-01-01"
            elif "end" in param["name"].lower():
                query_params[param["name"]] = "2023-01-31"
            elif param["schema"].get("type") == "boolean":
                query_params[param["name"]] = True
            elif param["schema"].get("type") == "integer":
                query_params[param["name"]] = 1
            elif param["schema"].get("type") == "number":
                query_params[param["name"]] = 1.0
            else:
                query_params[param["name"]] = "sample"
    
    start_time = time.time()
    try:
        if method == "get":
            response = requests.get(url, params=query_params, timeout=10)
        elif method == "post":
            response = requests.post(url, json={}, timeout=10)
        elif method == "put":
            response = requests.put(url, json={}, timeout=10)
        elif method == "delete":
            response = requests.delete(url, timeout=10)
        elif method == "patch":
            response = requests.patch(url, json={}, timeout=10)
        else:
            return {
                "status": "skipped",
                "status_code": None,
                "message": f"Unsupported method: {method}",
                "response_time": 0
            }
            
        response_time = time.time() - start_time
        
        # Check if response is valid JSON
        try:
            response_json = response.json()
            is_valid_json = True
        except:
            response_json = None
            is_valid_json = False
        
        # Determine status based on status code
        if 200 <= response.status_code < 300:
            status = "success"
        elif 300 <= response.status_code < 400:
            status = "redirect"
        elif 400 <= response.status_code < 500:
            status = "client_error"
        else:
            status = "server_error"
            
        return {
            "status": status,
            "status_code": response.status_code,
            "message": f"HTTP {response.status_code}",
            "response_time": response_time,
            "is_valid_json": is_valid_json
        }
        
    except requests.exceptions.Timeout:
        response_time = time.time() - start_time
        return {
            "status": "timeout",
            "status_code": None,
            "message": "Request timed out after 10 seconds",
            "response_time": response_time
        }
    except requests.exceptions.RequestException as e:
        response_time = time.time() - start_time
        return {
            "status": "error",
            "status_code": None,
            "message": str(e),
            "response_time": response_time
        }

def main():
    """Main function to run the API endpoint tests."""
    parser = argparse.ArgumentParser(description="Test API endpoints")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"=== Forex AI Trading System API Endpoint Test ===")
    print(f"Testing API endpoints at {base_url}")
    print()
    
    # Get OpenAPI schema
    print(f"Fetching OpenAPI schema...")
    schema = get_openapi_schema(base_url)
    
    # Extract endpoints
    endpoints = extract_endpoints_from_schema(schema)
    print(f"Found {len(endpoints)} endpoints to test")
    print()
    
    # Group endpoints by tags
    endpoints_by_tag = {}
    for endpoint in endpoints:
        for tag in endpoint["tags"]:
            if tag not in endpoints_by_tag:
                endpoints_by_tag[tag] = []
            endpoints_by_tag[tag].append(endpoint)
    
    # Test results
    results = []
    success_count = 0
    error_count = 0
    
    # Test endpoints by tag
    for tag in sorted(endpoints_by_tag.keys()):
        print(f"Testing {tag.upper()} endpoints:")
        
        for endpoint in endpoints_by_tag[tag]:
            path = endpoint["path"]
            method = endpoint["method"]
            
            print(f"  {method.ljust(6)} {path}", end="", flush=True)
            
            # Test endpoint
            result = test_endpoint(base_url, endpoint)
            
            # Update counts
            if result["status"] in ["success", "redirect"]:
                success_count += 1
            else:
                error_count += 1
            
            # Print result
            if result["status"] == "success":
                status_color = "\033[92m"  # Green
            elif result["status"] == "redirect":
                status_color = "\033[93m"  # Yellow
            else:
                status_color = "\033[91m"  # Red
                
            reset_color = "\033[0m"
            print(f" {status_color}{result['status'].upper()}{reset_color} ({result['status_code']}) - {result['response_time']:.2f}s")
            
            # Add result to list
            results.append({
                "path": path,
                "method": method,
                "tags": endpoint["tags"],
                "summary": endpoint["summary"],
                "result": result
            })
        
        print()
    
    # Print summary
    total = len(results)
    print(f"=== Test Summary ===")
    print(f"Total endpoints tested: {total}")
    print(f"Success: \033[92m{success_count}\033[0m ({success_count / total * 100:.1f}%)")
    print(f"Errors: \033[91m{error_count}\033[0m ({error_count / total * 100:.1f}%)")
    print()
    
    # Save results to file if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "base_url": base_url,
                "total_endpoints": total,
                "success_count": success_count,
                "error_count": error_count,
                "results": results
            }, f, indent=2)
        print(f"Test results saved to {args.output}")

if __name__ == "__main__":
    main()