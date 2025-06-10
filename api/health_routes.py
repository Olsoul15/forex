"""
Health check API routes.

This module provides API routes for checking system health, component status,
and managing error locks.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from http import HTTPStatus

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
health_bp = Blueprint("health", __name__, url_prefix="/api/health")


@health_bp.route("/", methods=["GET"])
def get_health_status():
    """
    Get overall system health status.

    Returns:
        JSON with system health status
    """
    try:
        # Get health monitor from app context
        monitor = current_app.health_monitor

        # Generate health report
        report = monitor.get_health_report()

        # Return report as JSON
        return jsonify(report), HTTPStatus.OK
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error getting health status: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@health_bp.route("/components", methods=["GET"])
def get_component_statuses():
    """
    Get status of all components.

    Returns:
        JSON with component health statuses
    """
    try:
        # Get health monitor from app context
        monitor = current_app.health_monitor

        # Get health report
        report = monitor.get_health_report()

        # Extract component statuses
        component_statuses = report.get("component_statuses", {})

        # Return statuses as JSON
        return (
            jsonify(
                {
                    "components": component_statuses,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.OK,
        )
    except Exception as e:
        logger.error(f"Error getting component statuses: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error getting component statuses: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@health_bp.route("/components/<component_name>", methods=["GET"])
def get_component_status(component_name):
    """
    Get status of a specific component.

    Args:
        component_name: Name of the component

    Returns:
        JSON with component health status
    """
    try:
        # Get health monitor from app context
        monitor = current_app.health_monitor

        # Get health report
        report = monitor.get_health_report()

        # Extract component statuses
        component_statuses = report.get("component_statuses", {})

        # Check if component exists
        if component_name not in component_statuses:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Component {component_name} not found",
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                HTTPStatus.NOT_FOUND,
            )

        # Return component status as JSON
        return (
            jsonify(
                {
                    "component": component_name,
                    "status": component_statuses[component_name],
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.OK,
        )
    except Exception as e:
        logger.error(f"Error getting component status: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error getting component status: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@health_bp.route("/history", methods=["GET"])
def get_health_history():
    """
    Get health status history.

    Query parameters:
        limit: Maximum number of history entries to return (default: 20)

    Returns:
        JSON with health status history
    """
    try:
        # Get health monitor from app context
        monitor = current_app.health_monitor

        # Get limit from query parameter
        limit = request.args.get("limit", default=20, type=int)

        # Get health history
        history = monitor.health_history[-limit:] if monitor.health_history else []

        # Return history as JSON
        return (
            jsonify(
                {
                    "history": history,
                    "total_entries": len(monitor.health_history),
                    "returned_entries": len(history),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.OK,
        )
    except Exception as e:
        logger.error(f"Error getting health history: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error getting health history: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@health_bp.route("/components/<component_name>/history", methods=["GET"])
def get_component_history(component_name):
    """
    Get health history for a specific component.

    Args:
        component_name: Name of the component

    Query parameters:
        limit: Maximum number of history entries to return (default: 20)

    Returns:
        JSON with component health history
    """
    try:
        # Get health monitor from app context
        monitor = current_app.health_monitor

        # Get limit from query parameter
        limit = request.args.get("limit", default=20, type=int)

        # Get component history
        component_history = monitor.get_component_history(component_name, limit)

        # Return history as JSON
        return (
            jsonify(
                {
                    "component": component_name,
                    "history": component_history,
                    "entries": len(component_history),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.OK,
        )
    except Exception as e:
        logger.error(f"Error getting component history: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error getting component history: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@health_bp.route("/locks", methods=["GET"])
def get_lock_status():
    """
    Get status of all error locks.

    Returns:
        JSON with lock statuses
    """
    try:
        # Get lock manager from app context
        lock_manager = current_app.lock_manager

        # Get all locks
        locks = lock_manager.get_all_locks()

        # Return locks as JSON
        return (
            jsonify(
                {
                    "locks": locks,
                    "total_locks": len(locks),
                    "active_locks": len(lock_manager.get_active_locks()),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.OK,
        )
    except Exception as e:
        logger.error(f"Error getting lock status: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error getting lock status: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@health_bp.route("/locks/active", methods=["GET"])
def get_active_locks():
    """
    Get status of active error locks.

    Returns:
        JSON with active lock statuses
    """
    try:
        # Get lock manager from app context
        lock_manager = current_app.lock_manager

        # Get active locks
        active_locks = lock_manager.get_active_locks()

        # Return active locks as JSON
        return (
            jsonify(
                {
                    "locks": active_locks,
                    "count": len(active_locks),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.OK,
        )
    except Exception as e:
        logger.error(f"Error getting active locks: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error getting active locks: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@health_bp.route("/locks/<component_name>/release", methods=["POST"])
def release_lock(component_name):
    """
    Release a lock for a component.

    Args:
        component_name: Name of the component

    Returns:
        JSON with lock release result
    """
    try:
        # Get lock manager from app context
        lock_manager = current_app.lock_manager

        # Release lock
        released = lock_manager.release(component_name)

        if released:
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": f"Lock for {component_name} released successfully",
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                HTTPStatus.OK,
            )
        else:
            return (
                jsonify(
                    {
                        "status": "warning",
                        "message": f"Component {component_name} is not locked",
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                HTTPStatus.OK,
            )
    except Exception as e:
        logger.error(f"Error releasing lock: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error releasing lock: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@health_bp.route("/locks/reset", methods=["POST"])
def reset_all_locks():
    """
    Reset all locks in the system.

    Returns:
        JSON with lock reset result
    """
    try:
        # Get lock manager from app context
        lock_manager = current_app.lock_manager

        # Reset all locks
        reset_count = lock_manager.reset_all_locks()

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Reset {reset_count} locks successfully",
                    "reset_count": reset_count,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.OK,
        )
    except Exception as e:
        logger.error(f"Error resetting locks: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error resetting locks: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
