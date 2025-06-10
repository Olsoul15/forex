"""
Example of using the Supabase client for the Forex AI Trading System.

This module demonstrates common database operations using the Supabase client.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from forex_ai.data.storage.supabase_client import get_supabase_db_client
from forex_ai.exceptions import DatabaseError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_connection():
    """Test the Supabase connection."""
    try:
        client = get_supabase_db_client()
        logger.info("Supabase client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        return False


def test_insert():
    """Test inserting data into a test table."""
    client = get_supabase_db_client()

    try:
        # Insert a record
        result = client.insert_one(
            "test_table",
            {"name": "Example Record", "value": "Inserted via Python client"},
        )

        if result:
            logger.info(f"Record inserted with ID: {result}")
            return result
        else:
            logger.warning("Insert operation did not return an ID")
            return None
    except DatabaseError as e:
        logger.error(f"Insert failed: {str(e)}")
        return None


def test_query():
    """Test querying data from a test table."""
    client = get_supabase_db_client()

    try:
        # Query all records
        records = client.fetch_all("test_table", order_by="-created_at", limit=5)

        if records:
            logger.info(f"Found {len(records)} records")
            for record in records:
                logger.info(f"Record: {record}")
            return records
        else:
            logger.info("No records found")
            return []
    except DatabaseError as e:
        logger.error(f"Query failed: {str(e)}")
        return []


def test_update():
    """Test updating data in a test table."""
    client = get_supabase_db_client()

    try:
        # Get the latest record
        latest = client.fetch_all("test_table", order_by="-created_at", limit=1)

        if not latest:
            logger.warning("No records found to update")
            return False

        record_id = latest[0]["id"]

        # Update the record
        updated_count = client.update(
            "test_table", {"value": "Updated via Python client"}, {"id": record_id}
        )

        if updated_count > 0:
            logger.info(f"Updated {updated_count} records")
            return True
        else:
            logger.warning("No records were updated")
            return False
    except DatabaseError as e:
        logger.error(f"Update failed: {str(e)}")
        return False


def test_delete():
    """Test deleting data from a test table."""
    client = get_supabase_db_client()

    try:
        # Get the oldest record
        oldest = client.fetch_all("test_table", order_by="created_at", limit=1)

        if not oldest:
            logger.warning("No records found to delete")
            return False

        record_id = oldest[0]["id"]

        # Delete the record
        deleted_count = client.delete("test_table", {"id": record_id})

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} records")
            return True
        else:
            logger.warning("No records were deleted")
            return False
    except DatabaseError as e:
        logger.error(f"Delete failed: {str(e)}")
        return False


def main():
    """Run the example tests."""
    print("Supabase Client Example")
    print("======================")

    # Test connection
    print("\nTesting connection...")
    if test_connection():
        print("✅ Supabase connection successful")
    else:
        print("❌ Supabase connection failed")
        return

    # Test operations
    print("\nTesting database operations:")

    print("\n1. Inserting a record...")
    insert_id = test_insert()
    if insert_id:
        print(f"✅ Record inserted with ID: {insert_id}")
    else:
        print("❌ Insert failed")

    print("\n2. Querying records...")
    records = test_query()
    if records:
        print(f"✅ Found {len(records)} records")
        for record in records[:3]:  # Show at most 3 records
            print(f"   - {record['name']}: {record['value']}")
    else:
        print("❌ Query failed or no records found")

    print("\n3. Updating a record...")
    if test_update():
        print("✅ Record updated successfully")
    else:
        print("❌ Update failed")

    print("\n4. Deleting a record...")
    if test_delete():
        print("✅ Record deleted successfully")
    else:
        print("❌ Delete failed")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
