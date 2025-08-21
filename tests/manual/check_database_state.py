#!/usr/bin/env python3
"""
Database state checker for MemFuse buffer system.

This script helps verify database state by directly querying the database tables.
Useful for debugging and verifying that data has been properly persisted.

Usage:
    python tests/manual/check_database_state.py
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import asyncpg
except ImportError:
    print("❌ asyncpg not installed. Install with: pip install asyncpg")
    sys.exit(1)


async def connect_to_database():
    """Connect to the MemFuse database."""
    try:
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="memfuse",
            password="memfuse",
            database="memfuse"
        )
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("Make sure PostgreSQL is running and MemFuse database exists")
        return None


async def check_recent_messages(conn, minutes=60):
    """Check for messages added in the last N minutes."""
    print(f"🔍 Checking for messages added in the last {minutes} minutes...")
    
    # Calculate cutoff time
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    
    try:
        # Query messages table
        query = """
        SELECT 
            m.id,
            m.content,
            m.role,
            m.created_at,
            s.name as session_name,
            u.name as user_name
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        JOIN users u ON s.user_id = u.id
        WHERE m.created_at >= $1
        ORDER BY m.created_at DESC
        LIMIT 20
        """
        
        rows = await conn.fetch(query, cutoff_time)
        
        if rows:
            print(f"✅ Found {len(rows)} recent messages:")
            for row in rows:
                print(f"   [{row['created_at']}] {row['user_name']}/{row['session_name']}")
                print(f"      {row['role']}: {row['content'][:100]}...")
                print()
        else:
            print(f"⚠️  No messages found in the last {minutes} minutes")
            
        return len(rows)
        
    except Exception as e:
        print(f"❌ Error querying messages: {e}")
        return 0


async def check_message_rounds(conn, minutes=60):
    """Check for message rounds added in the last N minutes."""
    print(f"🔍 Checking for message rounds added in the last {minutes} minutes...")
    
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    
    try:
        # Query message_rounds table
        query = """
        SELECT 
            mr.id,
            mr.round_data,
            mr.created_at,
            s.name as session_name,
            u.name as user_name
        FROM message_rounds mr
        JOIN sessions s ON mr.session_id = s.id
        JOIN users u ON s.user_id = u.id
        WHERE mr.created_at >= $1
        ORDER BY mr.created_at DESC
        LIMIT 10
        """
        
        rows = await conn.fetch(query, cutoff_time)
        
        if rows:
            print(f"✅ Found {len(rows)} recent message rounds:")
            for row in rows:
                print(f"   [{row['created_at']}] {row['user_name']}/{row['session_name']}")
                # Parse round_data if it's JSON
                try:
                    import json
                    round_data = json.loads(row['round_data']) if isinstance(row['round_data'], str) else row['round_data']
                    if isinstance(round_data, list) and len(round_data) > 0:
                        first_msg = round_data[0]
                        content = first_msg.get('content', 'No content')[:50]
                        print(f"      Round with {len(round_data)} messages: {content}...")
                except:
                    print(f"      Round data: {str(row['round_data'])[:50]}...")
                print()
        else:
            print(f"⚠️  No message rounds found in the last {minutes} minutes")
            
        return len(rows)
        
    except Exception as e:
        print(f"❌ Error querying message rounds: {e}")
        return 0


async def check_database_tables(conn):
    """Check what tables exist in the database."""
    print("🔍 Checking database schema...")
    
    try:
        # Get table names
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        
        rows = await conn.fetch(query)
        
        if rows:
            print("✅ Found tables:")
            for row in rows:
                table_name = row['table_name']
                
                # Get row count for each table
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = await conn.fetchrow(count_query)
                count = count_result['count']
                
                print(f"   {table_name}: {count} rows")
        else:
            print("⚠️  No tables found")
            
    except Exception as e:
        print(f"❌ Error checking tables: {e}")


async def check_buffer_related_data(conn):
    """Check for data that might have come from buffer flush."""
    print("🔍 Checking for buffer-related data patterns...")
    
    try:
        # Look for test data patterns
        test_patterns = [
            "Force flush timeout test",
            "Manual flush test", 
            "Graceful shutdown test",
            "space exploration",  # From quickstart example
            "Mars"
        ]
        
        for pattern in test_patterns:
            query = """
            SELECT COUNT(*) as count
            FROM messages 
            WHERE content ILIKE $1
            """
            
            result = await conn.fetchrow(query, f"%{pattern}%")
            count = result['count']
            
            if count > 0:
                print(f"✅ Found {count} messages matching '{pattern}'")
                
                # Get recent examples
                detail_query = """
                SELECT content, created_at, role
                FROM messages 
                WHERE content ILIKE $1
                ORDER BY created_at DESC
                LIMIT 3
                """
                
                examples = await conn.fetch(detail_query, f"%{pattern}%")
                for example in examples:
                    print(f"   [{example['created_at']}] {example['role']}: {example['content'][:80]}...")
            else:
                print(f"⚠️  No messages found matching '{pattern}'")
        
    except Exception as e:
        print(f"❌ Error checking buffer data: {e}")


async def main():
    """Main verification function."""
    print("🔍 MemFuse Database Consistency Checker")
    print("=" * 50)
    
    # Connect to database
    conn = await connect_to_database()
    if not conn:
        return False
    
    try:
        # Check database schema
        await check_database_tables(conn)
        print()
        
        # Check recent messages (last hour)
        message_count = await check_recent_messages(conn, minutes=60)
        print()
        
        # Check recent message rounds (last hour)
        round_count = await check_message_rounds(conn, minutes=60)
        print()
        
        # Check for buffer-related test data
        await check_buffer_related_data(conn)
        print()
        
        # Summary
        print("📋 Summary")
        print("=" * 20)
        print(f"Recent messages: {message_count}")
        print(f"Recent rounds: {round_count}")
        
        if message_count > 0 or round_count > 0:
            print("✅ Database contains recent data - buffer flush appears to be working")
            return True
        else:
            print("⚠️  No recent data found - may indicate buffer flush issues")
            return False
            
    finally:
        await conn.close()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
