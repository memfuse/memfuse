#!/usr/bin/env python3
"""
Verify chunks data stored in Qdrant database
Check if contextual chunks are generated correctly
"""

import os
from qdrant_client import QdrantClient


def verify_qdrant_data():
    """Verify chunks data in Qdrant database"""

    data_dir = "/Users/mxue/GitRepos/MemFuse/memfuse/data"
    print(f"🔍 Checking data directory: {data_dir}")

    # Find all user directories
    user_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "__pycache__"]
    print(f"📁 Found {len(user_dirs)} user directories: {user_dirs}")

    total_chunks = 0
    contextual_chunks = 0
    gpt_enhanced_chunks = 0

    for user_dir in user_dirs:
        qdrant_path = os.path.join(data_dir, user_dir, "qdrant")
        if not os.path.exists(qdrant_path):
            print(f"❌ {user_dir}: No qdrant directory found")
            continue

        print(f"\n🔍 Checking Qdrant data for user {user_dir}...")

        try:
            # Connect to Qdrant
            client = QdrantClient(path=qdrant_path)

            # Get collection information
            collections = client.get_collections().collections
            print(f"📊 Number of collections: {len(collections)}")

            for collection in collections:
                collection_name = collection.name
                print(f"📦 Collection name: {collection_name}")

                # Get collection info
                collection_info = client.get_collection(collection_name)
                points_count = collection_info.points_count
                print(f"📈 Points count: {points_count}")
                total_chunks += points_count

                if points_count > 0:
                    # Get first few points to check data
                    points = client.scroll(
                        collection_name=collection_name,
                        limit=min(10, points_count),
                        with_payload=True,
                        with_vectors=False
                    )[0]

                    print(f"🔍 Checking first {len(points)} chunks:")

                    for i, point in enumerate(points):
                        payload = point.payload
                        metadata = payload.get('metadata', {})

                        print(f"  📄 Chunk {i+1}:")
                        print(f"    ID: {point.id}")
                        print(f"    Payload Keys: {list(payload.keys())}")
                        print(f"    Metadata Keys: {list(metadata.keys())}")
                        print(f"    Strategy: {metadata.get('strategy', 'N/A')}")
                        print(f"    Has Context: {metadata.get('has_context', 'N/A')}")
                        print(f"    GPT Enhanced: {metadata.get('gpt_enhanced', 'N/A')}")
                        print(f"    Session ID: {metadata.get('session_id', 'N/A')}")
                        print(f"    Content Length: {len(payload.get('content', ''))}")
                        print(f"    Content Preview: {payload.get('content', '')[:100]}...")

                        # Check if this is a contextual chunk
                        if metadata.get('has_context'):
                            contextual_chunks += 1
                            print(f"    ✅ This is a contextual chunk!")

                            # Show contextual information
                            if 'context_chunk_ids' in metadata:
                                print(f"    🔗 Context Chunk IDs: {metadata['context_chunk_ids']}")
                            if 'context_window_size' in metadata:
                                print(f"    📏 Context Window Size: {metadata['context_window_size']}")

                        if metadata.get('gpt_enhanced'):
                            gpt_enhanced_chunks += 1
                            print(f"    🤖 This chunk is GPT enhanced!")

                            # Show GPT enhancement information
                            if 'contextual_description' in metadata:
                                desc = metadata['contextual_description'][:100] + "..."
                                print(f"    📝 Contextual Description: {desc}")

                        print()

        except Exception as e:
            print(f"❌ Error checking user {user_dir}: {e}")
            continue

    print(f"\n📊 Summary:")
    print(f"✅ Total chunks: {total_chunks}")
    print(f"🧠 Contextual chunks: {contextual_chunks}")
    print(f"🤖 GPT enhanced chunks: {gpt_enhanced_chunks}")
    print(f"📈 Contextual ratio: {contextual_chunks/total_chunks*100:.1f}%" if total_chunks > 0 else "📈 Contextual ratio: 0%")
    print(f"📈 Enhancement ratio: {gpt_enhanced_chunks/total_chunks*100:.1f}%" if total_chunks > 0 else "📈 Enhancement ratio: 0%")

    if total_chunks > 0:
        print(f"✅ Chunks data is indeed stored in Qdrant!")
    else:
        print(f"❌ No chunks data found in Qdrant!")

    if contextual_chunks > 0 or gpt_enhanced_chunks > 0:
        print(f"✅ Found enhanced chunks!")
    else:
        print(f"❌ No enhanced chunks found!")


def verify_contextual_chunking_effectiveness():
    """Verify the effectiveness of contextual chunking"""
    print("\n🎯 Verifying contextual chunking effectiveness...\n")

    data_dir = "/Users/mxue/GitRepos/MemFuse/memfuse/data"
    user_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "__pycache__"]

    total_chunks = 0
    total_contextual = 0
    total_enhanced = 0
    sessions_with_context = set()

    for user_dir in user_dirs:
        qdrant_path = os.path.join(data_dir, user_dir, "qdrant")
        if not os.path.exists(qdrant_path):
            continue

        try:
            client = QdrantClient(path=qdrant_path)
            collections = client.get_collections().collections

            for collection in collections:
                collection_name = collection.name
                collection_info = client.get_collection(collection_name)

                if collection_info.points_count > 0:
                    points = client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        with_payload=True,
                        with_vectors=False
                    )[0]

                    for point in points:
                        metadata = point.payload.get('metadata', {})
                        total_chunks += 1

                        if metadata.get('has_context'):
                            total_contextual += 1
                            sessions_with_context.add(metadata.get('session_id'))

                        if metadata.get('gpt_enhanced'):
                            total_enhanced += 1

        except Exception as e:
            print(f"❌ Error checking effectiveness for {user_dir}: {e}")
            continue

    print(f"📈 Contextual Chunking Effectiveness Report:")
    print(f"   Total chunks processed: {total_chunks}")
    print(f"   Chunks with context: {total_contextual} ({total_contextual/total_chunks*100:.1f}%)")
    print(f"   GPT enhanced chunks: {total_enhanced} ({total_enhanced/total_chunks*100:.1f}%)")
    print(f"   Sessions with contextual chunks: {len(sessions_with_context)}")

    if total_contextual > 0 or total_enhanced > 0:
        print(f"✅ Contextual chunking is working effectively!")
    else:
        print(f"⚠️  Contextual chunking may not be working as expected")
        print(f"   This could be normal if all chunks are from independent sessions")


if __name__ == "__main__":
    verify_qdrant_data()
    verify_contextual_chunking_effectiveness()
