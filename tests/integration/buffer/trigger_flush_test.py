#!/usr/bin/env python3
"""
手动触发flush来测试L1层处理
"""

import requests
import time
import json
import sqlite3
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_KEY = "memfuse-test-api-key"

def make_request(method: str, endpoint: str, data: dict = None) -> dict:
    """Make HTTP request to MemFuse API."""
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return {"error": str(e)}

def add_many_messages(session_id, count=10):
    """添加足够多的消息来触发flush"""
    print(f"\n📝 添加{count}条消息来触发flush...")
    
    base_messages = [
        "Alice Johnson is a 30-year-old marketing manager at Tesla. She graduated from UCLA with an MBA in 2018.",
        "Bob Wilson is a 45-year-old chef who owns a restaurant in New York. He trained at the Culinary Institute of America.",
        "Carol Davis is a 28-year-old software developer at Apple. She has expertise in iOS development and Swift programming.",
        "David Brown is a 35-year-old doctor specializing in cardiology. He works at Johns Hopkins Hospital.",
        "Emma Garcia is a 32-year-old lawyer working for a top law firm in Chicago. She specializes in corporate law.",
        "Frank Miller is a 40-year-old architect who designed several famous buildings in San Francisco.",
        "Grace Lee is a 26-year-old data scientist at Google. She has a PhD in Statistics from Stanford.",
        "Henry Taylor is a 38-year-old pilot for United Airlines. He has been flying commercial aircraft for 15 years.",
        "Ivy Chen is a 29-year-old fashion designer based in Paris. She launched her own clothing line in 2022.",
        "Jack Anderson is a 33-year-old financial analyst at Goldman Sachs. He specializes in tech stock analysis."
    ]
    
    message_ids = []
    for i in range(count):
        content = base_messages[i % len(base_messages)]
        # 添加变化使每条消息唯一
        content = f"[Message {i+1}] {content} Additional info: ID-{i+1:03d}."
        
        message = {
            "role": "user",
            "content": content,
            "metadata": {"batch": "flush_test", "index": i+1}
        }
        
        print(f"添加消息 {i+1}: {content[:50]}...")
        
        response = make_request("POST", f"/api/v1/sessions/{session_id}/messages", {
            "messages": [message]
        })
        
        if "error" not in response:
            print(f"✅ 消息 {i+1} 添加成功")
            if "data" in response and "message_ids" in response["data"]:
                message_ids.extend(response["data"]["message_ids"])
        else:
            print(f"❌ 消息 {i+1} 添加失败: {response}")
        
        # 短暂延迟避免过快请求
        time.sleep(0.1)
    
    print(f"✅ 总共添加了 {len(message_ids)} 条消息")
    return message_ids

def check_facts_table():
    """检查所有用户的facts表"""
    print("\n🔍 检查所有用户的L1 facts表...")
    
    data_dir = Path("data")
    user_dirs = [d for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 36]
    
    total_facts = 0
    for user_dir in user_dirs:
        facts_db_path = user_dir / "facts.db"
        if facts_db_path.exists():
            try:
                conn = sqlite3.connect(str(facts_db_path))
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM l1_facts")
                count = cursor.fetchone()[0]
                total_facts += count
                print(f"📊 用户 {user_dir.name[:8]}... 的facts表中有 {count} 条记录")
                
                if count > 0:
                    cursor.execute("SELECT id, content, confidence, created_at FROM l1_facts ORDER BY created_at DESC LIMIT 3")
                    facts = cursor.fetchall()
                    print("📝 最新的3个事实:")
                    for fact in facts:
                        print(f"  - 置信度: {fact[2]:.2f}, 内容: {fact[1][:80]}...")
                
                conn.close()
                
            except Exception as e:
                print(f"❌ 检查facts表失败: {e}")
    
    return total_facts > 0

def main():
    """主函数"""
    print("🚀 手动触发flush测试L1层")
    print("=" * 50)
    
    # 检查服务器
    health = make_request("GET", "/api/v1/health")
    if "error" in health:
        print("❌ 服务器不可用")
        return
    print("✅ 服务器运行正常")
    
    # 使用现有的测试会话
    session_id = "838c60d3-aa22-4445-8935-e42d65a22eac"  # 从之前的测试获取
    
    print(f"📋 使用会话ID: {session_id}")
    
    # 添加足够多的消息来触发flush
    message_ids = add_many_messages(session_id, 15)  # 添加15条消息，超过buffer限制
    
    print("\n⏳ 等待buffer处理和flush...")
    print("   (RoundBuffer max_size=5, HybridBuffer max_size=5)")
    print("   (应该触发多次transfer和flush)")
    
    # 等待处理
    for i in range(20):  # 等待20秒
        time.sleep(1)
        print(f"   等待中... {i+1}/20秒", end="\r")
    print("\n")
    
    # 检查facts表
    has_facts = check_facts_table()
    
    if has_facts:
        print("\n🎉 成功！L1层已经处理数据并提取了事实")
    else:
        print("\n⚠️ L1层可能还在处理中，或者存在问题")
        print("   建议检查服务器日志")
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()
