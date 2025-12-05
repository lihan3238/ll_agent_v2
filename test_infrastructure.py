# test_infrastructure.py
from pydantic import BaseModel, Field
from typing import List
from src.agents.base import BaseAgent

# 1. 定义预期的输出格式 (Schema)
class PersonInfo(BaseModel):
    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., description="Age extracted or estimated")
    skills: List[str] = Field(..., description="List of technical skills")

# 2. 定义具体的测试 Agent
class DemoAgent(BaseAgent[PersonInfo]):
    def __init__(self):

        super().__init__(role_name="demo_agent", output_schema=PersonInfo)

def main():
    print(">>> Starting Infrastructure Test...")
    
    # 实例化 Agent
    agent = DemoAgent()
    
    # 输入测试数据
    user_input = "My name is Alice, I am 28 years old. I love Python, AI, and Pizza."
    
    try:
        # 运行
        result = agent.run(user_input=user_input)
        
        print("\n>>> ✅ Test Passed! Structured Output Received:")
        print(f"Name: {result.name}")
        print(f"Age: {result.age}")
        print(f"Skills: {result.skills}")
        print(f"Raw Object: {result}")
        
    except Exception as e:
        print(f"\n>>> ❌ Test Failed: {e}")

if __name__ == "__main__":
    main()