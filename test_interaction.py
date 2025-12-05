# test_interaction.py
from src.core.interaction import interactor
from src.utils.logger import sys_logger

def main():
    print(">>> Starting Interaction Test...")
    
    # 模拟一些数据
    mock_data = {
        "Current_Idea": "Use Mamba for Weather Forecasting",
        "Status": "Drafting"
    }
    mock_context = "We plan to compare Mamba with Transformer on the Traffic dataset."
    
    # 发起交互
    # 运行到这里时，程序会暂停。
    # 请你去 workspace/xxx/reviews/test_phase_review.md 修改内容
    # 比如把 Action 改为 REVISE，反馈写：“请改用 Weather 数据集，不要用 Traffic。”
    result = interactor.start_review("test_phase", mock_data, mock_context)
    
    print("\n" + "="*50)
    print("✅ TEST RESULT")
    print("="*50)
    print(f"Action:   {result.action}")
    print(f"Feedback: {result.feedback_en}")
    print(f"Comments: {result.comments}")
    print("="*50)

if __name__ == "__main__":
    main()