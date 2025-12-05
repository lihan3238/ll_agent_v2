# test_researcher.py
from src.agents.researcher import ResearcherAgent

def main():
    agent = ResearcherAgent()
    
    idea = "I want to use Mamba state space models for time series forecasting on weather data, comparing it with Transformer."
    
    try:
        report = agent.run(user_idea=idea)
        
        print("\n" + "="*50)
        print("RESEARCH REPORT (Visual Check)")
        print("="*50)
        print(f"Refined Idea: {report.refined_idea}\n")
        print(f"Keywords: {report.keywords}\n")
        print(f"Related Work: {report.related_work_summary[:200]}...\n")
        print("-" * 20)
        print("Top Papers Found:")
        for p in report.top_papers:
            print(f"* {p.title} ({p.year}) - Citations: {p.citations}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()