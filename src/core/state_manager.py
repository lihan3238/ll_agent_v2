# src/core/state_manager.py
import os
import yaml
from src.core.state import ProjectState
from src.utils.logger import sys_logger

class StateManager:
    def __init__(self):
        self.config = self._load_config()
        self.project_name = self.config.get("project", {}).get("name", "default")
        self.workspace_dir = os.path.join("workspace", self.project_name)
        self.state_file = os.path.join(self.workspace_dir, "project_state.json")
        
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def _load_config(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_state(self) -> ProjectState:
        """加载项目状态，如果不存在则初始化一个新的"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    content = f.read()
                # Pydantic 自动反序列化
                state = ProjectState.model_validate_json(content)
                sys_logger.info(f"📂 Project State loaded from {self.state_file}")
                return state
            except Exception as e:
                sys_logger.error(f"Failed to load state: {e}. Starting fresh.")
        
        # 初始化新状态
        sys_logger.info("✨ Initializing new Project State.")
        return ProjectState(project_name=self.project_name)

    def save_state(self, state: ProjectState):
        """保存当前状态"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))
            sys_logger.info(f"💾 Project State saved to {self.state_file}")
        except Exception as e:
            sys_logger.error(f"Failed to save state: {e}")

state_manager = StateManager()