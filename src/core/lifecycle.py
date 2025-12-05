# src/core/lifecycle.py
from abc import ABC, abstractmethod
from src.core.state import ProjectState
from src.core.state_manager import state_manager
from src.utils.logger import sys_logger

class BasePhase(ABC):
    def __init__(self, phase_name: str):
        self.phase_name = phase_name

    def execute(self) -> ProjectState:
        """
        [Template Method] 标准执行流程
        """
        sys_logger.info(f"\n{'='*20} PHASE START: {self.phase_name.upper()} {'='*20}")
        
        # 1. 加载最新状态
        state = state_manager.load_state()
        state.current_phase = self.phase_name
        state_manager.save_state(state) # 更新指针

        # 2. 检查是否已有成果 (断点恢复)
        if self.check_completion(state):
            sys_logger.info(f"⏩ Phase {self.phase_name} already completed. Skipping...")
            return state

        # 3. 运行具体业务逻辑
        new_state = self.run_phase_logic(state)
        
        # 4. 最终保存
        state_manager.save_state(new_state)
        sys_logger.info(f"✅ PHASE COMPLETE: {self.phase_name.upper()}")
        
        return new_state

    @abstractmethod
    def check_completion(self, state: ProjectState) -> bool:
        """Check if phase is done"""
        pass

    @abstractmethod
    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        """Core logic implementation"""
        pass