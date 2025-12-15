"""
State Management Module
State management: tracks tool call counts, step sequences, checks fallback, etc.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class StepStatus(Enum):
    """Step status"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ExecutionState:
    """Execution state management class"""
    
    def __init__(self):
        """Initialize execution state"""
        self.query: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Tool call records
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_call_count: Dict[str, int] = {}
        
        # Step sequence
        self.steps: List[Dict[str, Any]] = []
        self.current_step: int = 0
        
        # Execution plan
        self.plan: Optional[Dict[str, Any]] = None
        
        # Observations
        self.observations: List[str] = []
        
        # Thought process
        self.thoughts: List[str] = []
        
        # Fallback flag
        self.needs_fallback: bool = False
        self.fallback_reason: Optional[str] = None
        
        # Final answer
        self.final_answer: Optional[str] = None
    
    def start_execution(self, query: str, plan: Dict[str, Any]) -> None:
        """Start execution"""
        self.query = query
        self.plan = plan
        self.start_time = datetime.now()
        self.current_step = 0
    
    def add_thought(self, thought: str) -> None:
        """Add thought"""
        self.thoughts.append({
            "step": self.current_step,
            "thought": thought,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Dict[str, Any],
        status: str = "success"
    ) -> None:
        """Record tool call"""
        call_record = {
            "step": self.current_step,
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        self.tool_calls.append(call_record)
        
        # Update count
        self.tool_call_count[tool_name] = self.tool_call_count.get(tool_name, 0) + 1
    
    def add_observation(self, observation: str) -> None:
        """Add observation"""
        self.observations.append({
            "step": self.current_step,
            "observation": observation,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_step(
        self,
        step_name: str,
        status: StepStatus = StepStatus.PENDING
    ) -> None:
        """Add step"""
        step_record = {
            "step_id": len(self.steps),
            "name": step_name,
            "status": status.value,
            "timestamp": datetime.now().isoformat()
        }
        self.steps.append(step_record)
        self.current_step = len(self.steps) - 1
    
    def update_step_status(
        self,
        step_id: int,
        status: StepStatus
    ) -> None:
        """Update step status"""
        if 0 <= step_id < len(self.steps):
            self.steps[step_id]["status"] = status.value
            self.steps[step_id]["updated_at"] = datetime.now().isoformat()
    
    def check_fallback(self) -> bool:
        """
        Check if fallback is needed
        
        Returns:
            Whether fallback is needed
        """
        # Check tool call failure count
        failed_calls = sum(
            1 for call in self.tool_calls
            if call["status"] != "success"
        )
        
        if failed_calls > len(self.tool_calls) * 0.5:  # More than 50% failed
            self.needs_fallback = True
            self.fallback_reason = f"Too many tool call failures: {failed_calls}/{len(self.tool_calls)}"
            return True
        
        # Check if any steps failed
        failed_steps = sum(
            1 for step in self.steps
            if step["status"] == StepStatus.FAILED.value
        )
        
        if failed_steps > 0:
            self.needs_fallback = True
            self.fallback_reason = f"Failed steps: {failed_steps}"
            return True
        
        # Check tool call count limit
        max_calls_per_tool = 5
        for tool_name, count in self.tool_call_count.items():
            if count > max_calls_per_tool:
                self.needs_fallback = True
                self.fallback_reason = f"Tool {tool_name} called too many times: {count}"
                return True
        
        return False
    
    def finish_execution(self, final_answer: str) -> None:
        """Finish execution"""
        self.end_time = datetime.now()
        self.final_answer = final_answer
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            "query": self.query,
            "duration_seconds": duration,
            "total_tool_calls": len(self.tool_calls),
            "tool_call_breakdown": self.tool_call_count,
            "total_steps": len(self.steps),
            "total_thoughts": len(self.thoughts),
            "needs_fallback": self.needs_fallback,
            "fallback_reason": self.fallback_reason,
            "plan_strategy": self.plan.get("strategy") if self.plan else None
        }
    
    def get_execution_trace(self) -> str:
        """Get execution trace (for debugging)"""
        trace = f"=== Execution Trace ===\n"
        trace += f"Query: {self.query}\n"
        trace += f"Start Time: {self.start_time}\n\n"
        
        trace += "Thoughts:\n"
        for thought in self.thoughts:
            trace += f"  [{thought['step']}] {thought['thought']}\n"
        
        trace += "\nTool Calls:\n"
        for call in self.tool_calls:
            trace += f"  [{call['step']}] {call['tool_name']}: {call['status']}\n"
        
        trace += "\nObservations:\n"
        for obs in self.observations:
            trace += f"  [{obs['step']}] {obs['observation'][:100]}...\n"
        
        if self.final_answer:
            trace += f"\nFinal Answer: {self.final_answer[:200]}...\n"
        
        return trace

