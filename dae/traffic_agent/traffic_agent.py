import os
import json
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# Structured Output Models ensuring forced JSON formats from LLM
class SubAgentReportDefinition(BaseModel):
    priority_rating: str = Field(description="Priority fraction out of 10, e.g., '10/10' or '5/10'")
    utility_score: float = Field(description="Calculated utility score based on weights.")
    reasoning: str = Field(description="Natural language reasoning explaining the priority.")

class MasterDecisionDefinition(BaseModel):
    command: str = Field(description="Either 'SWITCH_PHASE' or 'MAINTAIN_PHASE'")
    target_lane: str = Field(description="The lane that will receive or maintain the green light")
    duration: str = Field(description="Typically 'Dynamic'")
    reason: str = Field(description="Detailed reason for the decision, referring to the sub-agent reports.")

class LaneState:
    """Tracks state that persists across frames for a given lane."""
    def __init__(self, name: str):
        self.name = name
        self.wait_time = 0.0

class LangChainTrafficAgent:
    """
    Agentic System powered by LangChain.
    Features 4 Independent Sub-Agents evaluating lanes in parallel,
    and 1 Master Agent performing conflict resolution.
    """
    def __init__(self, dt_step: float = 1.0):
        # Tracking states that span across frames
        self.lanes_state = {
            "North": LaneState("North"),
            "South": LaneState("South"),
            "East": LaneState("East"),
            "West": LaneState("West")
        }
        self.active_lane = "North"
        self.interrupted_lane = None
        self.time_in_phase = 0.0
        self.dt_step = dt_step
        
        # Configuration Thresholds
        self.min_green_time = 10.0
        self.max_green_time = 90.0 # Force switch if a lane is green for too long (starvation prevention)
        self.max_wait_time = 120.0 # Absolute max wait threshold (if someone has waited 2 mins, they get priority)
        
        # Initialize LLM via native ChatOllama integration
        # Note: adjust the model name to exactly match the tag you used when downloading via ollama
        self.llm = ChatOllama(
            model="kimi-k2.5:cloud", 
            temperature=0
        )
        self.sub_agent_chain = self._build_sub_agent_chain()
        self.master_agent_chain = self._build_master_agent_chain()

    def _build_sub_agent_chain(self):
        prompt = PromptTemplate(
            template="""You are the Intelligence Sub-Agent for the {lane_name} lane of an intersection.
Your role: Evaluate the current traffic density and emergency status to generate a priority report.

Inputs for {lane_name}:
- Car Count: {car_count}
- Emergency Detected: {emergency_detected}
- Wait Time (seconds in RED): {wait_time}

Process:
1. Calculate a raw utility score using this logic: (Density * 2.0) + (Wait time * 0.5) + (500.0 if Emergency else 0).
2. Determine a Priority Rating out of 10 (e.g. 10/10 for emergency, 8/10 for wait time > 60s, 5/10 for moderate traffic, etc.)
3. Produce a concise natural language reasoning explaining your priority.

Output format must exactly match the requested schema.""",
            input_variables=["lane_name", "car_count", "emergency_detected", "wait_time"]
        )
        return prompt | self.llm.with_structured_output(SubAgentReportDefinition)

    def _build_master_agent_chain(self):
        prompt = PromptTemplate(
            template="""You are the Master Traffic Referee Agent. Your role is conflict resolution across an intersection.
You must analyze the inner cognitive reports from 4 Sub-Agents, consider incoming traffic from neighboring intersections, and determine the next signal phase.

Current State:
- Currently Active Signal (Green Light): {active_lane}
- Time in current phase: {time_in_phase} seconds
- Minimum Green Time Buffer: {min_green_time} seconds
- Maximum Green Time Limit: {max_green_time} seconds 
- Previously Interrupted Lane (if any): {interrupted_lane}

Neighboring Intersections Data (Approaching traffic):
{neighbor_data}

Sub-Agent Reports:
{sub_agent_reports}

The Decision Matrix Rulebook:
1. MAX GREEN OVERRIDE (Starvation Prevention): If the Time in current phase >= {max_green_time}, you MUST rotate the light away from the active lane to the highest waiting lane, regardless of density.
2. PRE-EMPT OVERRIDE (Emergency): If any lane reports an emergency and its utility score > 400, you MUST pre-empt the current light immediately (ignore Minimum Green Time) and command SWITCH_PHASE to that lane. 
3. DYNAMIC TAIL-END DETECTION (After Emergency): If an interrupted lane exists, AND the currently active lane's emergency has passed (Emergency Detected = False) AND its density has dropped (car count <= 5), you MUST return the green light to the interrupted lane (SWITCH_PHASE). Do not stay green if the ambulance has crossed.
4. GREEN WAVE COORDINATION: If a neighbor is sending heavy traffic (e.g., > 15 cars) or an emergency vehicle towards a specific lane in this intersection, artificially boost that lane's priority to prepare a "Green Wave".
5. STANDARD AUCTION: If time > Minimum Green Time, grant the green light to the lane with the highest utility score (including neighbor boosts). Maintain current light (MAINTAIN_PHASE) if it has the highest score.

You must output a JSON object containing your commanding decision.""",
            input_variables=["active_lane", "time_in_phase", "min_green_time", "max_green_time", "interrupted_lane", "neighbor_data", "sub_agent_reports"]
        )
        return prompt | self.llm.with_structured_output(MasterDecisionDefinition)

    async def aprocess_frame(self, lane_data: Dict[str, Any], neighbor_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async process the frame to do concurrent LLM calls for speed"""
        self.time_in_phase += self.dt_step
        
        # 1. Update internal wait times and prepare Sub-Agent inputs
        sub_agent_inputs = []
        for name in ["North", "South", "East", "West"]:
            is_active = (name == self.active_lane)
            
            # Increment wait times
            if is_active:
                self.lanes_state[name].wait_time = 0.0
            else:
                self.lanes_state[name].wait_time += self.dt_step
                
            # Prepare payload for LLM flexibly
            data = lane_data.get(name)
            if isinstance(data, dict):
                 c_count = data.get("car_count", 0)
                 e_flag = data.get("emergency_detected", False)
            else:
                 c_count = data.car_count
                 e_flag = data.emergency_detected
                 
            sub_agent_inputs.append({
                "lane_name": name,
                "car_count": c_count,
                "emergency_detected": e_flag,
                "wait_time": self.lanes_state[name].wait_time
            })
            
        # 2. Sub-Agent Reasoning (Parallel Execution using LangChain abatch)
        # Resolves all 4 agent reasonings concurrently to minimize latency
        responses = await self.sub_agent_chain.abatch(sub_agent_inputs)
        
        reports = {}
        for idx, name in enumerate(["North", "South", "East", "West"]):
            response_obj = responses[idx]
            reports[name] = {
                "utility": response_obj.utility_score,
                "reasoning": response_obj.reasoning,
                "priority": response_obj.priority_rating,
                "density": sub_agent_inputs[idx]["car_count"],
                "e_flag": sub_agent_inputs[idx]["emergency_detected"]
            }

        # 3. Master Agent Conflict Resolution
        master_input = {
            "active_lane": self.active_lane,
            "time_in_phase": round(self.time_in_phase, 1),
            "min_green_time": self.min_green_time,
            "max_green_time": self.max_green_time,
            "interrupted_lane": self.interrupted_lane if self.interrupted_lane else "None",
            "neighbor_data": json.dumps(neighbor_data, indent=2) if neighbor_data else "No neighbor data available.",
            "sub_agent_reports": json.dumps(reports, indent=2)
        }
        
        master_decision = await self.master_agent_chain.ainvoke(master_input)
        
        # 4. Actuation Update State based on Master's command
        if master_decision.command == "SWITCH_PHASE" and master_decision.target_lane != self.active_lane:
            # Track interrupted lane if this was an emergency preempt
            decision_reason_lower = master_decision.reason.lower()
            is_preempt = ("pre-empt" in decision_reason_lower or "emergency" in decision_reason_lower)
            
            if is_preempt and self.interrupted_lane is None:
                self.interrupted_lane = self.active_lane # Save state before override
            elif master_decision.target_lane == self.interrupted_lane:
                self.interrupted_lane = None # Cleared tail-end condition
                
            self.active_lane = master_decision.target_lane
            self.time_in_phase = 0.0

        # Output the required Actuation JSON format
        return {
            "command": master_decision.command,
            "target_lane": master_decision.target_lane,
            "duration": master_decision.duration,
            "reason": master_decision.reason,
            "sub_agent_reports": reports
        }
