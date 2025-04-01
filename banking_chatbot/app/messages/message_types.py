

from typing import List
from dataclasses import dataclass
from pydantic import BaseModel, field_validator
from autogen_core.models import LLMMessage


class UserLogin(BaseModel):
    
    username: str

class UserTask(BaseModel):
    
    context: List[LLMMessage]

class AgentResponse(BaseModel):
   
    reply_to_topic_type: str
    context: List[LLMMessage]


class UserCredentials(BaseModel):
    username: str
    password: str

@dataclass
class MyMessageType:
    content: str

class DomainClassifierOutput(BaseModel):
    agent_name: str

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, value):
        allowed_agents = {
            "RetailBankingAgent",
            "CorporateBusinessBankingAgent",
            "InvestmentBankingAgent",
            "WealthManagementAgent",
            "RiskManagementAgent",
            "InsuranceAgent",
            "ITOpsAgent",
            "PaymentsAgent",
            "CapitalTreasuryAgent",
            "AnalyticsAgent",
        }
        if value not in allowed_agents:
            raise ValueError(f"Invalid agent name: {value}")
        return value
