

from typing import List
from dataclasses import dataclass
from pydantic import BaseModel, field_validator
from autogen_core.models import LLMMessage


class UserLogin(BaseModel):
    """
    Fired when a new user session begins.
    The username is used as the session identifier.
    """
    username: str

class UserTask(BaseModel):
    """
    Carries the conversation context (list of LLM messages)
    between agents in a handoff scenario.
    """
    context: List[LLMMessage]

class AgentResponse(BaseModel):
    """
    The AI agent's final response to the user, plus the topic
    the user should reply to next.
    """
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
