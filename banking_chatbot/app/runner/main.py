# app/runner/main.py

import uuid
import asyncio
from autogen_core import SingleThreadedAgentRuntime, TopicId, TypeSubscription
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import SystemMessage

# Message types
from app.messages.message_types import UserLogin

# Agents
from app.agents.user_agent import UserAgent
from app.agents.domain_classifier_agent import DomainClassifierAgent

import os 
from dotenv import load_dotenv

load_dotenv()

# Domain registration helpers
from app.agents.domain_agents import (
    register_retail_banking_agent,
    register_corporate_banking_agent,
    register_investment_banking_agent,
    register_wealth_management_agent,
    register_risk_management_agent,
    register_insurance_agent,
    register_it_ops_agent,
    register_payments_agent,
    register_capital_treasury_agent,
    register_analytics_agent,
)



# Delegate tools used by DomainClassifierAgent
from app.tools.delegate_tools import (
    transfer_to_retail_banking_tool,
    transfer_to_corporate_banking_tool,
    transfer_to_investment_banking_tool,
    transfer_to_wealth_management_tool,
    transfer_to_risk_management_tool,
    transfer_to_insurance_tool,
    transfer_to_it_ops_tool,
    transfer_to_payments_tool,
    transfer_to_capital_treasury_tool,
    transfer_to_analytics_tool,
)


async def main():
    # 1) Create a single-threaded runtime
    runtime = SingleThreadedAgentRuntime()

    # 2) Create a chat completion client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",  
        api_key=None,         # Replace with your real API key or environment variable if needed
    )

    # 3) Register the UserAgent on topic="User"
    user_agent_type = await UserAgent.register(
        runtime,
        type="User",  # the "User" topic
        factory=lambda: UserAgent(
            description="UserAgent for the Banking Chatbot",
            user_topic_type="User",
            classifier_topic="DomainClassifier",
        ),
    )
    # Subscribe the newly-registered agent to the "User" topic
    await runtime.add_subscription(TypeSubscription(topic_type="User", agent_type=user_agent_type.type))

    # 4) Register the DomainClassifierAgent on topic="DomainClassifier"
    delegate_tools = [
        transfer_to_retail_banking_tool,
        transfer_to_corporate_banking_tool,
        transfer_to_investment_banking_tool,
        transfer_to_wealth_management_tool,
        transfer_to_risk_management_tool,
        transfer_to_insurance_tool,
        transfer_to_it_ops_tool,
        transfer_to_payments_tool,
        transfer_to_capital_treasury_tool,
        transfer_to_analytics_tool,
    ]
    classifier_agent_type = await DomainClassifierAgent.register(
        runtime,
        type="DomainClassifier",
        factory=lambda: DomainClassifierAgent(
            description="DomainClassifierAgent",
            system_message=SystemMessage(content="You are the triage agent for banking queries."),
            model_client=model_client,
            delegate_tools=delegate_tools,
            my_topic_type="DomainClassifier",
            user_topic_type="User",
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type="DomainClassifier", agent_type=classifier_agent_type.type))

    # 5) Register all the domain agents using the helper functions from domain_agents.py
    await register_retail_banking_agent(runtime, model_client)
    await register_corporate_banking_agent(runtime, model_client)
    await register_investment_banking_agent(runtime, model_client)
    await register_wealth_management_agent(runtime, model_client)
    await register_risk_management_agent(runtime, model_client)
    await register_insurance_agent(runtime, model_client)
    await register_it_ops_agent(runtime, model_client)
    await register_payments_agent(runtime, model_client)
    await register_capital_treasury_agent(runtime, model_client)
    await register_analytics_agent(runtime, model_client)

    # 6) Start processing messages
    runtime.start()

    # 7) Simulate a user session: publish a UserLogin message to the "User" topic
    session_id = str(uuid.uuid4())
    await runtime.publish_message(
        UserLogin(),
        topic_id=TopicId("User", source=session_id),
    )

    # 8) Stop when the runtime is idle (no pending messages)
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
