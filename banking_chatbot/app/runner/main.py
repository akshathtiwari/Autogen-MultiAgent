# app/runner/main.py

import uuid
import asyncio
from autogen_core import SingleThreadedAgentRuntime, TopicId, TypeSubscription
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import SystemMessage
from app.messages.message_types import UserLogin, UserCredentials

# Agents
from app.agents.authentication_agent import AuthenticationAgent
from app.agents.user_agent import UserAgent
from app.agents.domain_classifier_agent import DomainClassifierAgent

import os
from dotenv import load_dotenv
# from main import model_client

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

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
    runtime = SingleThreadedAgentRuntime()

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        
    )

    # 1) Register the AuthenticationAgent on the "Auth" topic.
    # The AuthenticationAgent will load valid credentials from a CSV file.
    credentials_csv = "C:/Users/akstiwari/OneDrive - Deloitte (O365D)/Desktop/Laptop Files/Desktop Backup/learning/Autogen-MultiAgent/banking_chatbot/app/credentials/users.csv"  # Update the path as needed.
    auth_agent_type = await AuthenticationAgent.register(
        runtime,
        type="Auth",
        factory=lambda: AuthenticationAgent(credentials_csv_path=credentials_csv, user_topic="User")
    )
    await runtime.add_subscription(TypeSubscription(topic_type="Auth", agent_type=auth_agent_type.type))

    # 2) Register the UserAgent on the "User" topic.
    user_agent_type = await UserAgent.register(
        runtime,
        type="User",
        factory=lambda: UserAgent(
            description="UserAgent for the Banking Chatbot",
            user_topic_type="User",
            classifier_topic="DomainClassifier",
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type="User", agent_type=user_agent_type.type))

    # 3) Register the DomainClassifierAgent on the "DomainClassifier" topic.
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

    # 4) Register all domain agents.
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

    # 5) Start the runtime.
    runtime.start()

    # 6) Prompt the user for their credentials.
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    # 7) Publish a UserCredentials message to the "Auth" topic.
    await runtime.publish_message(
        UserCredentials(username=username, password=password),
        topic_id=TopicId("Auth", source=username)
    )

    # 8) Wait until the runtime is idle.
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
