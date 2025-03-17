# app/agents/domain_agents.py

from autogen_core import TypeSubscription
from autogen_core.models import SystemMessage, ChatCompletionClient
from app.agents.base_agent import BankingAIAgent

#
# Each function registers a BankingAIAgent under a specific topic
# ("RetailBanking", "CorporateBanking", etc.).
#
# The agent_type string is a descriptive name (like "RetailBankingAgent"),
# while the topic_type is how the runtime routes messages (like "RetailBanking").
# 
# The system_prompt is a short instruction telling this agent what domain
# expertise it has. In a real application, you'd expand the prompt accordingly.
#

async def register_retail_banking_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="RetailBanking",  # topic type
        factory=lambda: BankingAIAgent(
            agent_type="RetailBankingAgent",
            system_message=SystemMessage(
                content="You are a Retail Banking expert. Handle queries about personal accounts, loans, etc."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="RetailBanking",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="RetailBanking", agent_type=agent_type.type)
    )


async def register_corporate_banking_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="CorporateBanking",
        factory=lambda: BankingAIAgent(
            agent_type="CorporateBusinessBankingAgent",
            system_message=SystemMessage(
                content="You are a Corporate & Business Banking expert."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="CorporateBanking",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="CorporateBanking", agent_type=agent_type.type)
    )


async def register_investment_banking_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="InvestmentBanking",
        factory=lambda: BankingAIAgent(
            agent_type="InvestmentBankingAgent",
            system_message=SystemMessage(
                content="You are an Investment Banking expert. Handle queries on M&A, capital raising, underwriting, etc."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="InvestmentBanking",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="InvestmentBanking", agent_type=agent_type.type)
    )


async def register_wealth_management_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="WealthManagement",
        factory=lambda: BankingAIAgent(
            agent_type="WealthManagementAgent",
            system_message=SystemMessage(
                content="You are a Wealth Management & Private Banking expert. Handle high-net-worth client queries."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="WealthManagement",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="WealthManagement", agent_type=agent_type.type)
    )


async def register_risk_management_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="RiskManagement",
        factory=lambda: BankingAIAgent(
            agent_type="RiskManagementAgent",
            system_message=SystemMessage(
                content="You are a Risk Management & Compliance expert. Address regulatory, credit, and operational risks."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="RiskManagement",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="RiskManagement", agent_type=agent_type.type)
    )


async def register_insurance_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="Insurance",
        factory=lambda: BankingAIAgent(
            agent_type="InsuranceAgent",
            system_message=SystemMessage(
                content="You are an Insurance & Bancassurance expert. Handle policy selection, claims, and related queries."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="Insurance",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="Insurance", agent_type=agent_type.type)
    )


async def register_it_ops_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="ITOps",
        factory=lambda: BankingAIAgent(
            agent_type="ITOpsAgent",
            system_message=SystemMessage(
                content="You are an IT-Ops expert. Handle queries about banking infrastructure, IT systems, security, etc."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="ITOps",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="ITOps", agent_type=agent_type.type)
    )


async def register_payments_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="Payments",
        factory=lambda: BankingAIAgent(
            agent_type="PaymentsAgent",
            system_message=SystemMessage(
                content="You are a Payments & Settlement Systems expert. Handle money transfer, gateway, or settlement queries."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="Payments",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="Payments", agent_type=agent_type.type)
    )


async def register_capital_treasury_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="CapitalTreasury",
        factory=lambda: BankingAIAgent(
            agent_type="CapitalTreasuryAgent",
            system_message=SystemMessage(
                content="You are a Capital & Treasury Operations expert. Handle liquidity management, capital markets, etc."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="CapitalTreasury",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="CapitalTreasury", agent_type=agent_type.type)
    )


async def register_analytics_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="Analytics",
        factory=lambda: BankingAIAgent(
            agent_type="AnalyticsAgent",
            system_message=SystemMessage(
                content="You are an Analytics & Business Intelligence expert. Handle data analysis and reporting queries."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[],
            my_topic_type="Analytics",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="Analytics", agent_type=agent_type.type)
    )
