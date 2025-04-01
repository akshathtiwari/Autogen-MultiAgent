from autogen_core import TypeSubscription
from autogen_core.models import SystemMessage, ChatCompletionClient
from app.agents.base_agent import BankingAIAgent
from app.agents.payments_agent import PaymentsAgent


async def register_retail_banking_agent(runtime, model_client: ChatCompletionClient):
    agent_type = await BankingAIAgent.register(
        runtime,
        type="RetailBanking",
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


async def register_payments_agent(runtime, model_client: ChatCompletionClient, conversation_state_accessor):
    agent_type = await PaymentsAgent.register(
        runtime,
        type="Payments",
        factory=lambda: PaymentsAgent(
            system_message=SystemMessage(
                content="""
You are a Payments & Settlement Systems expert.

When the user reports a payment discrepancy, such as:
- "payment mismatch"
- "payment not reflecting in core banking"
- "gateway success but not in the bank system"
Then:
1. Ask the user for transaction ID if you don't have it.
2. Use the 'lookup_transaction' tool to retrieve PaymentStatus and CoreBankingStatus.
3. If PaymentStatus=Success AND CoreBankingStatus != Success,
   confirm with the user if they'd like to fix the mismatch.
4. If user confirms, call 'fix_core_banking_status' for that transaction ID,
   and let them know the mismatch is fixed.

Remember:
- Always confirm you have the TransactionID before you attempt a fix.
- Only fix if PaymentStatus=Success and CoreBankingStatus != Success.
- If the user uses words like "mismatch", "not reflecting", or "discrepancy", you should address it.
- If a transaction isn't found or doesn't match these conditions, let the user know.
"""
            ),
            model_client=model_client,
            conversation_state_accessor=conversation_state_accessor
        )
    )
    await runtime.add_subscription(TypeSubscription(topic_type="Payments", agent_type=agent_type.type))


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
