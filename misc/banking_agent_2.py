import asyncio
import json
import uuid
from typing import List, Tuple

from dataclasses import dataclass

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, field_validator

#
# -------------------- MESSAGE TYPES --------------------
#

class UserLogin(BaseModel):
    """Fired when a new user session begins."""
    pass

class UserTask(BaseModel):
    """Carries the conversation context (list of LLM messages) between agents."""
    context: List[LLMMessage]

class AgentResponse(BaseModel):
    """The AI agent's final response to the user, plus the topic the user should reply to."""
    reply_to_topic_type: str
    context: List[LLMMessage]

#
# -------------------- DOMAIN CLASSIFIER MODEL --------------------
#

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


#
# -------------------- DELEGATE TOOLS (mapping agent name -> topic type) --------------------
#
# Each tool returns the name of the topic type that corresponds to a domain agent.

def transfer_to_retail_banking() -> str:
    return "RetailBanking"

def transfer_to_corporate_banking() -> str:
    return "CorporateBanking"

def transfer_to_investment_banking() -> str:
    return "InvestmentBanking"

def transfer_to_wealth_management() -> str:
    return "WealthManagement"

def transfer_to_risk_management() -> str:
    return "RiskManagement"

def transfer_to_insurance() -> str:
    return "Insurance"

def transfer_to_it_ops() -> str:
    return "ITOps"

def transfer_to_payments() -> str:
    return "Payments"

def transfer_to_capital_treasury() -> str:
    return "CapitalTreasury"

def transfer_to_analytics() -> str:
    return "Analytics"


transfer_to_retail_banking_tool = FunctionTool(
    transfer_to_retail_banking,
    description="Call this to route user to RetailBankingAgent if LLM decides it's a Retail Banking issue."
)
transfer_to_corporate_banking_tool = FunctionTool(
    transfer_to_corporate_banking,
    description="Call this to route user to CorporateBusinessBankingAgent if LLM decides it's a corporate banking issue."
)
transfer_to_investment_banking_tool = FunctionTool(
    transfer_to_investment_banking,
    description="Call this to route user to InvestmentBankingAgent."
)
transfer_to_wealth_management_tool = FunctionTool(
    transfer_to_wealth_management,
    description="Call this to route user to WealthManagementAgent."
)
transfer_to_risk_management_tool = FunctionTool(
    transfer_to_risk_management,
    description="Call this to route user to RiskManagementAgent."
)
transfer_to_insurance_tool = FunctionTool(
    transfer_to_insurance,
    description="Call this to route user to InsuranceAgent."
)
transfer_to_it_ops_tool = FunctionTool(
    transfer_to_it_ops,
    description="Call this to route user to ITOpsAgent."
)
transfer_to_payments_tool = FunctionTool(
    transfer_to_payments,
    description="Call this to route user to PaymentsAgent."
)
transfer_to_capital_treasury_tool = FunctionTool(
    transfer_to_capital_treasury,
    description="Call this to route user to CapitalTreasuryAgent."
)
transfer_to_analytics_tool = FunctionTool(
    transfer_to_analytics,
    description="Call this to route user to AnalyticsAgent."
)

#
# -------------------- BANKING AI AGENT BASE CLASS --------------------
#
class BankingAIAgent(RoutedAgent):
    def __init__(
        self,
        agent_type: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[Tool],
        delegate_tools: List[Tool],
        my_topic_type: str,
        user_topic_type: str,
    ) -> None:
        super().__init__(agent_type)
        self._system_message = system_message
        self._model_client = model_client
        self._tools = {tool.name: tool for tool in tools}
        self._tool_schema = [tool.schema for tool in tools]
        self._delegate_tools = {tool.name: tool for tool in delegate_tools}
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._my_topic_type = my_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        # 1. Call the LLM with current context
        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        print(f"\n*** {self.id.type} initial LLM response ***\n{llm_result.content}", flush=True)

        # 2. If the LLM response is a list of function calls, handle them (like the example).
        while isinstance(llm_result.content, list) and all(isinstance(m, FunctionCall) for m in llm_result.content):
            tool_call_results: List[FunctionExecutionResult] = []

            for call in llm_result.content:
                arguments = json.loads(call.arguments)
                if call.name in self._tools:
                    # Direct tool call
                    result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token)
                    tool_call_results.append(
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=self._tools[call.name].return_value_as_string(result),
                            is_error=False,
                            name=call.name
                        )
                    )
                elif call.name in self._delegate_tools:
                    raise ValueError("Domain agent does not delegate further in this example.")
                else:
                    raise ValueError(f"Unknown tool called: {call.name}")

            # Re-call the LLM with function execution results
            message.context.extend([
                AssistantMessage(content=llm_result.content, source=self.id.type),
                FunctionExecutionResultMessage(content=tool_call_results),
            ])
            llm_result = await self._model_client.create(
                messages=[self._system_message] + message.context,
                tools=self._tool_schema + self._delegate_tool_schema,
                cancellation_token=ctx.cancellation_token,
            )
            print(f"\n*** {self.id.type} subsequent LLM response ***\n{llm_result.content}", flush=True)

        # 3. Publish the final response
        assert isinstance(llm_result.content, str)
        message.context.append(AssistantMessage(content=llm_result.content, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )


#
# -------------------- DOMAIN CLASSIFIER AGENT (TRIAGE) --------------------
#
class DomainClassifierAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        delegate_tools: List[Tool],
        my_topic_type: str,
        user_topic_type: str
    ) -> None:
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._delegate_tools = {tool.name: tool for tool in delegate_tools}
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._my_topic_type = my_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        # Gather user text from the last user message
        user_content = ""
        for m in message.context:
            if isinstance(m, UserMessage):
                user_content += m.content + "\n"

        classification_prompt = (
            "You are an expert banking domain classifier. Your task is to determine which "
            "banking domain is most relevant for the user's query below. The available domains "
            "and their corresponding agent names are:\n\n"
            "1. RetailBankingAgent\n"
            "2. CorporateBusinessBankingAgent\n"
            "3. InvestmentBankingAgent\n"
            "4. WealthManagementAgent\n"
            "5. RiskManagementAgent\n"
            "6. InsuranceAgent\n"
            "7. ITOpsAgent\n"
            "8. PaymentsAgent\n"
            "9. CapitalTreasuryAgent\n"
            "10. AnalyticsAgent\n\n"
            f"User query: {user_content}\n\n"
            "You MUST respond with valid JSON containing exactly one key 'agent_name'. "
            "That 'agent_name' MUST be exactly one of these strings: \n"
            "[\"RetailBankingAgent\", \"CorporateBusinessBankingAgent\", \"InvestmentBankingAgent\", "
            "\"WealthManagementAgent\", \"RiskManagementAgent\", \"InsuranceAgent\", \"ITOpsAgent\", "
            "\"PaymentsAgent\", \"CapitalTreasuryAgent\", \"AnalyticsAgent\"]\n\n"
            "Example valid response: {\"agent_name\": \"RetailBankingAgent\"}"
        )
        llm_result = await self._model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=classification_prompt, source="System"),
            ],
            tools=self._delegate_tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        print(f"\n*** DomainClassifierAgent LLM response ***\n{llm_result.content}", flush=True)

        # Parse the LLM response as JSON. Default to RetailBankingAgent if invalid
        llm_output = llm_result.content.strip()
        agent_name = "RetailBankingAgent"  # fallback
        try:
            json_start = llm_output.find("{")
            json_end = llm_output.rfind("}") + 1
            json_str = llm_output[json_start:json_end]
            classifier_output = DomainClassifierOutput.model_validate_json(json_str)
            agent_name = classifier_output.agent_name
        except Exception as e:
            print(f"Classification parse error: {e}; defaulting to RetailBankingAgent")

        print(f"Classified domain as: {agent_name}")

        # Map the agent_name -> domain topic
        domain_map = {
            "RetailBankingAgent": "RetailBanking",
            "CorporateBusinessBankingAgent": "CorporateBanking",
            "InvestmentBankingAgent": "InvestmentBanking",
            "WealthManagementAgent": "WealthManagement",
            "RiskManagementAgent": "RiskManagement",
            "InsuranceAgent": "Insurance",
            "ITOpsAgent": "ITOps",
            "PaymentsAgent": "Payments",
            "CapitalTreasuryAgent": "CapitalTreasury",
            "AnalyticsAgent": "Analytics",
        }
        tool_name = {
            "RetailBankingAgent": "transfer_to_retail_banking",
            "CorporateBusinessBankingAgent": "transfer_to_corporate_banking",
            "InvestmentBankingAgent": "transfer_to_investment_banking",
            "WealthManagementAgent": "transfer_to_wealth_management",
            "RiskManagementAgent": "transfer_to_risk_management",
            "InsuranceAgent": "transfer_to_insurance",
            "ITOpsAgent": "transfer_to_it_ops",
            "PaymentsAgent": "transfer_to_payments",
            "CapitalTreasuryAgent": "transfer_to_capital_treasury",
            "AnalyticsAgent": "transfer_to_analytics",
        }.get(agent_name, "transfer_to_retail_banking")

        if tool_name not in self._delegate_tools:
            print(f"Unknown agent_name = {agent_name}, defaulting to RetailBanking.")
            tool_name = "transfer_to_retail_banking"

        # "Call" that function
        tool = self._delegate_tools[tool_name]
        result = await tool.run_json({}, ctx.cancellation_token)
        target_topic = tool.return_value_as_string(result)
        print(f"Delegating to topic: {target_topic}", flush=True)

        # Publish a new UserTask to the domain topic
        delegate_messages = list(message.context) + [
            AssistantMessage(
                content=[FunctionCall(id="auto_handoff", name=tool.name, arguments="{}")],
                source=self.id.type
            ),
            FunctionExecutionResultMessage(
                content=[
                    FunctionExecutionResult(
                        call_id="auto_handoff",
                        content=f"Transferred to {target_topic}. Adopt persona immediately.",
                        is_error=False,
                        name=tool.name,
                    )
                ]
            ),
        ]
        await self.publish_message(UserTask(context=delegate_messages), topic_id=TopicId(target_topic, source=self.id.key))


#
# -------------------- USER AGENT --------------------
#
class UserAgent(RoutedAgent):
    def __init__(self, description: str, user_topic_type: str, classifier_topic: str) -> None:
        super().__init__(description)
        self._user_topic_type = user_topic_type
        self._classifier_topic = classifier_topic

    @message_handler
    async def handle_user_login(self, message: UserLogin, ctx: MessageContext) -> None:
        """When a new user session starts, greet them and ask for their banking query."""
        session_id = self.id.key
        print(f"\n=== Welcome to the Banking Chatbot! (session ID: {session_id}) ===")
        user_name = input("May I know your name?: ")
        print(f"Hello, {user_name}!")
        user_query = input("Please describe your banking issue or question: ")

        initial_messages = [UserMessage(content=user_query, source="User")]

        await self.publish_message(
            UserTask(context=initial_messages),
            topic_id=TopicId(self._classifier_topic, source=session_id),
        )

    @message_handler
    async def handle_agent_response(self, message: AgentResponse, ctx: MessageContext) -> None:
        """When any domain agent publishes a final response, show it to the user and let them keep chatting or exit."""
        # Print the last assistant message
        last_assistant_msg = ""
        for m in reversed(message.context):
            if isinstance(m, AssistantMessage):
                if isinstance(m.content, str):
                    last_assistant_msg = m.content
                else:
                    last_assistant_msg = str(m.content)
                break
        print(f"\nAgent says: {last_assistant_msg}")

        # Let user respond or exit
        user_input = input("User (type 'exit' to close): ")
        if user_input.strip().lower() == "exit":
            print(f"\nThank you for using the Banking Chatbot. Session {self.id.key} ended.")
            return

        message.context.append(UserMessage(content=user_input, source="User"))
        await self.publish_message(
            UserTask(context=message.context),
            topic_id=TopicId(message.reply_to_topic_type, source=self.id.key),
        )

#
# -------------------- MAIN SETUP --------------------
#

async def main():
    runtime = SingleThreadedAgentRuntime()

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=None,  # or your real key
    )

    # 1. Register UserAgent on topic="User"
    user_agent_type = await UserAgent.register(
        runtime,
        type="User",
        factory=lambda: UserAgent(
            description="A user agent interface for the Banking chatbot",
            user_topic_type="User",
            classifier_topic="DomainClassifier",
        )
    )
    await runtime.add_subscription(TypeSubscription(topic_type="User", agent_type=user_agent_type.type))

    # 2. Register DomainClassifierAgent on topic="DomainClassifier"
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
            description="A domain classifier agent for banking queries",
            system_message=SystemMessage(content="You are the triage agent for banking queries."),
            model_client=model_client,
            delegate_tools=delegate_tools,
            my_topic_type="DomainClassifier",
            user_topic_type="User",
        )
    )
    await runtime.add_subscription(TypeSubscription(topic_type="DomainClassifier", agent_type=classifier_agent_type.type))

    # 3. Helper to register domain agents
    async def register_domain_agent(
        topic_type: str,
        agent_type_name: str,
        system_prompt: str,
    ):
        agent_type = await BankingAIAgent.register(
            runtime,
            type=topic_type,
            factory=lambda: BankingAIAgent(
                agent_type=agent_type_name,
                system_message=SystemMessage(content=system_prompt),
                model_client=model_client,
                tools=[],         # add any domain-specific tools if needed
                delegate_tools=[],# domain agents typically won't delegate
                my_topic_type=topic_type,
                user_topic_type="User",
            )
        )
        await runtime.add_subscription(TypeSubscription(topic_type=topic_type, agent_type=agent_type.type))

    # Register each domain topic and agent
    await register_domain_agent(
        "RetailBanking",
        "RetailBankingAgent",
        "You are a Retail Banking expert. Provide assistance on retail bank accounts, personal loans, etc."
    )
    await register_domain_agent(
        "CorporateBanking",
        "CorporateBusinessBankingAgent",
        "You are a Corporate & Business Banking expert..."
    )
    await register_domain_agent(
        "InvestmentBanking",
        "InvestmentBankingAgent",
        "You are an Investment Banking expert..."
    )
    await register_domain_agent(
        "WealthManagement",
        "WealthManagementAgent",
        "You are a Wealth Management & Private Banking expert..."
    )
    await register_domain_agent(
        "RiskManagement",
        "RiskManagementAgent",
        "You are a Risk Management & Compliance expert..."
    )
    await register_domain_agent(
        "Insurance",
        "InsuranceAgent",
        "You are an Insurance & Bancassurance expert..."
    )
    await register_domain_agent(
        "ITOps",
        "ITOpsAgent",
        "You are an IT-Ops expert..."
    )
    await register_domain_agent(
        "Payments",
        "PaymentsAgent",
        "You are a Payments & Settlement Systems expert..."
    )
    await register_domain_agent(
        "CapitalTreasury",
        "CapitalTreasuryAgent",
        "You are a Capital & Treasury Operations expert..."
    )
    await register_domain_agent(
        "Analytics",
        "AnalyticsAgent",
        "You are an Analytics & Business Intelligence expert..."
    )

    # Start the runtime
    runtime.start()

    # Simulate a user session: publish a UserLogin event
    session_id = str(uuid.uuid4())
    await runtime.publish_message(UserLogin(), topic_id=TopicId("User", source=session_id))

    # Run until idle
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
