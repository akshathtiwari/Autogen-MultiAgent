import asyncio
import json
from dataclasses import dataclass
from typing import Set
from pydantic import BaseModel, field_validator
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_core.models import AssistantMessage, ChatCompletionClient, SystemMessage, UserMessage

from autogen_core.model_context import BufferedChatCompletionContext

import os
from dotenv import load_dotenv
from main import model_client

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

@dataclass
class MyMessageType:
    content: str


class DomainClassifierOutput(BaseModel):
    agent_name: str

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, value):
        allowed_agents: Set[str] = {
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


class DomainClassifierAgent(RoutedAgent):
    def __init__(self, runtime) -> None:
        super().__init__("DomainClassifierAgent")        
        self._runtime = runtime
        self.model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=API_KEY)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Received query: {message.content}")        
        
        prompt = (
            "You are an expert banking domain classifier. Your task is to determine which "
            "banking domain is most relevant for the following query. The available domains "
            "and their corresponding agent names are:\n\n"
            "1. Retail Banking Agent: RetailBankingAgent\n"
            "2. Corporate & Business Banking Agent: CorporateBusinessBankingAgent\n"
            "3. Investment Banking Agent: InvestmentBankingAgent\n"
            "4. Wealth Mgmt & Private Banking Agent: WealthManagementAgent\n"
            "5. Risk Management & Compliance Agent: RiskManagementAgent\n"
            "6. Insurance & Bancassurance Agent: InsuranceAgent\n"
            "7. IT-Ops Agent: ITOpsAgent\n"
            "8. Payments & Settlement Systems Agent: PaymentsAgent\n"
            "9. Capital & Treasury Operations Agent: CapitalTreasuryAgent\n"
            "10. Analytics & Business Intelligence Agent: AnalyticsAgent\n\n"
            f"User Query: \"{message.content}\"\n\n"
            "Respond with a valid JSON object that contains exactly one key 'agent_name' whose"
            "value is one of the agent names listed above. For example: {\"agent_name\": \"RetailBankingAgent\"}"
        )        
        
        response = await self.model_client.create(
            messages=[
            SystemMessage(content=prompt)
        ],
        )
        llm_output = response.content.strip()
        print(f"[{self.id.type}] Raw LLM response: {llm_output}")

        
        try:
            json_start = llm_output.find("{")
            json_end = llm_output.rfind("}") + 1
            json_str = llm_output[json_start:json_end]
            classifier_output = DomainClassifierOutput.model_validate_json(json_str)
            classification = classifier_output.agent_name
        except Exception as e:
            print(f"[{self.id.type}] Error parsing structured output: {e}. Defaulting to RetailBankingAgent.")
            classification = "RetailBankingAgent"

        print(f"[{self.id.type}] LLM classified query as: {classification}")
        target = AgentId(classification, "default")
        print(f"[{self.id.type}] Forwarding query to {target.type}")
        await self._runtime.send_message(message, target)


class RetailBankingAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A Retail Banking Agent")
        self._system_messages = [SystemMessage(content="You are a Retail Banking expert. Process the customer's query regarding retail banking services such as account details, transactions, or branch services and provide an appropriate response")]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)
        
    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)
        
        response = await self._model_client.create(self._system_messages + (await self._model_context.get_messages()), cancellation_token=ctx.cancellation_token,)
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.metadata["type"]))
        return MyMessageType(content=response.content)

class CorporateBusinessBankingAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("CorporateBusinessBankingAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "Corporate & Business Banking Response: Your business inquiry has been addressed."
        print(f"[{self.id.type}] Response: {response}")

class InvestmentBankingAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("InvestmentBankingAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "Investment Banking Response: Your investment-related query is under review."
        print(f"[{self.id.type}] Response: {response}")

class WealthManagementAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("WealthManagementAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "Wealth Management Response: Your private banking query has been resolved."
        print(f"[{self.id.type}] Response: {response}")

class RiskManagementAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("RiskManagementAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "Risk Management Response: Your risk and compliance query has been processed."
        print(f"[{self.id.type}] Response: {response}")

class InsuranceAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("InsuranceAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "Insurance Response: Your insurance and bancassurance inquiry is resolved."
        print(f"[{self.id.type}] Response: {response}")

class ITOpsAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("ITOpsAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "IT-Ops Response: Your IT operations query has been handled."
        print(f"[{self.id.type}] Response: {response}")

class PaymentsAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("PaymentsAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "Payments & Settlement Systems Response: Your payment query has been successfully processed."
        print(f"[{self.id.type}] Response: {response}")

class CapitalTreasuryAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("CapitalTreasuryAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "Capital & Treasury Operations Response: Your query regarding capital/treasury operations is handled."
        print(f"[{self.id.type}] Response: {response}")

class AnalyticsAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("AnalyticsAgent")

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] Processing query: {message.content}")
        response = "Analytics & Business Intelligence Response: Your analytics inquiry has been resolved."
        print(f"[{self.id.type}] Response: {response}")

async def main():
    runtime = SingleThreadedAgentRuntime()
    await DomainClassifierAgent.register(runtime, "DomainClassifierAgent", lambda: DomainClassifierAgent(runtime))
    await RetailBankingAgent.register(runtime,"RetailBankingAgent", lambda: RetailBankingAgent(OpenAIChatCompletionClient(model="gpt-4o-mini")))
    await CorporateBusinessBankingAgent.register(runtime, "CorporateBusinessBankingAgent", lambda: CorporateBusinessBankingAgent())
    await InvestmentBankingAgent.register(runtime, "InvestmentBankingAgent", lambda: InvestmentBankingAgent())
    await WealthManagementAgent.register(runtime, "WealthManagementAgent", lambda: WealthManagementAgent())
    await RiskManagementAgent.register(runtime, "RiskManagementAgent", lambda: RiskManagementAgent())
    await InsuranceAgent.register(runtime, "InsuranceAgent", lambda: InsuranceAgent())
    await ITOpsAgent.register(runtime, "ITOpsAgent", lambda: ITOpsAgent())
    await PaymentsAgent.register(runtime, "PaymentsAgent", lambda: PaymentsAgent())
    await CapitalTreasuryAgent.register(runtime, "CapitalTreasuryAgent", lambda: CapitalTreasuryAgent())
    await AnalyticsAgent.register(runtime, "AnalyticsAgent", lambda: AnalyticsAgent())

    runtime.start()  

    
    classifier_id = AgentId("DomainClassifierAgent", "default")
    sample_query = MyMessageType(content="How can I avail loan from bank?")
    await runtime.send_message(sample_query, classifier_id)

    await runtime.stop_when_idle()

asyncio.run(main())
