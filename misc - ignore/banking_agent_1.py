import asyncio
import json
from dataclasses import dataclass
from typing import Set
from pydantic import BaseModel, field_validator

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    message_handler,
    SingleThreadedAgentRuntime,
)
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
            "Respond with a valid JSON object that contains exactly one key 'agent_name' whose "
            "value is one of the agent names listed above. For example: {\"agent_name\": \"RetailBankingAgent\"}"
        )

        response = await self.model_client.create(
            messages=[SystemMessage(content=prompt)],
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


#
# Domain-Specific Agents
#

class RetailBankingAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("RetailBankingAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are a Retail Banking expert. Process the customer's query "
                    "regarding retail banking services such as account details, "
                    "transactions, or branch services and provide an appropriate response"
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        ## TODO: Retrieve vector embedding data against the user message and provide context to LLM
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class CorporateBusinessBankingAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("CorporateBusinessBankingAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are an expert in Corporate & Business Banking. Handle queries "
                    "related to business accounts, loans, and corporate financial services "
                    "while providing clear and concise advice."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class InvestmentBankingAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("InvestmentBankingAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are an Investment Banking expert. Address queries involving "
                    "securities underwriting, M&A, capital raising, and other "
                    "investment-related services."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class WealthManagementAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("WealthManagementAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are a Wealth Management expert. Handle queries related to private "
                    "banking, portfolio management, and personalized financial advice."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class RiskManagementAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("RiskManagementAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are a Risk Management & Compliance expert. Handle queries about "
                    "regulatory compliance, operational risk, credit risk, and overall "
                    "risk mitigation strategies."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class InsuranceAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("InsuranceAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are an Insurance & Bancassurance expert. Address questions on policy "
                    "selection, premium payments, claims processing, and related insurance matters."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class ITOpsAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("ITOpsAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are an IT-Ops expert. Handle queries about core banking systems, "
                    "infrastructure, IT security, and technology-related operations in banking."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class PaymentsAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("PaymentsAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are a Payments & Settlement Systems expert. Handle queries relating "
                    "to money transfers, payment gateways, settlement systems, and other "
                    "payment processes."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class CapitalTreasuryAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("CapitalTreasuryAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are a Capital & Treasury Operations expert. Address queries related "
                    "to liquidity management, capital markets, treasury operations, and "
                    "funding strategies."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


class AnalyticsAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("AnalyticsAgent")
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are an Analytics & Business Intelligence expert. Handle queries "
                    "regarding data analysis, reporting, and insights for business "
                    "decision-making."
                )
            )
        ]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_query(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        print(f"[{self.id.type}] Processing query: {message.content}")
        user_message = UserMessage(content=message.content, source="User")
        await self._model_context.add_message(user_message)

        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        print(f"[{self.id.type}] Response: {response}")
        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.metadata["type"])
        )
        return MyMessageType(content=response.content)


#
# Main to register and test
#
async def main():
    runtime = SingleThreadedAgentRuntime()

    # Register the DomainClassifierAgent
    await DomainClassifierAgent.register(
        runtime, "DomainClassifierAgent",
        lambda: DomainClassifierAgent(runtime)
    )

    # Now register all domain-specific agents with an OpenAIChatCompletionClient
    await RetailBankingAgent.register(
        runtime, "RetailBankingAgent",
        lambda: RetailBankingAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await CorporateBusinessBankingAgent.register(
        runtime, "CorporateBusinessBankingAgent",
        lambda: CorporateBusinessBankingAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await InvestmentBankingAgent.register(
        runtime, "InvestmentBankingAgent",
        lambda: InvestmentBankingAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await WealthManagementAgent.register(
        runtime, "WealthManagementAgent",
        lambda: WealthManagementAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await RiskManagementAgent.register(
        runtime, "RiskManagementAgent",
        lambda: RiskManagementAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await InsuranceAgent.register(
        runtime, "InsuranceAgent",
        lambda: InsuranceAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await ITOpsAgent.register(
        runtime, "ITOpsAgent",
        lambda: ITOpsAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await PaymentsAgent.register(
        runtime, "PaymentsAgent",
        lambda: PaymentsAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await CapitalTreasuryAgent.register(
        runtime, "CapitalTreasuryAgent",
        lambda: CapitalTreasuryAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )
    await AnalyticsAgent.register(
        runtime, "AnalyticsAgent",
        lambda: AnalyticsAgent(OpenAIChatCompletionClient(model="gpt-4o-mini"))
    )

    runtime.start()

    # Send a test message to the DomainClassifierAgent
    classifier_id = AgentId("DomainClassifierAgent", "default")
    sample_query = MyMessageType(content="How to resolve 404 error faced while making payment")
    await runtime.send_message(sample_query, classifier_id)

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
