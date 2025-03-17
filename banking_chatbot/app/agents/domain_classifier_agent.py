# app/agents/domain_classifier_agent.py

import json
from autogen_core import RoutedAgent, message_handler, MessageContext, FunctionCall
from autogen_core.models import (
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
)
from autogen_core.tools import Tool

from app.messages.message_types import UserTask, AgentResponse, DomainClassifierOutput
from app.messages.message_types import MyMessageType  # if needed


class DomainClassifierAgent(RoutedAgent):
    """
    A triage agent that uses an LLM to decide which domain agent
    to forward the user query to, using "delegate tools."
    """
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        delegate_tools: list[Tool],
        my_topic_type: str,
        user_topic_type: str,
    ):
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._delegate_tools = {tool.name: tool for tool in delegate_tools}
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._my_topic_type = my_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        # 1) Gather user message(s)
        user_content = ""
        for m in message.context:
            if isinstance(m, UserMessage):
                user_content += m.content + "\n"

        # 2) Build classification prompt
        classification_prompt = (
            "You are an expert banking domain classifier. Your task is to decide which domain is relevant.\n"
            "Possible agent names: RetailBankingAgent, CorporateBusinessBankingAgent, InvestmentBankingAgent,\n"
            "WealthManagementAgent, RiskManagementAgent, InsuranceAgent, ITOpsAgent, PaymentsAgent,\n"
            "CapitalTreasuryAgent, AnalyticsAgent\n\n"
            f"User query: {user_content}\n\n"
            "Respond with JSON: {\"agent_name\": \"OneOfTheAbove\"}"
        )

        llm_result = await self._model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=classification_prompt, source="System"),
            ],
            tools=self._delegate_tool_schema,  # can call the delegate tools
            cancellation_token=ctx.cancellation_token,
        )

        print(f"\n*** DomainClassifierAgent LLM response ***\n{llm_result.content}", flush=True)

        # 3) Parse JSON or fallback
        agent_name = "RetailBankingAgent"
        try:
            content = llm_result.content.strip()
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]
            cls_output = DomainClassifierOutput.model_validate_json(json_str)
            agent_name = cls_output.agent_name
        except Exception as e:
            print(f"Classification parse error: {e} => defaulting to RetailBankingAgent")

        print(f"Classified domain as: {agent_name}")

        # 4) Decide which delegate tool to call
        agent_to_tool = {
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
        }
        tool_name = agent_to_tool.get(agent_name, "transfer_to_retail_banking")
        if tool_name not in self._delegate_tools:
            print("Delegate tool not found. Defaulting to retail banking.")
            tool_name = "transfer_to_retail_banking"

        tool = self._delegate_tools[tool_name]
        result = await tool.run_json({}, ctx.cancellation_token)
        target_topic = tool.return_value_as_string(result)

        print(f"Forwarding user task to topic: {target_topic}", flush=True)

        # 5) Append a function call + function result message to the user context, then publish
        new_context = list(message.context)
        new_context.append(
            AssistantMessage(
                content=[FunctionCall(id="auto_handoff", name=tool.name, arguments="{}")],
                source=self.id.type,
            )
        )
        new_context.append(
            FunctionExecutionResultMessage(
                content=[
                    FunctionExecutionResult(
                        call_id="auto_handoff",
                        content=f"Transferred to {target_topic} agent. Please adopt persona immediately.",
                        is_error=False,
                        name=tool.name,
                    )
                ]
            )
        )

        await self.publish_message(
            UserTask(context=new_context),
            topic_id=ctx.topic_id._replace(type=target_topic),  # same session, new topic
        )
