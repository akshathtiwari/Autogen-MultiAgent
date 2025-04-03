import json
from autogen_core import RoutedAgent, message_handler, MessageContext, FunctionCall, TopicId
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
from app.messages.message_types import MyMessageType


class DomainClassifierAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        delegate_tools: list[Tool],
        my_topic_type: str,
        user_topic_type: str,
        
        conversation_state_accessor=None,
    ):
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._delegate_tools = {tool.name: tool for tool in delegate_tools}
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._my_topic_type = my_topic_type
        self._user_topic_type = user_topic_type

        
        self._conversation_state_accessor = conversation_state_accessor

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        session_id = ctx.topic_id.source

        
        if self._conversation_state_accessor is not None:
            st = self._conversation_state_accessor.get_status(session_id)
        else:
            st = "fresh"

        
        user_input = ""
        for m in message.context:
            if isinstance(m, UserMessage):
                user_input = m.content.strip().lower()

        
        if st == "post_action":
            if user_input in ["yes", "y"]:
                followup_prompt = "Great! Do you have any additional queries? (yes/no)"
                message.context.append(AssistantMessage(content=followup_prompt, source=self.id.type))
                if self._conversation_state_accessor:
                    self._conversation_state_accessor.set_status(session_id, "ask_additional")
                await self.publish_message(
                    AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
                    topic_id=TopicId(self._user_topic_type, session_id),
                )
                return
            elif user_input in ["no", "n"]:
                followup_prompt = "Would you like a follow-up on the same issue? (yes/no)"
                message.context.append(AssistantMessage(content=followup_prompt, source=self.id.type))
                if self._conversation_state_accessor:
                    self._conversation_state_accessor.set_status(session_id, "ask_followup")
                await self.publish_message(
                    AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
                    topic_id=TopicId(self._user_topic_type, session_id),
                )
                return

        if st == "ask_additional":
            if user_input in ["yes", "y"]:
                
                if self._conversation_state_accessor:
                    self._conversation_state_accessor.reset(session_id)
                    self._conversation_state_accessor.reset_messages(session_id)
    
                
                new_prompt = "Okay, I've cleared the old conversation. Please type your new query now."
                message.context.append(AssistantMessage(content=new_prompt, source=self.id.type))
                await self.publish_message(
                    AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
                    topic_id=TopicId(self._user_topic_type, session_id),
                )
                return
            else:
                end_msg = "Thank you! Have a great day!"
                message.context.append(AssistantMessage(content=end_msg, source=self.id.type))
                if self._conversation_state_accessor:
                    self._conversation_state_accessor.reset(session_id)
                await self.publish_message(
                    AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
                    topic_id=TopicId(self._user_topic_type, session_id),
                )
                return


        if st == "ask_followup":
            if user_input in ["yes", "y"]:
                last_agent = self._conversation_state_accessor.get_last_agent(session_id) if self._conversation_state_accessor else None
                if last_agent:
                    self._conversation_state_accessor.set_status(session_id, "follow_up")
                    self._conversation_state_accessor.set_last_agent(session_id, last_agent)
                    msg_text = "Please describe your follow-up question regarding the same issue."
                    message.context.append(AssistantMessage(content=msg_text, source=self.id.type))
                    await self.publish_message(
                        AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
                        topic_id=TopicId(self._user_topic_type, session_id),
                    )
                    return
            end_msg = "Thank you! Have a wonderful day!"
            message.context.append(AssistantMessage(content=end_msg, source=self.id.type))
            if self._conversation_state_accessor:
                self._conversation_state_accessor.reset(session_id)
            await self.publish_message(
                AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
                topic_id=TopicId(self._user_topic_type, session_id),
            )
            return

        
        user_content = ""
        for m in message.context:
            if isinstance(m, UserMessage):
                user_content += m.content + "\n"

        classification_prompt = (
    "You are an expert banking domain classifier. Your task is to decide which domain is relevant.\n"
    "Possible agent names: RetailBankingAgent, CorporateBusinessBankingAgent, InvestmentBankingAgent,\n"
    "WealthManagementAgent, RiskManagementAgent, InsuranceAgent, ITOpsAgent, PaymentsAgent,\n"
    "CapitalTreasuryAgent, AnalyticsAgent\n\n"
    f"User query: {user_content}\n\n"
    "Respond with JSON: {\"agent_name\": \"OneOfTheAbove\"}\n"
    "If the user wants to do a normal payment or personal transaction, use RetailBankingAgent, "
    "not PaymentsAgent. Only route to PaymentsAgent if the user specifically reports a mismatch "
    "or discrepancy or some failure in the payment process.\n"
    "Examples:\n"
    "User: \"How do I pay John 100?\" => Agent: \"RetailBankingAgent\"\n"
    "User: \"My payment success isn't reflecting in the system\" => Agent: \"PaymentsAgent\""
)


        llm_result = await self._model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=classification_prompt, source="System"),
            ],
            cancellation_token=ctx.cancellation_token,
        )

        print(f"\n*** DomainClassifierAgent LLM response ***\n{llm_result.content}", flush=True)

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

        new_topic = TopicId(target_topic, source=ctx.topic_id.source)
        await self.publish_message(
            UserTask(context=new_context),
            topic_id=new_topic,
        )
