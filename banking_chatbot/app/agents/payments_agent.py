import json
from autogen_core import message_handler, MessageContext, TopicId
from autogen_core.models import (
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ChatCompletionClient,
)
from app.messages.message_types import UserTask, AgentResponse
from app.agents.base_agent import BankingAIAgent
from app.tools.transaction_tools import lookup_transaction_tool, fix_core_banking_status_tool

class PaymentsAgent(BankingAIAgent):
    def __init__(self, system_message: SystemMessage, model_client: ChatCompletionClient, conversation_state_accessor):
        super().__init__(
            agent_type="PaymentsAgent",
            system_message=system_message,
            model_client=model_client,
            tools=[lookup_transaction_tool, fix_core_banking_status_tool],
            delegate_tools=[],
            my_topic_type="Payments",
            user_topic_type="User",
        )
        self._conversation_accessor = conversation_state_accessor

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        session_id = ctx.topic_id.source

        
        if not any(isinstance(m, UserMessage) and m.content.strip().startswith("TX") for m in message.context):
            prompt = "I can help you with the payment mismatch. Please provide your transaction ID (e.g., TX1001)."
            message.context.append(AssistantMessage(content=prompt, source=self.id.type))
            await self.publish_message(
                AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
                topic_id=TopicId(self._user_topic_type, session_id),
            )
            return

        
        for m in message.context:
            if isinstance(m, UserMessage) and m.content.strip().startswith("TX"):
                transaction_id = m.content.strip()
                break

        
        lookup_result = await lookup_transaction_tool.run_json({"transaction_id": transaction_id}, ctx.cancellation_token)
        lookup_result_str = json.dumps(lookup_result, ensure_ascii=False)
        try:
            tx_info = json.loads(lookup_result_str)
        except Exception:
            tx_info = {}

       
        if tx_info.get("PaymentStatus") == "Success" and tx_info.get("CoreBankingStatus") != "Success":
            fix_result = await fix_core_banking_status_tool.run_json({"transaction_id": transaction_id}, ctx.cancellation_token)
            fix_result_str = json.dumps(fix_result, ensure_ascii=False)
            response_text = f"Transaction {transaction_id} updated: {fix_result_str}"

            
            resolved_text = "Your payment mismatch has been resolved successfully! Are you satisfied with the resolution? (yes/no)"
            message.context.append(AssistantMessage(content=resolved_text, source=self.id.type))

            
            self._conversation_accessor.set_status(session_id, "post_action")
            self._conversation_accessor.set_last_agent(session_id, "PaymentsAgent")
        else:
            
            response_text = f"No discrepancy detected for transaction {transaction_id}."

        message.context.append(AssistantMessage(content=response_text, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
            topic_id=TopicId(self._user_topic_type, session_id),
        )
