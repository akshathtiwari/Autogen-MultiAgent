

import json
from autogen_core import message_handler, MessageContext
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
    """
    A specialized PaymentsAgent that handles payment discrepancies:
      - Ask user for transaction ID
      - Look up PaymentStatus & CoreBankingStatus
      - If mismatch, ask user to fix; if yes, call fix_core_banking_status
    """

    def __init__(self, system_message: SystemMessage, model_client: ChatCompletionClient):
        super().__init__(
            agent_type="PaymentsAgent",
            system_message=system_message,
            model_client=model_client,
            tools=[lookup_transaction_tool, fix_core_banking_status_tool],
            delegate_tools=[],
            my_topic_type="Payments",
            user_topic_type="User",
        )

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        print(f"[Payments] Handling user task with {len(message.context)} messages.")

       
        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            
            tools=[lookup_transaction_tool.schema, fix_core_banking_status_tool.schema],
            cancellation_token=ctx.cancellation_token,
        )
        print(f"\n*** Payments LLM response ***\n{llm_result.content}")

        
        transaction_id = input("\nEnter transaction ID: ")

        
        message.context.append(UserMessage(content=f"TransactionID: {transaction_id}", source="User"))

        
        lookup_result_python = await lookup_transaction_tool.run_json(
            {"transaction_id": transaction_id},
            ctx.cancellation_token
        )
        
        lookup_result_str = json.dumps(lookup_result_python, ensure_ascii=False)
        print(f"[Payments] Lookup result: {lookup_result_str}")

        
        try:
            tx_info = json.loads(lookup_result_str)
        except Exception as e:
            tx_info = {}
            print(f"Error parsing lookup result: {e}")

        
        if (
            tx_info.get("PaymentStatus") == "Success"
            and tx_info.get("CoreBankingStatus") != "Success"
        ):
            confirmation = input(
                "Payment gateway is success, but not reflected in core banking. Fix? (yes/no): "
            )
            if confirmation.lower().startswith("y"):
                fix_result_python = await fix_core_banking_status_tool.run_json(
                    {"transaction_id": transaction_id},
                    ctx.cancellation_token
                )
                fix_result_str = json.dumps(fix_result_python, ensure_ascii=False)
                response_text = f"Transaction {transaction_id} updated: {fix_result_str}"
            else:
                response_text = f"No changes made for transaction {transaction_id}."
        else:
            response_text = f"No discrepancy detected for transaction {transaction_id}."

        
        message.context.append(AssistantMessage(content=response_text, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
            topic_id=self._user_topic_type
        )
