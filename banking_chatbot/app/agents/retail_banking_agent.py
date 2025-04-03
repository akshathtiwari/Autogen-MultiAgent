import json
from autogen_core import message_handler, MessageContext, FunctionCall, TopicId
from autogen_core.models import (
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ChatCompletionClient
)
from autogen_core.tools import Tool
from app.messages.message_types import UserTask, AgentResponse
from app.agents.base_agent import BankingAIAgent
from app.tools.transaction_tools import check_balance_tool, make_payment_tool


class RetailBankingAgent(BankingAIAgent):
    def __init__(self, agent_type: str, system_message: SystemMessage, model_client: ChatCompletionClient):
        super().__init__(
            agent_type=agent_type,
            system_message=system_message,
            model_client=model_client,
            tools=[],  
            delegate_tools=[check_balance_tool, make_payment_tool],  
            my_topic_type="RetailBanking",
            user_topic_type="User",
        )

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        print(f"[RetailBankingAgent] handle_task triggered with user content: "
              f"{[m.content for m in message.context if hasattr(m, 'content')]}")

        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            tools=[check_balance_tool.schema, make_payment_tool.schema],
            cancellation_token=ctx.cancellation_token,
        )
        print(f"[RetailBankingAgent] LLM raw output: {llm_result.content}")

        
        if isinstance(llm_result.content, list) and all(isinstance(fc, FunctionCall) for fc in llm_result.content):
            recognized_call = False

            for call in llm_result.content:
                if call.name == "check_balance_func":
                    recognized_call = True
                    await self.publish_message(
                        message,
                        topic_id=TopicId("CheckBalance", ctx.topic_id.source)
                    )
                elif call.name == "make_payment_func":
                    recognized_call = True
                    await self.publish_message(
                        message,
                        topic_id=TopicId("MakePayment", ctx.topic_id.source)
                    )
                else:
                    print(f"[RetailBankingAgent] Unknown tool call: {call.name}")

            
            if not recognized_call:
                fallback_text = "Hmm, I'm not sure how to proceed. Could you clarify your request?"
                new_context = list(message.context)
                new_context.append(AssistantMessage(content=fallback_text, source=self.metadata["type"]))

                await self.publish_message(
                    AgentResponse(context=new_context, reply_to_topic_type=self._my_topic_type),
                    topic_id=TopicId(self._user_topic_type, ctx.topic_id.source)
                )

        else:
           
            final_text = llm_result.content
            if not isinstance(final_text, str):
                final_text = json.dumps(final_text, ensure_ascii=False)

            print(f"[RetailBankingAgent] LLM normal text => {final_text}")

            new_context = list(message.context)
            new_context.append(AssistantMessage(content=final_text, source=self.metadata["type"]))

            await self.publish_message(
                AgentResponse(context=new_context, reply_to_topic_type=self._my_topic_type),
                topic_id=TopicId(self._user_topic_type, ctx.topic_id.source)
            )
