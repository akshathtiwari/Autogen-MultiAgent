

import json
from typing import List
from autogen_core import RoutedAgent, message_handler, MessageContext, FunctionCall
from autogen_core.models import (
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
)
from autogen_core.tools import Tool

from app.messages.message_types import UserTask, AgentResponse


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
       
        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        print(f"\n*** {self.id.type} initial LLM response ***\n{llm_result.content}", flush=True)

        
        while (
            isinstance(llm_result.content, list) 
            and all(isinstance(m, FunctionCall) for m in llm_result.content)
        ):
            tool_call_results: List[FunctionExecutionResult] = []

            for call in llm_result.content:
                arguments = json.loads(call.arguments)
                if call.name in self._tools:
                    
                    result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token)
                    tool_call_results.append(
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=self._tools[call.name].return_value_as_string(result),
                            is_error=False,
                            name=call.name
                        )
                    )
                else:
                    raise ValueError(f"Unexpected tool called: {call.name}")

            
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

        
        assert isinstance(llm_result.content, str)
        message.context.append(AssistantMessage(content=llm_result.content, source=self.id.type))

        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._my_topic_type),
            topic_id=self._user_topic_type,  
        )
