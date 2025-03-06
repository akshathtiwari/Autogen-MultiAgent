import asyncio
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler, BaseAgent

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_core import SingleThreadedAgentRuntime

import os 
from dotenv import load_dotenv

load_dotenv()

API_KEY= os.environ.get("OPENAI_API_KEY")

@dataclass
class MyMessageType:
    content: str

@dataclass
class Message:
    content: str

#creating a MyAgent from parent RoutedAgent class
#defined message handler using decorator @message_handler
class MyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")


class MyAssistant(RoutedAgent):
    def __init__(self, description: str, my_agent_type: str) -> None:
        super().__init__(description)
        model_client = OpenAIChatCompletionClient(
                        model="gpt-4o-mini",
                        api_key=API_KEY,
                        )
        self._delegate = AssistantAgent(description, model_client=model_client)
        self.my_agent_id = AgentId(my_agent_type, self.id.key)

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")


async def main():
    runtime = SingleThreadedAgentRuntime()
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())
    await MyAssistant.register(runtime, "my_assistant", lambda: MyAssistant("my_assistant", "my_agent"))

    runtime.start()  # Start processing messages in the background.
    my_assistant_id = AgentId("my_assistant", "default")
    await runtime.send_message(MyMessageType(content="Hello, World!"), my_assistant_id)
    await runtime.stop_when_idle()    

asyncio.run(main())