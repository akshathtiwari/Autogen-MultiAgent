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
class Message:
    content: str


class InnerAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        return Message(content=f"Hello from inner, {message.content}")


class OuterAgent(RoutedAgent):
    def __init__(self, description: str, inner_agent_type: str):
        super().__init__(description)
        self.inner_agent_id = AgentId(inner_agent_type, self.id.key)

    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"Received message: {message.content}")
        # Send a direct message to the inner agent and receves a response.
        response = await self.send_message(Message(f"Hello from outer, {message.content}"), self.inner_agent_id)
        print(f"Received inner response: {response.content}")

async def main():
    runtime = SingleThreadedAgentRuntime()
    await InnerAgent.register(runtime, "inner_agent", lambda: InnerAgent("InnerAgent"))
    await OuterAgent.register(runtime, "outer_agent", lambda: OuterAgent("OuterAgent", "inner_agent"))
    runtime.start()
    outer_agent_id = AgentId("outer_agent", "default")
    await runtime.send_message(Message(content="Hello, World!"), outer_agent_id)
    await runtime.stop_when_idle()


asyncio.run(main())