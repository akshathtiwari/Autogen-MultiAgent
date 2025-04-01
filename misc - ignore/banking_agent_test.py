import os
from dotenv import load_dotenv
import asyncio

from autogen_core import RoutedAgent, message_handler, MessageContext, SingleThreadedAgentRuntime, AgentId

from dataclasses import dataclass

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")

# Dataclass is used to store data in structured classes
@dataclass
class MessageType():
    content: str

class RouterAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("RouterAgent")
        
    @message_handler
    async def handle_message(self, message: MessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
        

async def main():
    runtime = SingleThreadedAgentRuntime()    
    
    await RoutedAgent.register(runtime=runtime,type="rounter_agent", factory=RouterAgent())
    runtime.start()
    await runtime.send_message(MessageType("Hey Akshath"), AgentId("rounter_agent", "default"))
    await runtime.stop()
    
if __name__ == "__main__":
    asyncio.run(main())
    
