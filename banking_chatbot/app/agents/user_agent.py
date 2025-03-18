

from autogen_core import RoutedAgent, message_handler, MessageContext, TopicId
from app.messages.message_types import UserLogin, UserTask, AgentResponse
from autogen_core.models import UserMessage


class UserAgent(RoutedAgent):
    """
    The user-facing agent. Handles user logins, gathers queries,
    sends them to DomainClassifier, and displays final responses.
    """
    def __init__(self, description: str, user_topic_type: str, classifier_topic: str):
        super().__init__(description)
        self._user_topic_type = user_topic_type
        self._classifier_topic = classifier_topic

    @message_handler
    async def handle_user_login(self, message: UserLogin, ctx: MessageContext) -> None:
        session_id = self.id.key
        print(f"\n=== Welcome to the Banking Chatbot (session: {session_id}) ===")
        user_name = input("May I know your name?: ")
        print(f"Hello, {user_name}!\n")
        user_query = input("Please describe your banking issue or question: ")

        initial_messages = [UserMessage(content=user_query, source="User")]

        await self.publish_message(
            UserTask(context=initial_messages),
            topic_id=TopicId(self._classifier_topic, source=session_id),
        )

    @message_handler
    async def handle_agent_response(self, message: AgentResponse, ctx: MessageContext) -> None:
        
        last_assistant_msg = ""
        for m in reversed(message.context):
            if hasattr(m, "content") and m.content and isinstance(m.content, str):
                last_assistant_msg = m.content
                break

        print(f"\nAgent says: {last_assistant_msg}")
        user_input = input("User (type 'exit' to close): ")
        if user_input.lower() == "exit":
            print(f"\nEnding session {self.id.key}. Thank you!")
            return

        
        message.context.append(UserMessage(content=user_input, source="User"))
        await self.publish_message(
            UserTask(context=message.context),
            topic_id=TopicId(message.reply_to_topic_type, source=self.id.key),
        )
