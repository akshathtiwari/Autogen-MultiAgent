

from autogen_core import RoutedAgent, message_handler, MessageContext, TopicId
from app.messages.message_types import UserCredentials, UserLogin
from autogen_core.models import SystemMessage
from app.tools.credential_utils import load_credentials_from_csv

class AuthenticationAgent(RoutedAgent):
    
    def __init__(self, credentials_csv_path: str, user_topic: str) -> None:
        super().__init__("AuthenticationAgent")
        self.valid_users = load_credentials_from_csv(credentials_csv_path)
        self.user_topic = user_topic

    @message_handler
    async def handle_credentials(self, message: UserCredentials, ctx: MessageContext) -> None:
        username = message.username
        password = message.password

        if username in self.valid_users and self.valid_users[username] == password:
            print(f"[{self.id.type}] User '{username}' authenticated successfully.")
            
            await self.publish_message(
                UserLogin(username=username),
                topic_id=TopicId(self.user_topic, source=username)
            )
        else:
            print(f"[{self.id.type}] Authentication failed for user '{username}'.")
            
