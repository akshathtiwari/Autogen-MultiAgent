import asyncio
from typing import Dict, List, Any
from collections import defaultdict
from fastapi import WebSocket

from autogen_core import SingleThreadedAgentRuntime, TopicId, TypeSubscription
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import SystemMessage

from app.messages.message_types import (
    UserCredentials,
    UserLogin,
    UserTask,
    AgentResponse
)

from app.agents.authentication_agent import AuthenticationAgent
from app.agents.domain_classifier_agent import DomainClassifierAgent
from app.agents.domain_agents import (
    register_retail_banking_agent,
    register_make_payment_agent,
    register_check_balance_agent,
    register_payments_agent,
    register_corporate_banking_agent,
    register_investment_banking_agent,
    register_wealth_management_agent,
    register_risk_management_agent,
    register_insurance_agent,
    register_it_ops_agent,
    register_capital_treasury_agent,
    register_analytics_agent
)
from app.tools.delegate_tools import (
    transfer_to_retail_banking_tool,
    transfer_to_corporate_banking_tool,
    transfer_to_investment_banking_tool,
    transfer_to_wealth_management_tool,
    transfer_to_risk_management_tool,
    transfer_to_insurance_tool,
    transfer_to_it_ops_tool,
    transfer_to_payments_tool,
    transfer_to_capital_treasury_tool,
    transfer_to_analytics_tool
)


class ConversationStateAccessor:
    def __init__(self, state_dict, msg_context_dict):
        self._state_dict = state_dict
        self._msg_context_dict = msg_context_dict

    def get_status(self, session_id: str) -> str:
        return self._state_dict[session_id].get("status", "fresh")

    def set_status(self, session_id: str, val: str):
        self._state_dict[session_id]["status"] = val

    def reset(self, session_id: str):
        self._state_dict[session_id] = {}

    def get_last_agent(self, session_id: str):
        return self._state_dict[session_id].get("last_agent", None)

    def set_last_agent(self, session_id: str, agent: str):
        self._state_dict[session_id]["last_agent"] = agent

    def reset_messages(self, session_id: str):
        self._msg_context_dict[session_id] = []


class HookedAgentRuntime(SingleThreadedAgentRuntime):
    def __init__(self, on_agent_response_callback):
        super().__init__()
        self._on_agent_response_callback = on_agent_response_callback

    async def publish_message(self, message, topic_id, **kwargs):
        if isinstance(message, AgentResponse):
            self._on_agent_response_callback(message, topic_id)
        return await super().publish_message(message, topic_id, **kwargs)


class RuntimeManager:
    def __init__(self):
        self._runtime = HookedAgentRuntime(self._on_agent_response)
        self._model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=None)

        self._response_queues: Dict[str, List[AgentResponse]] = defaultdict(list)
        self._websockets: Dict[str, WebSocket] = {}
        self._conversation_context: Dict[str, List] = defaultdict(list)
        self._conversation_state: Dict[str, Dict[str, Any]] = defaultdict(dict)

        self.conversation_accessor = ConversationStateAccessor(
            self._conversation_state,
            self._conversation_context
        )

    async def start_runtime(self):
        delegate_tools = [
            transfer_to_retail_banking_tool,
            transfer_to_corporate_banking_tool,
            transfer_to_investment_banking_tool,
            transfer_to_wealth_management_tool,
            transfer_to_risk_management_tool,
            transfer_to_insurance_tool,
            transfer_to_it_ops_tool,
            transfer_to_payments_tool,
            transfer_to_capital_treasury_tool,
            transfer_to_analytics_tool
        ]

        # Register DomainClassifierAgent
        classifier_type = await DomainClassifierAgent.register(
            self._runtime,
            type="DomainClassifier",
            factory=lambda: DomainClassifierAgent(
                description="DomainClassifierAgent",
                system_message=SystemMessage(content="You are the triage agent for banking queries."),
                model_client=self._model_client,
                delegate_tools=delegate_tools,
                my_topic_type="DomainClassifier",
                user_topic_type="User",
                conversation_state_accessor=self.conversation_accessor,
            )
        )
        await self._runtime.add_subscription(
            TypeSubscription(topic_type="DomainClassifier", agent_type=classifier_type.type)
        )

        credentials_csv = (
            "C:/Users/akstiwari/OneDrive - Deloitte (O365D)/Desktop/Laptop Files/Desktop Backup/"
            "learning/Autogen-MultiAgent/banking_chatbot/app/credentials/users.csv"
        )
        auth_type = await AuthenticationAgent.register(
            self._runtime,
            type="Auth",
            factory=lambda: AuthenticationAgent(credentials_csv_path=credentials_csv, user_topic="User")
        )
        await self._runtime.add_subscription(
            TypeSubscription(topic_type="Auth", agent_type=auth_type.type)
        )

        await register_retail_banking_agent(self._runtime, self._model_client)
        await register_check_balance_agent(self._runtime, self._model_client)
        await register_make_payment_agent(self._runtime, self._model_client)
        await register_payments_agent(self._runtime, self._model_client, self.conversation_accessor)
        await register_corporate_banking_agent(self._runtime, self._model_client)
        await register_investment_banking_agent(self._runtime, self._model_client)
        await register_wealth_management_agent(self._runtime, self._model_client)
        await register_risk_management_agent(self._runtime, self._model_client)
        await register_insurance_agent(self._runtime, self._model_client)
        await register_it_ops_agent(self._runtime, self._model_client)
        await register_capital_treasury_agent(self._runtime, self._model_client)
        await register_analytics_agent(self._runtime, self._model_client)

        self._runtime.start()

    async def stop_runtime(self):
        await self._runtime.stop_when_idle()

    async def publish_credentials(self, creds: UserCredentials, session_id: str):
        await self._runtime.publish_message(
            creds,
            topic_id=TopicId("Auth", source=session_id)
        )

    async def publish_user_message(self, user_text: str, session_id: str):
        from autogen_core.models import UserMessage
        from app.messages.message_types import UserTask

        self._conversation_context[session_id].append(UserMessage(content=user_text, source="User"))
        user_task = UserTask(context=self._conversation_context[session_id])

        st = self._conversation_state[session_id].get("status", "fresh")
        last_agent = self._conversation_state[session_id].get("last_agent", None)

        if st == "follow_up" and last_agent is not None:
            await self._runtime.send_message(
                user_task,
                agent_id=TopicId(last_agent, source=session_id)
            )
        elif st == "post_action":
            await self._runtime.publish_message(
                user_task,
                topic_id=TopicId("DomainClassifier", source=session_id)
            )
        else:
            await self._runtime.publish_message(
                user_task,
                topic_id=TopicId("DomainClassifier", source=session_id)
            )

    def register_websocket(self, session_id: str, ws: WebSocket):
        self._websockets[session_id] = ws

    def unregister_websocket(self, session_id: str):
        if session_id in self._websockets:
            del self._websockets[session_id]

    def drain_agent_responses(self, session_id: str):
        if session_id not in self._response_queues:
            return []
        arr = self._response_queues[session_id][:]
        self._response_queues[session_id].clear()
        return arr

    def set_post_action_state(self, session_id: str, agent_type: str):
        self._conversation_state[session_id]["status"] = "post_action"
        self._conversation_state[session_id]["last_agent"] = agent_type

    def set_follow_up_state(self, session_id: str, agent_type: str):
        self._conversation_state[session_id]["status"] = "follow_up"
        self._conversation_state[session_id]["last_agent"] = agent_type

    def reset_state(self, session_id: str):
        self._conversation_state[session_id] = {}

    def _on_agent_response(self, response: AgentResponse, topic_id):
       
        if not hasattr(topic_id, "source"):
            session_id = ""
            for msg in response.context:
                if hasattr(msg, "source") and msg.source:
                    session_id = msg.source
                    break
            topic_id = TopicId(topic_id, session_id)
        session_id = topic_id.source
        if session_id in self._websockets:
            ws = self._websockets[session_id]
            import json
            texts = []
            for msg in response.context:
                if hasattr(msg, "content"):
                    texts.append(str(msg.content))
            joined = "\n".join(texts)
            payload = {"type": "agent_response", "text": joined}
            asyncio.create_task(ws.send_json(payload))
        else:
            self._response_queues[session_id].append(response)
