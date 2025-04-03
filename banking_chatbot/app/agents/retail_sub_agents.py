import csv
import os
from datetime import datetime

from autogen_core import RoutedAgent, message_handler, MessageContext, TopicId
from autogen_core.models import (
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ChatCompletionClient,
)
from app.messages.message_types import UserTask, AgentResponse

ACC_CSV_PATH = (
    "C:/Users/akstiwari/OneDrive - Deloitte (O365D)/Desktop/Laptop Files/"
    "Desktop Backup/learning/Autogen-MultiAgent/banking_chatbot/app/credentials/accounts.csv"
)
LEDGER_CSV_PATH = (
    "C:/Users/akstiwari/OneDrive - Deloitte (O365D)/Desktop/Laptop Files/"
    "Desktop Backup/learning/Autogen-MultiAgent/banking_chatbot/app/credentials/ledger.csv"
)


class CheckBalanceAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("CheckBalanceAgent")
        self._model_client = model_client

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        session_id = ctx.topic_id.source
        username = session_id
        balance_val = self.get_balance(username)
        response_text = f"Your current balance is ${balance_val}. Anything else I can help with?"

        new_context = list(message.context)
        new_context.append(AssistantMessage(content=response_text, source=self.id.type))

        await self.publish_message(
            AgentResponse(context=new_context, reply_to_topic_type=self.metadata["type"]),
            topic_id=TopicId("User", session_id)
        )

    def get_balance(self, username: str) -> float:
        if not os.path.exists(ACC_CSV_PATH):
            return 0.0
        with open(ACC_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["username"].strip().lower() == username.strip().lower():
                    return float(row["balance"])
        return 0.0


class MakePaymentAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("MakePaymentAgent")
        self._model_client = model_client

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        print(f"[MakePaymentAgent] handle_task triggered with user content: "
              f"{[m.content for m in message.context if hasattr(m, 'content')]}")

        session_id = ctx.topic_id.source
        details = self.parse_payment_details(message.context)
        missing_fields = self.find_missing(details)

        if missing_fields:
            prompt = f"Please provide missing fields: {', '.join(missing_fields)}"
            new_context = list(message.context)
            new_context.append(AssistantMessage(content=prompt, source=self.id.type))

            await self.publish_message(
                AgentResponse(context=new_context, reply_to_topic_type=self.metadata["type"]),
                topic_id=TopicId("User", session_id)
            )
            return

        amount = float(details["amount"])
        current_balance = self.get_balance(session_id)
        if amount > current_balance:
            fail_resp = f"Payment failed! You only have ${current_balance}, but tried ${amount}."
            new_context = list(message.context)
            new_context.append(AssistantMessage(content=fail_resp, source=self.id.type))
            await self.publish_message(
                AgentResponse(context=new_context, reply_to_topic_type=self.metadata["type"]),
                topic_id=TopicId("User", session_id)
            )
            return

        new_balance = current_balance - amount
        self.update_balance(session_id, new_balance)
        new_tx_id = self.get_next_txid()
        self.write_ledger(sender=session_id, receiver=details["receiver"], transaction_id=new_tx_id)

        success_resp = f"Payment success! TxID={new_tx_id}. New balance=${new_balance}."
        new_context = list(message.context)
        new_context.append(AssistantMessage(content=success_resp, source=self.id.type))

        await self.publish_message(
            AgentResponse(context=new_context, reply_to_topic_type=self.metadata["type"]),
            topic_id=TopicId("User", session_id)
        )

    def parse_payment_details(self, conversation) -> dict:
        details = {"receiver": None, "ifsc": None, "amount": None}
        user_lines = []
        for msg in conversation:
            if isinstance(msg, UserMessage):
                user_lines.append(msg.content.lower())
        joined = "\n".join(user_lines)

        if "receiver" in joined:
            splitted = joined.replace("=", " ").split()
            if "receiver" in splitted:
                idx = splitted.index("receiver")
                if idx + 1 < len(splitted):
                    details["receiver"] = splitted[idx + 1]
        if "ifsc" in joined:
            splitted = joined.replace("=", " ").split()
            if "ifsc" in splitted:
                idx = splitted.index("ifsc")
                if idx + 1 < len(splitted):
                    details["ifsc"] = splitted[idx + 1]
        if "amount" in joined:
            splitted = joined.replace("=", " ").split()
            if "amount" in splitted:
                idx = splitted.index("amount")
                if idx + 1 < len(splitted):
                    details["amount"] = splitted[idx + 1]
        elif "pay" in joined:
            splitted = joined.split()
            if "pay" in splitted:
                idx = splitted.index("pay")
                if idx + 1 < len(splitted):
                    maybe_amt = splitted[idx + 1]
                    if maybe_amt.isdigit():
                        details["amount"] = maybe_amt

        return details

    def find_missing(self, details: dict) -> list:
        missing = []
        for f in ["receiver", "ifsc", "amount"]:
            if not details[f]:
                missing.append(f)
        return missing

    def get_balance(self, username: str) -> float:
        if not os.path.exists(ACC_CSV_PATH):
            return 0.0
        with open(ACC_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["username"].strip().lower() == username.strip().lower():
                    return float(row["balance"])
        return 0.0

    def update_balance(self, username: str, new_balance: float):
        if not os.path.exists(ACC_CSV_PATH):
            return
        rows = []
        updated = False
        with open(ACC_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["username"].strip().lower() == username.strip().lower():
                    row["balance"] = str(new_balance)
                    updated = True
                rows.append(row)
        if updated and fieldnames:
            with open(ACC_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def get_next_txid(self) -> str:
        if not os.path.exists(LEDGER_CSV_PATH):
            return "TX001"
        max_num = 0
        with open(LEDGER_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                txid = row["transaction_id"]
                if txid.lower().startswith("tx"):
                    try:
                        num_val = int(txid[2:])
                        if num_val > max_num:
                            max_num = num_val
                    except:
                        pass
        new_val = max_num + 1
        return f"TX{new_val:03d}"

    def write_ledger(self, sender: str, receiver: str, transaction_id: str):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = {
            "sender": sender,
            "receiver": receiver,
            "transaction_id": transaction_id,
            "time_stamp": now_str
        }
        file_exists = os.path.exists(LEDGER_CSV_PATH)
        fieldnames = ["sender", "receiver", "transaction_id", "time_stamp"]
        with open(LEDGER_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_row)
