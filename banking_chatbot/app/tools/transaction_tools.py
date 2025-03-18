

import csv
import os
from autogen_core.tools import FunctionTool

CSV_PATH = "C:/Users/akstiwari/OneDrive - Deloitte (O365D)/Desktop/Laptop Files/Desktop Backup/learning/Autogen-MultiAgent/payment_gateway/transactions.csv"


def lookup_transaction(transaction_id: str, csv_path: str = CSV_PATH) -> dict:
    """
    Reads the CSV to find the given transaction_id.
    Returns a dict with TransactionID, Timestamp, Amount, PaymentStatus, and CoreBankingStatus.
    If not found, returns an empty dict or a dict with an 'error' field.
    """
    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found at {csv_path}"}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["TransactionID"] == transaction_id:
                return {
                    "TransactionID": row["TransactionID"],
                    "Timestamp": row["Timestamp"],
                    "Amount": row["Amount"],
                    "PaymentStatus": row["PaymentStatus"],
                    "CoreBankingStatus": row["CoreBankingStatus"],
                }

    return {"error": f"Transaction '{transaction_id}' not found."}


def fix_core_banking_status(transaction_id: str, csv_path: str = CSV_PATH) -> dict:
    """
    If PaymentStatus is Success and CoreBankingStatus != Success, updates the CSV row
    so that CoreBankingStatus=Success for the given transaction ID.
    Returns a dict indicating success or error.
    """
    print(f"CSV path: {csv_path}")
    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found at {csv_path}"}

    updated = False
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return {"error": "CSV has no headers."}

        for row in reader:
            if row["TransactionID"] == transaction_id:
                if row["PaymentStatus"] == "Success" and row["CoreBankingStatus"] != "Success":
                    row["CoreBankingStatus"] = "Success"
                    updated = True
            rows.append(row)

    if not updated:
        return {
            "error": (
                f"No update done. Possibly transaction not found or PaymentStatus not Success."
            )
        }

    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "success": True,
        "message": f"Updated transaction {transaction_id} so CoreBankingStatus=Success."
    }


lookup_transaction_tool = FunctionTool(
    lookup_transaction,
    description=(
        "Look up a transaction by ID in the CSV. Returns PaymentStatus and CoreBankingStatus. "
        "Args: transaction_id (str) and optional csv_path (str)."
    ),
)

fix_core_banking_status_tool = FunctionTool(
    fix_core_banking_status,
    description=(
        "Fix a mismatch by setting CoreBankingStatus=Success if PaymentStatus=Success. "
        "Args: transaction_id (str) and optional csv_path (str)."
    ),
)
