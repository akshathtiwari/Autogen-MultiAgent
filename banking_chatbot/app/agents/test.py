

import csv
import os
ACC_CSV_PATH = (
    "C:/Users/akstiwari/OneDrive - Deloitte (O365D)/Desktop/Laptop Files/"
    "Desktop Backup/learning/Autogen-MultiAgent/banking_chatbot/app/credentials/accounts.csv"
)
LEDGER_CSV_PATH = (
    "app/credentials/ledger.csv"
)



if not os.path.exists(LEDGER_CSV_PATH):
    print("TX001")
max_num = 0
with open(LEDGER_CSV_PATH, newline="\n", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Row: {row}")
        txid = row["transaction_id"]
        # print(f"Row: {row}")
        print(f"txid: {txid}")
        if txid.lower().startswith("tx"):
            try:
                num_val = int(txid[2:])
                if num_val > max_num:
                    max_num = num_val
            except:
                pass
new_val = max_num + 1
print(f"TX{new_val:03d}")
    
    

    
    