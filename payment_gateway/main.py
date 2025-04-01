import tkinter as tk
from tkinter import ttk
import pandas as pd
import threading
import time


CSV_FILE = "C:/Users/akstiwari/OneDrive - Deloitte (O365D)/Desktop/Laptop Files/Desktop Backup/learning/Autogen-MultiAgent/payment_gateway/transactions.csv"

def load_data():
    
    try:
        df = pd.read_csv(CSV_FILE)
        return df
    except Exception as e:
        print("Error loading CSV:", e)
        return pd.DataFrame()

def update_table(tree):
    
    df = load_data()
    
    for row in tree.get_children():
        tree.delete(row)
    
    
    for _, row in df.iterrows():
        
        tags = ()
        if row['PaymentStatus'] != row['CoreBankingStatus']:
            tags = ('discrepancy',)
        tree.insert("", "end", values=(row['TransactionID'], row['Timestamp'],
                                       row['Amount'], row['PaymentStatus'],
                                       row['CoreBankingStatus']), tags=tags)
    
   
    tree.tag_configure('discrepancy', background='red')

def refresh_data(tree, interval=5):
    
    while True:
        update_table(tree)
        time.sleep(interval)

def main():
    
    root = tk.Tk()
    root.title("Payment Gateway Logs Dashboard")

    
    columns = ('TransactionID', 'Timestamp', 'Amount', 'PaymentStatus', 'CoreBankingStatus')
    tree = ttk.Treeview(root, columns=columns, show='headings')
    
  
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)
    tree.pack(fill=tk.BOTH, expand=True)
    
   
    threading.Thread(target=refresh_data, args=(tree,), daemon=True).start()
    
    
    root.mainloop()

if __name__ == "__main__":
    main()
