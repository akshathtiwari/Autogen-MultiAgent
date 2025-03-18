import tkinter as tk
from tkinter import ttk
import pandas as pd
import threading
import time

# Path to your CSV file containing transaction logs
CSV_FILE = "C:/Users/akstiwari/OneDrive - Deloitte (O365D)/Desktop/Laptop Files/Desktop Backup/learning/Autogen-MultiAgent/payment_gateway/transactions.csv"

def load_data():
    """
    Load transaction data from CSV.
    Expected columns: TransactionID, Timestamp, Amount, PaymentStatus, CoreBankingStatus
    """
    try:
        df = pd.read_csv(CSV_FILE)
        return df
    except Exception as e:
        print("Error loading CSV:", e)
        return pd.DataFrame()

def update_table(tree):
    """
    Clears and updates the GUI table with transaction data.
    Highlights rows with discrepancies between PaymentStatus and CoreBankingStatus.
    """
    df = load_data()
    # Clear existing data in the tree
    for row in tree.get_children():
        tree.delete(row)
    
    # Insert new rows and highlight discrepancies
    for _, row in df.iterrows():
        # A discrepancy is when PaymentStatus does not match CoreBankingStatus
        tags = ()
        if row['PaymentStatus'] != row['CoreBankingStatus']:
            tags = ('discrepancy',)
        tree.insert("", "end", values=(row['TransactionID'], row['Timestamp'],
                                       row['Amount'], row['PaymentStatus'],
                                       row['CoreBankingStatus']), tags=tags)
    
    # Configure the discrepancy tag to have a red background
    tree.tag_configure('discrepancy', background='red')

def refresh_data(tree, interval=5):
    """
    Periodically refresh the table every 'interval' seconds.
    """
    while True:
        update_table(tree)
        time.sleep(interval)

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Payment Gateway Logs Dashboard")

    # Define the columns for the transaction logs
    columns = ('TransactionID', 'Timestamp', 'Amount', 'PaymentStatus', 'CoreBankingStatus')
    tree = ttk.Treeview(root, columns=columns, show='headings')
    
    # Set headings for each column
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)
    tree.pack(fill=tk.BOTH, expand=True)
    
    # Start a background thread to refresh the data periodically
    threading.Thread(target=refresh_data, args=(tree,), daemon=True).start()
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
