

import csv

def load_credentials_from_csv(file_path: str) -> dict:
    """
    Load user credentials from a CSV file.
    The CSV is expected to have two columns: username,password
    Returns a dictionary mapping usernames to passwords.
    """
    credentials = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            credentials[row['username']] = row['password']
    return credentials
