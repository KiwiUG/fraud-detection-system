import pandas as pd
import random
import numpy as np
import os

print("Starting data generation with new user profiles...")

N_ROWS = 1000
DATA_FILE = os.path.join("data", "user_data.csv")

# --- NEW USER PROFILES ---
USER_SAFE = "utsav16"
USER_MIXED_FRAUD = "aarnov123" # Changed ID
USER_FULL_SCAMMER = "himansu367"

USERS = [USER_SAFE, USER_MIXED_FRAUD, USER_FULL_SCAMMER]
# ALL_TYPES removed as it's no longer needed

generated_data = []

# --- TRANSACTION GENERATORS ---

def generate_safe_transaction():
    """Generates a simple, safe transaction (PAYMENT or CASH_IN)"""
    ttype = random.choice(['PAYMENT', 'CASH_IN'])
    amount = round(random.uniform(10.0, 500.0), 2)
    s_old = round(random.uniform(amount, 50000.0), 2)
    r_old = round(random.uniform(0.0, 100000.0), 2)
    
    if ttype == 'PAYMENT':
        s_new = s_old - amount
        r_new = r_old + amount
    else: # CASH_IN
        s_new = s_old + amount
        r_new = r_old - amount 

    return ttype, amount, s_old, s_new, r_old, r_new

# generate_mixed_safe_transaction() removed for simplicity

def generate_fraud_transaction():
    """Generates a classic high-risk, account-draining TRANSFER"""
    ttype = 'TRANSFER'
    amount = round(random.uniform(50000.0, 1000000.0), 2)
    s_old = amount # Draining the account
    s_new = 0.0
    r_old = 0.0 # New/mule account
    r_new = amount
    
    return ttype, amount, s_old, s_new, r_old, r_new

# --- MAIN GENERATION LOOP (Updated Logic) ---

for i in range(N_ROWS):
    step = i + 1
    # Pick one of the users at random to assign this transaction to
    user_id = random.choice(USERS)
    
    if user_id == USER_SAFE:
        # This user is *only* allowed to have safe transactions
        ttype, amount, s_old, s_new, r_old, r_new = generate_safe_transaction()
    
    elif user_id == USER_MIXED_FRAUD:
        # CHANGED: 85% chance of a normal transaction, 15% chance of a fraud one
        if random.random() < 0.0015: # 15% fraud chance
            ttype, amount, s_old, s_new, r_old, r_new = generate_fraud_transaction()
        else: # 85% normal chance
            # CHANGED: Use simple safe transaction
            ttype, amount, s_old, s_new, r_old, r_new = generate_safe_transaction()
            
    elif user_id == USER_FULL_SCAMMER:
        # 40% chance of a fraud transaction
        if random.random() < 0.40: # 40% fraud chance
            ttype, amount, s_old, s_new, r_old, r_new = generate_fraud_transaction()
        else: # 60% normal chance
            # CHANGED: Use simple safe transaction
            ttype, amount, s_old, s_new, r_old, r_new = generate_safe_transaction()

    # Generate random sender/receiver IDs
    sender = f"C{random.randint(1000000000, 9999999999)}"
    receiver = f"{random.choice(['C', 'M'])}{random.randint(1000000000, 9999999999)}"

    # Append to our list using the correct CSV format
    generated_data.append({
        'user_id': user_id,
        'step': step,
        'type': ttype,
        'amount': amount,
        'sender_old_bal': s_old,
        'sender_new_bal': s_new,
        'receiver_old_bal': r_old,
        'receiver_new_bal': r_new,
        'sender': sender,
        'receiver': receiver
    })

# Convert to DataFrame and save
df = pd.DataFrame(generated_data)

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
df.to_csv(DATA_FILE, index=False)

print(f"✅ Successfully generated {len(df)} rows and saved to {DATA_FILE}")
print("\nUser value counts:")
print(df['user_id'].value_counts())

