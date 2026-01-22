
import wandb
import os

# Set the key provided by the user
os.environ["WANDB_API_KEY"] = "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"
os.environ["proxy"] = "http://10.127.12.17:3128"
os.environ["https_proxy"] = "http://10.127.12.17:3128"
os.environ["http_proxy"] = "http://10.127.12.17:3128"

print("Attempting to init wandb with entity='in-context'...")
try:
    wandb.init(project="moe-icl", entity="in-context", name="test_connection_in_context")
    print("SUCCESS: Connected to entity 'in-context'")
    wandb.finish()
except Exception as e:
    print(f"FAILED: {e}")

print("\nAttempting to init wandb with entity='jinruilin-aijobtech'...")
try:
    wandb.init(project="moe-icl", entity="jinruilin-aijobtech", name="test_connection_jinruilin")
    print("SUCCESS: Connected to entity 'jinruilin-aijobtech'")
    wandb.finish()
except Exception as e:
    print(f"FAILED: {e}")

print("\nAttempting to init wandb with no entity (default)...")
try:
    wandb.init(project="moe-icl", name="test_connection_default")
    print("SUCCESS: Connected to default entity")
    wandb.finish()
except Exception as e:
    print(f"FAILED: {e}")
