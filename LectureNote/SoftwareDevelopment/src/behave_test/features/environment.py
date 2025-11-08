# File: features/environment.py

# Global shared database
db = {}

def before_all(context):
    context.db = db  # Attach to context for easy access

def before_scenario(context, scenario):
    # Optional: reset per scenario
    # context.db.clear()
    pass