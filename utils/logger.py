import datetime

def log_event(event_type, message):
    """Appends an event to the run_logs.txt file with a timestamp."""
    log_path = "logs/run_logs.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] [{event_type}] {message}\n")

# Example usage inside your main loop:
# log_event("INFO", f"Processing Question ID: {q['id']}")
# log_event("AGENT_RETRIEVER", "Context successfully retrieved from FAISS.")