# AI Call Logs

This directory contains detailed logs of all AI interactions in the AI Analytics Engine application.

## Log Files

- **File naming**: `ai_calls_YYYYMMDD.log` - One file per day
- **Location**: `app/logs/`

## What's Logged

Each AI call includes:
1. **Timestamp** - When the call was made
2. **Function Name** - Which function made the AI call (e.g., `decide_encode_type`, `decide_model`)
3. **Model Used** - Which AI model was used (e.g., `x-ai/grok-4.1-fast:free`)
4. **Full Prompt** - The complete prompt sent to the AI
5. **Full Response** - The complete response received from the AI
6. **Errors** - Any errors that occurred during the call

## Log Format Example

```
================================================================================
AI CALL - Function: decide_encode_type
Model: x-ai/grok-4.1-fast:free
Timestamp: 2025-11-23 14:30:45

Prompt sent:
[Full prompt text here...]

Response received:
[Full AI response here...]
================================================================================
```

## How to View Logs

### In Console (while running)
Logs are automatically printed to the console when you run the Streamlit app.

### In Log Files
```bash
# View today's log
cat app/logs/ai_calls_$(date +%Y%m%d).log

# Or on Windows
type app\logs\ai_calls_20251123.log

# Follow logs in real-time
tail -f app/logs/ai_calls_$(date +%Y%m%d).log
```

### Search for specific functions
```bash
# Find all calls to decide_model
grep -A 20 "Function: decide_model" app/logs/ai_calls_*.log

# Count how many times each function was called
grep "AI CALL - Function:" app/logs/ai_calls_*.log | sort | uniq -c
```

## Privacy Note

These logs may contain your data. Do not share these log files publicly if they contain sensitive information.
