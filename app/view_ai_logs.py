#!/usr/bin/env python3
"""
Simple script to view and analyze AI call logs
Usage: python view_ai_logs.py [options]
"""

import os
import glob
from datetime import datetime
import argparse

def get_latest_log_file():
    """Get the most recent log file"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    log_files = glob.glob(os.path.join(log_dir, 'ai_calls_*.log'))
    if not log_files:
        return None
    return max(log_files, key=os.path.getctime)

def count_calls(log_file):
    """Count AI calls by function"""
    function_counts = {}
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'AI CALL - Function:' in line:
                func_name = line.split('Function: ')[1].strip()
                function_counts[func_name] = function_counts.get(func_name, 0) + 1
    return function_counts

def show_latest_call(log_file):
    """Show the most recent AI call"""
    lines = []
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the last separator
    for i in range(len(lines) - 1, -1, -1):
        if '=' * 80 in lines[i]:
            # Print from this separator to the end
            print(''.join(lines[i:]))
            break

def show_function_calls(log_file, function_name):
    """Show all calls to a specific function"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by separator
    calls = content.split('=' * 80)
    matching_calls = [call for call in calls if f'Function: {function_name}' in call]

    print(f"\nFound {len(matching_calls)} calls to {function_name}\n")
    for i, call in enumerate(matching_calls, 1):
        print(f"\n{'=' * 80}")
        print(f"Call #{i}")
        print(call)
        print('=' * 80)

def main():
    parser = argparse.ArgumentParser(description='View AI call logs')
    parser.add_argument('--latest', action='store_true', help='Show the latest AI call')
    parser.add_argument('--count', action='store_true', help='Count calls by function')
    parser.add_argument('--function', type=str, help='Show all calls to a specific function')
    parser.add_argument('--file', type=str, help='Specific log file to analyze')

    args = parser.parse_args()

    # Get log file
    if args.file:
        log_file = args.file
    else:
        log_file = get_latest_log_file()

    if not log_file or not os.path.exists(log_file):
        print("No log files found!")
        return

    print(f"Analyzing: {log_file}\n")

    # Execute requested action
    if args.count:
        counts = count_calls(log_file)
        print("AI Calls by Function:")
        print("-" * 50)
        for func, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{func:30} : {count:3} calls")
    elif args.function:
        show_function_calls(log_file, args.function)
    elif args.latest:
        show_latest_call(log_file)
    else:
        # Default: show statistics
        counts = count_calls(log_file)
        total_calls = sum(counts.values())
        print(f"Total AI calls: {total_calls}")
        print(f"Unique functions: {len(counts)}")
        print("\nTop functions:")
        for func, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {func}: {count} calls")

if __name__ == '__main__':
    main()
