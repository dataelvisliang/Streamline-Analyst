import json
from src.llm_service import call_openrouter, get_model_name

class PlanningAgent:
    """
    Task planning agent that analyzes user queries and determines:
    1. Which data analysis mode to use
    2. What steps to take next
    3. How to respond to non-data analysis queries
    4. Autonomously executes analysis when appropriate
    """

    def __init__(self, api_key, model_type=4):
        self.api_key = api_key
        self.model_type = model_type
        self.conversation_history = []
        self.should_execute = False
        self.execution_mode = None

    def analyze_query(self, user_query, has_data_uploaded=False):
        """
        Analyze user query and determine intent and next steps.

        Args:
            user_query: User's question or request
            has_data_uploaded: Whether user has uploaded data

        Returns:
            dict with keys:
                - intent: 'data_analysis', 'general_question', or 'clarification_needed'
                - analysis_mode: One of the analysis modes or None
                - response: Text response to user
                - next_steps: List of actions to take
        """

        system_prompt = f"""You are a task planning agent for AI Analytics Engine, a data analysis application.

Your role is to:
1. Understand the user's intent from their query
2. Determine which data analysis mode is appropriate (if any)
3. Plan the next steps for the user
4. Respond helpfully to non-data analysis questions

Available data analysis modes:
- Predictive Classification: For predicting categorical outcomes (e.g., "Will customer churn?", "Classify spam emails")
- Clustering Model: For finding groups/patterns in data (e.g., "Segment customers", "Find similar items")
- Regression Model: For predicting numerical values (e.g., "Predict house prices", "Forecast sales")
- Data Visualization: For creating charts and exploring data visually

User has uploaded data: {has_data_uploaded}

Analyze the user's query and respond with a JSON object containing:
{{
    "intent": "data_analysis" | "general_question" | "clarification_needed",
    "analysis_mode": "Predictive Classification" | "Clustering Model" | "Regression Model" | "Data Visualization" | null,
    "response": "Your helpful response to the user",
    "next_steps": ["Step 1", "Step 2", ...],
    "needs_data": true | false
}}

Guidelines:
- If the query is about data analysis but no data is uploaded, suggest uploading data first
- For general questions about the tool, provide helpful information
- If unclear what the user wants, ask clarifying questions
- Be conversational and friendly
- Keep responses concise but informative
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {user_query}"}
        ]

        try:
            model_name = get_model_name(self.model_type)
            response = call_openrouter(
                messages=messages,
                model_name=model_name,
                api_key=self.api_key,
                function_name="analyze_query"
            )

            # Parse JSON response
            result = json.loads(response)

            # Set execution flags if analysis mode is determined and data is available
            if result.get('analysis_mode') and has_data_uploaded:
                self.should_execute = True
                self.execution_mode = result.get('analysis_mode')
            else:
                self.should_execute = False
                self.execution_mode = None

            # Store in conversation history
            self.conversation_history.append({
                "user_query": user_query,
                "analysis": result
            })

            return result

        except json.JSONDecodeError:
            # Fallback if response isn't valid JSON
            return {
                "intent": "general_question",
                "analysis_mode": None,
                "response": response,
                "next_steps": ["Please clarify your request"],
                "needs_data": False
            }
        except Exception as e:
            return {
                "intent": "error",
                "analysis_mode": None,
                "response": f"I encountered an error: {str(e)}. Could you rephrase your question?",
                "next_steps": [],
                "needs_data": False
            }

    def chat(self, user_message, has_data_uploaded=False):
        """
        Simple chat interface that maintains context.

        Args:
            user_message: User's message
            has_data_uploaded: Whether data is uploaded

        Returns:
            String response to user
        """

        # Build context from conversation history
        context = ""
        if self.conversation_history:
            context = "Previous conversation:\n"
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                context += f"User: {entry['user_query']}\n"
                context += f"Assistant: {entry['analysis']['response']}\n\n"

        system_prompt = f"""You are a helpful AI assistant for AI Analytics Engine, a data analysis application.

You can help users with:
1. Understanding what the application can do
2. Choosing the right analysis mode for their data
3. Answering general questions about data analysis
4. Planning their analysis workflow

Available analysis modes:
- Predictive Classification: Predict categories/classes
- Clustering Model: Find groups and patterns
- Regression Model: Predict numerical values
- Data Visualization: Explore and visualize data

User has data uploaded: {has_data_uploaded}

{context}

Respond naturally and helpfully. If the user wants to perform data analysis, guide them on which mode to use and what to do next."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            model_name = get_model_name(self.model_type)
            response = call_openrouter(
                messages=messages,
                model_name=model_name,
                api_key=self.api_key,
                function_name="planning_agent_chat"
            )

            # Store in conversation history
            self.conversation_history.append({
                "user_query": user_message,
                "analysis": {
                    "response": response,
                    "intent": "chat"
                }
            })

            return response

        except Exception as e:
            return f"I encountered an error: {str(e)}. Could you try again?"

    def get_conversation_history(self):
        """Return the conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def reset_execution_flags(self):
        """Reset execution flags after analysis is triggered"""
        self.should_execute = False
        self.execution_mode = None
