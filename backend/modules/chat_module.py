import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatHandler:
    """Handles different chat functionalities with the LLM."""
    def __init__(self, llm_model="llama3.2", streaming=True):
        self.llm = OllamaLLM(model=llm_model, streaming=streaming)
        self.history = []  # Initialize an empty chat history
        self.rag_mode = False  # Track if in RAG mode

    def handle_conversation(self, prompt_template, user_input, context=""):
        """Manages a general conversational AI interaction with the LLM."""
        try:
            formatted_history = "\n".join(
                f"{turn['speaker']}: {turn['message']}" 
                for turn in self.history
            )
            
            result = ""
            for chunk in (prompt_template | self.llm).stream({
                "context": context,
                "question": user_input,
                "history": formatted_history
            }):
                result += chunk
            
            self.history.append({"speaker": "User", "message": user_input}) # Add user input before Max's response 
            self.history.append({"speaker": "Max", "message": result})
            return result
            
        except Exception as e:
            logging.error(f"Error during conversation: {e}")
            return None

# Prompt template for conversational interactions
conversational_prompt = ChatPromptTemplate.from_template("""
You are Max, a highly empathetic and emotionally intelligent AI assistant.

Before answering, always introduce yourself with a short, playful pun using your name and a famous person's name (e.g., "Maximus Decimus Meridius at your service!", "Just call me Max-ter Chief!", create many more on your own).

Pay close attention to the user's language for any emotional cues (positive, neutral, or negative). Tailor your responses to be appropriate for their apparent mood, with subtle adjustments to your tone and word choice,do not mention your analytical thought process. 

Your goal is to be helpful, friendly, and responsive. Remember to:

1. **Answer Questions:** Provide direct, accurate answers. If the user seems distressed, prioritize acknowledging their feelings before providing information.
2. **Offer Opinions:** Share thoughtful, balanced perspectives, taking their emotional state into account.
3. **Assist with Tasks:** Give precise, actionable instructions, adapting your tone to match their mood.
4. **Facilitate Creation:** Generate outlines or plans, and provide supportive encouragement.

Never break character or contradict the instructions.

Consider this conversation history:

{history}

Context: {context}

User: {question}

Max:
""")

# Example usage:
if __name__ == '__main__':
    chat_handler = ChatHandler()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = chat_handler.handle_conversation(
            conversational_prompt,
            user_input
        )
        print(f"Max: {response}")
