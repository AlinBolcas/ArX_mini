import os
import json
from typing import List, Dict, Optional
from datetime import datetime
import sys

# Ensure we always resolve to the correct output directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import Utils  # Importing the utility functions


class Memory:
    """Handles short-term and long-term memory for conversational context retention."""

    def __init__(self, short_term_limit: int = 8000, long_term_interval: int = 3):
        """
        Initialize Memory system.
        
        Args:
            short_term_limit (int): Max token count for short-term memory before trimming.
            long_term_interval (int): Number of interactions after which to attempt saving long-term memory.
        """
        # Ensure the output directory is always the top-level project output folder
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.memory_dir = os.path.join(project_root, "output", "memory")

        os.makedirs(self.memory_dir, exist_ok=True)

        self.short_term_file = os.path.join(self.memory_dir, "short_term_memory.json")
        self.long_term_file = os.path.join(self.memory_dir, "long_term_memory.json")
        self.short_term_markdown = os.path.join(self.memory_dir, "short_term_memory.md")
        self.long_term_markdown = os.path.join(self.memory_dir, "long_term_memory.md")
        self.short_term_limit = short_term_limit

        # Initialize an internal counter for short-term interactions.
        self._short_term_counter = 0
        self.long_term_interval = long_term_interval

        # Ensure files exist
        if not os.path.exists(self.short_term_file):
            with open(self.short_term_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)

        if not os.path.exists(self.long_term_file):
            with open(self.long_term_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)

    def reboot_memory(self):
        """Deletes the entire memory folder and resets all memory files."""
        if os.path.exists(self.memory_dir):
            import shutil
            shutil.rmtree(self.memory_dir)
            print("Memory has been reset.")

        # Reinitialize memory directory and files
        os.makedirs(self.memory_dir, exist_ok=True)

        with open(self.short_term_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)

        with open(self.long_term_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)

        print("Memory has been reinitialized.")

    def save_short_term(self, system_message: str, user_message: str, assistant_message: str):
        """
        Save interaction to the single short-term memory JSON file.
        If the file exceeds the token limit, trim older interactions.
        Automatically triggers long-term memory extraction every `long_term_interval` interactions.
        """
        try:
            with open(self.short_term_file, "r", encoding="utf-8") as f:
                short_term_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            short_term_data = []

        # Append the new interaction
        short_term_data.append({
            "system": system_message,
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": datetime.now().isoformat()
        })

        # Trim the oldest interactions if exceeding the token limit
        total_tokens = sum(len(entry["user"]) + len(entry["assistant"]) for entry in short_term_data)
        while total_tokens > self.short_term_limit and short_term_data:
            removed_entry = short_term_data.pop(0)
            total_tokens -= len(removed_entry["user"]) + len(removed_entry["assistant"])

        # Save updated short-term memory
        with open(self.short_term_file, "w", encoding="utf-8") as f:
            json.dump(short_term_data, f, indent=4)

        # Keep markdown synced
        self.convert_memory_to_markdown(self.short_term_file, self.short_term_markdown)

        # Increment our counter and automatically store long-term memory if threshold is reached.
        self._short_term_counter += 1
        if self._short_term_counter >= self.long_term_interval:
            self.store_long_term_memory(interactions=self.long_term_interval)
            self._short_term_counter = 0

    def retrieve_short_term(self) -> List[Dict[str, str]]:
        """Retrieve the most recent interactions while keeping the total tokens under `short_term_limit`."""
        try:
            with open(self.short_term_file, "r", encoding="utf-8") as f:
                short_term_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        selected_interactions = []
        total_tokens = 0

        # Start from the most recent and work backward until reaching the token limit
        for entry in reversed(short_term_data):
            entry_tokens = len(entry["user"]) + len(entry["assistant"])
            if total_tokens + entry_tokens > self.short_term_limit:
                break  # Stop when exceeding the token limit

            selected_interactions.append(entry)
            total_tokens += entry_tokens

        return list(reversed(selected_interactions))  # Return in chronological order
        
    def retrieve_short_term_formatted(self) -> List[Dict[str, str]]:
        """Return short-term memory entries formatted as messages with roles."""
        raw_history = self.retrieve_short_term()
        formatted = []
        for entry in raw_history:
            if "system" in entry:
                formatted.append({"role": "system", "content": entry["system"]})
            if "user" in entry:
                formatted.append({"role": "user", "content": entry["user"]})
            if "assistant" in entry:
                formatted.append({"role": "assistant", "content": entry["assistant"]})
        return formatted

    def save_long_term(self, extracted_insight: str):
        """
        Save extracted insight to long-term memory.
        """
        with open(self.long_term_file, "r", encoding="utf-8") as f:
            long_term_data = json.load(f)

        long_term_data.append({
            "insight": extracted_insight,
            "timestamp": datetime.now().isoformat()
        })

        with open(self.long_term_file, "w", encoding="utf-8") as f:
            json.dump(long_term_data, f, indent=4)

        self.convert_memory_to_markdown(self.long_term_file, self.long_term_markdown)

    def retrieve_long_term(self) -> List[str]:
        """Retrieve all long-term memory insights."""
        with open(self.long_term_file, "r", encoding="utf-8") as f:
            return [entry["insight"] for entry in json.load(f)]

    def convert_memory_to_markdown(self, json_file: str, md_file: str):
        """Convert JSON memory to a Markdown file."""
        with open(json_file, "r", encoding="utf-8") as f:
            memory_data = json.load(f)

        markdown_content = "# Memory Log\n\n"
        for entry in memory_data:
            markdown_content += f"### {entry['timestamp']}\n"
            if "user" in entry and "assistant" in entry:  # Short-term format
                markdown_content += f"**User:** {entry['user']}\n\n"
                markdown_content += f"**Assistant:** {entry['assistant']}\n\n"
            else:  # Long-term format
                markdown_content += f"{entry['insight']}\n\n"
            markdown_content += "---\n\n"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def store_long_term_memory(self, interactions: int = 3):
        """
        Process the last `interactions` short-term memory entries for long-term storage.
        This ensures frequent knowledge extraction without waiting for the full short-term memory to fill up.
        """
        # Retrieve only the last `interactions` messages
        recent_memories = self.retrieve_short_term()[-interactions:]

        if not recent_memories:
            return  

        oai = Utils.import_file("oai.py")
        if not oai:
            print("Error: OAI module not found.")
            return

        oai_instance = oai.OAI(api_keys_path=None)

        decision_system_prompt = (
            "You are an AI librarian responsible for identifying valuable knowledge in conversations. "
            "Analyze the following interactions and determine if they contain useful, factual insights "
            "that should be stored in long-term memory. Respond with 'Yes' or 'No' only."
        )

        decision_prompt = "Recent Conversation History:\n"
        for idx, memory in enumerate(recent_memories, start=1):
            decision_prompt += (
                f"\nInteraction {idx}:\n"
                f"User: {memory['user']}\n"
                f"Assistant: {memory['assistant']}\n"
            )

        decision_prompt += "\nShould any information from these interactions be stored in long-term memory?\nResponse:"

        decision = oai_instance.chat_completion(
            decision_prompt, 
            system_prompt=decision_system_prompt, 
            temperature=0.2, 
            max_tokens=3
        ).strip().lower()

        if decision == "yes":
            extraction_prompt = (
                "Extract the most valuable factual insight from the following interactions.\n\n"
                "Recent Conversation History:\n"
            )
            for idx, memory in enumerate(recent_memories, start=1):
                extraction_prompt += (
                    f"\nInteraction {idx}:\n"
                    f"User: {memory['user']}\n"
                    f"Assistant: {memory['assistant']}\n"
                )

            extraction_prompt += (
                "\nSummarize the most important insight in a single, factual statement.\nInsight:"
            )

            extracted_insight = oai_instance.chat_completion(
                extraction_prompt, 
                temperature=0.3, 
                max_tokens=50
            ).strip()

            self.save_long_term(extracted_insight)
            
if __name__ == "__main__":
    memory = Memory()
    memory.reboot_memory()

    oai = Utils.import_file("oai.py")
    if not oai:
        print("Error: OAI module not found.")
        exit(1)

    oai_instance = oai.OAI(api_keys_path=None)

    # Generate structured output for 10 PADI Open Water Certification questions
    question_prompt = (
        "Generate a structured JSON list of 10 essential exam questions for a PADI Open Water Certification. "
        "Return the output in JSON format with the key 'questions' containing an array of strings."
    )

    structured_output = oai_instance.structured_output(question_prompt, temperature=0.5, max_tokens=300)

    if not structured_output or "questions" not in structured_output:
        print("Error: Failed to generate valid questions.")
        exit(1)

    questions = structured_output["questions"]

    print("\n=== Generated PADI Open Water Exam Questions ===")
    for idx, question in enumerate(questions, start=1):
        print(f"{idx}. {question}")

    system_message = "PADI Open Water Certification Exam Assistant"

    # Process each question and store short-term memory.
    # Long-term memory storage is automatically triggered every 3 interactions.
    for index, question in enumerate(questions, start=1):
        response = oai_instance.chat_completion(question, temperature=0.5, max_tokens=300).strip()
        memory.save_short_term(system_message, question, response)

        print(f"\n=== Interaction {index} ===")
        print(f"Q: {question}")
        print(f"A: {response}")

    print("\n=== Long-Term Memory Insights ===")
    for insight in memory.retrieve_long_term():
        print(f"- {insight}")
