"""
Prompt templates for LLM interactions.

Provides reusable prompt templates for different use cases.
"""
from typing import Optional, List, Dict, Tuple


class PromptTemplates:
    """
    Collection of prompt templates for various LLM use cases.

    All templates are designed to be customizable and composable.
    """

    # System prompts
    CHAT_SYSTEM_PROMPT = """You are a professional wealth-management customer service assistant at a financial institution. You help users with consultation about wealth-management products, funds, insurance, portfolio queries, risk assessments, and return calculations.

Guidelines:
- Be clear, concise, professional, and friendly
- NEVER promise returns, and NEVER use phrases like "保本" (principal-guaranteed), "稳赚" (sure profit), or "无风险" (risk-free)
- When discussing products, present fees, risk levels, and terms accurately
- Always include risk disclaimers when discussing investment returns: 理财非存款，产品有风险，投资须谨慎
- If you're not sure about something, admit it
- Ask clarifying questions when needed (e.g., risk tolerance level before recommending products)"""

    RAG_SYSTEM_PROMPT = """You are a professional wealth-management customer service assistant with access to relevant context from a financial knowledge base. Answer user questions using the provided context while being clear about what information comes from the context versus your general knowledge.

Guidelines:
- Use the provided context to answer questions when relevant
- Cite sources when referencing specific information from context
- If the context doesn't contain relevant information, say so honestly instead of guessing
- Never promise returns or guarantee investment outcomes; add risk disclaimers when discussing returns
- If the context contradicts your knowledge, mention this discrepancy"""

    # Template methods
    @staticmethod
    def get_chat_system_prompt() -> str:
        """
        Get the standard chat system prompt.

        Returns:
            str: Chat system prompt
        """
        return PromptTemplates.CHAT_SYSTEM_PROMPT

    @staticmethod
    def get_rag_system_prompt() -> str:
        """
        Get the RAG system prompt.

        Returns:
            str: RAG system prompt
        """
        return PromptTemplates.RAG_SYSTEM_PROMPT

    @staticmethod
    def format_chat_prompt(system_prompt: str, **kwargs) -> str:
        """
        Format a chat prompt with variables.

        Args:
            system_prompt: System prompt template with placeholders
            **kwargs: Variables to substitute in template

        Returns:
            str: Formatted prompt
        """
        return system_prompt.format(**kwargs)

    @staticmethod
    def format_rag_prompt(
        query: str,
        retrieved_docs: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format a RAG prompt with query and retrieved documents.

        Args:
            query: User's query
            retrieved_docs: List of retrieved documents with 'content' and 'source' keys
            system_prompt: Optional custom system prompt

        Returns:
            str: Formatted RAG prompt
        """
        if system_prompt is None:
            system_prompt = PromptTemplates.RAG_SYSTEM_PROMPT

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            context_parts.append(f"[Document {i} - {source}]\n{content}\n")

        context = "\n".join(context_parts) if context_parts else "No relevant documents found."

        # Format the prompt
        prompt = f"""{system_prompt}

Relevant Context:
{context}

User Query: {query}

Answer:"""

        return prompt

    @staticmethod
    def get_summarization_prompt(text: str, max_length: Optional[int] = None) -> str:
        """
        Get a summarization prompt.

        Args:
            text: Text to summarize
            max_length: Optional maximum length for summary

        Returns:
            str: Summarization prompt
        """
        if max_length:
            instruction = f"Summarize the following text in {max_length} characters or less."
        else:
            instruction = "Summarize the following text concisely."

        return f"""{instruction}

Text:
{text}

Summary:"""

    @staticmethod
    def get_intent_detection_prompt(
        query: str,
        intents: List[str],
    ) -> str:
        """
        Get an intent detection prompt.

        Args:
            query: User query to classify
            intents: List of possible intents

        Returns:
            str: Intent detection prompt
        """
        intents_str = ", ".join(intents)

        return f"""Classify the user's query into one of the following intents: {intents_str}

User Query: {query}

Respond with only the intent name, nothing else.

Intent:"""

    @staticmethod
    def format_template(template: str, **kwargs) -> str:
        """
        Format a custom template with variables.

        Args:
            template: Template string with placeholders
            **kwargs: Variables to substitute

        Returns:
            str: Formatted template
        """
        return template.format(**kwargs)

    @staticmethod
    def get_memory_aware_prompt(
        conversation_summary: str,
        current_message: str,
    ) -> str:
        """
        Get a memory-aware prompt with conversation context.

        Args:
            conversation_summary: Summary of previous conversation
            current_message: Current user message

        Returns:
            str: Memory-aware prompt
        """
        return f"""{PromptTemplates.CHAT_SYSTEM_PROMPT}

Conversation Summary:
{conversation_summary}

Current Message: {current_message}

Response:"""

    @staticmethod
    def get_code_generation_prompt(
        requirements: str,
        language: str = "Python",
    ) -> str:
        """
        Get a code generation prompt.

        Args:
            requirements: Description of code requirements
            language: Programming language

        Returns:
            str: Code generation prompt
        """
        return f"""You are an expert programmer. Write {language} code to fulfill the following requirements:

{requirements}

Provide only the code with brief comments. Include error handling and best practices.

Code:"""

    @staticmethod
    def get_few_shot_prompt(
        examples: List[Dict[str, str]],
        test_input: str,
        task_description: Optional[str] = None,
    ) -> str:
        """
        Get a few-shot learning prompt.

        Args:
            examples: List of example dictionaries with 'input' and 'output'
            test_input: Input to process
            task_description: Optional description of the task

        Returns:
            str: Few-shot prompt
        """
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Input: {example['input']}\n"
            examples_text += f"Output: {example['output']}\n\n"

        if task_description:
            prompt = f"""{task_description}

{examples_text}Input: {test_input}
Output:"""
        else:
            prompt = f"""{examples_text}Input: {test_input}
Output:"""

        return prompt

    @staticmethod
    def get_chain_of_thought_prompt(problem: str) -> str:
        """
        Get a chain-of-thought reasoning prompt.

        Args:
            problem: Problem statement to solve

        Returns:
            str: Chain-of-thought prompt
        """
        return f"""Think step by step to solve the following problem. Show your reasoning process.

Problem: {problem}

Reasoning:"""

    @staticmethod
    def get_multi_turn_prompt(
        turns: List[Tuple[str, str]],
        current_user_message: str,
    ) -> str:
        """
        Get a multi-turn conversation prompt.

        Args:
            turns: List of (user_message, assistant_response) tuples
            current_user_message: Current user message

        Returns:
            str: Multi-turn conversation prompt
        """
        conversation = ""
        for i, (user_msg, assistant_msg) in enumerate(turns, 1):
            conversation += f"Turn {i}:\n"
            conversation += f"User: {user_msg}\n"
            conversation += f"Assistant: {assistant_msg}\n\n"

        conversation += f"User: {current_user_message}\n"
        conversation += "Assistant:"

        return f"""{PromptTemplates.CHAT_SYSTEM_PROMPT}

{conversation}"""
