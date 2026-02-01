from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from ingest.extractors.base import BaseExtractor
from ingest.models import Fact, FactCategory, FactSource, MessageRole, Prompt, PromptBatch


class LLMExtractor(BaseExtractor):
    """LLM-based fact extractor using LangChain

    This extractor uses a language model to understand context and
    extract nuanced facts from text. It's slower than rules but
    provides higher accuracy and can detect implicit facts.

    The LLM is configured to output structured JSON facts with
    categories and confidence scores.
    """

    SYSTEM_PROMPT = """You are an expert at extracting facts, insights, and information from text.
Your task is to analyze the given text and extract important facts about the person speaking.

Extract facts in the following categories:
- preference: Likes, dislikes, preferences (e.g., "likes ice cream", "hates waking up early")
- event: Things that happened (e.g., "went to the park yesterday", "had dinner with friends")
- relationship: People and connections (e.g., "John is my brother", "broke up with girlfriend")
- belief: Opinions and beliefs (e.g., "believes AI will change the world", "thinks politics is corrupt")
- fact: Objective statements (e.g., "lives in New York", "works as a software engineer")
- location: Places mentioned (e.g., "San Francisco", "the office", "home")
- goal: Intentions and goals (e.g., "wants to learn Spanish", "planning to travel to Japan")
- emotion: Emotional states (e.g., "feeling excited", "worried about the presentation")

For each fact extracted:
1. Keep it concise and factual
2. Assign a confidence score (0.5-1.0) based on how certain you are
3. Only extract facts that are meaningful and worth remembering
4. Avoid trivial or obvious information

Return your response as a JSON array of objects with these fields:
- content: the fact text
- category: one of the categories listed above
- confidence: a number between 0.5 and 1.0

Example output:
[
  {"content": "likes ice cream", "category": "preference", "confidence": 0.9},
  {"content": "visited Paris last summer", "category": "event", "confidence": 0.85}
]"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """Initialize LLM extractor

        Args:
            model_name: OpenAI model to use (default: gpt-3.5-turbo)
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = JsonOutputParser()
        self.prompt_template = ChatPromptTemplate.from_messages(
            [SystemMessage(content=self.SYSTEM_PROMPT), HumanMessage(content="{text}")]
        )

    def extract(self, prompts: PromptBatch) -> List[Fact]:
        """Extract facts from a batch of prompts

        Processes each prompt individually through the LLM.

        Args:
            prompts: Batch of prompts to process

        Returns:
            List of extracted facts with metadata
        """
        facts = []

        for prompt in prompts.prompts:
            extracted = self._extract_from_prompt(prompt)
            facts.extend(extracted)

        return facts

    def _extract_from_prompt(self, prompt: Prompt) -> List[Fact]:
        """Extract facts from a single prompt

        Args:
            prompt: Single prompt to process

        Returns:
            List of facts extracted from this prompt
        """
        facts = []

        try:
            chain = self.prompt_template | self.llm | self.parser
            result = chain.invoke({"text": prompt.text})

            for item in result:
                fact = Fact(
                    content=item["content"],
                    category=FactCategory(item["category"]),
                    confidence=item["confidence"],
                    source=FactSource.USER
                    if prompt.role == MessageRole.USER
                    else FactSource.AI,
                    source_timestamp=prompt.timestamp,
                    source_id=prompt.id,
                    metadata={"extractor": "llm"},
                )
                facts.append(fact)

        except Exception as e:
            print(f"Error extracting from prompt {prompt.id}: {e}")

        return facts

    def can_handle(self, prompt: Prompt) -> float:
        """Determine confidence of handling this prompt

        LLM extractor can handle any text, but confidence is based
        on text length and complexity. Longer, more complex text
        gets higher confidence.

        Args:
            prompt: Single prompt to evaluate

        Returns:
            Confidence score from 0.0 to 1.0
        """
        word_count = len(prompt.text.split())

        if word_count < 3:
            return 0.3
        elif word_count < 10:
            return 0.5
        elif word_count < 30:
            return 0.7
        else:
            return 0.9
