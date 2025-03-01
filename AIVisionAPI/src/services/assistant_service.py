from typing import Iterator, List, Literal

# from phi.agent.agent import Agent
from agno.agent.agent import Agent
from agno.run.response import RunResponse
from agno.memory.agent import AgentMemory
from textwrap import dedent

# from phi.run.response import RunResponse
from agno.models.message import Message as PhiMessage
from agno.models.openrouter.openrouter import OpenRouter
from agno.tools.tavily import TavilyTools

from src.models.chat import Message

from src.services.todos.todos_service_sync import TodosServiceSync

import os


class AssistantService:
    def __init__(self):
        todos_service = TodosServiceSync("sqlite:///./prisma/dev.db")

        self.agent = Agent(
            model=OpenRouter(
                id=os.getenv("BACKEND_MODEL_NAME"),
                api_key=os.getenv("BACKEND_API_KEY"),
            ),
            # markdown=True,
            debug_mode=True,
            telemetry=False,
            tools=[
                TavilyTools(
                    api_key=os.getenv("TAVILY_API_KEY"),
                    # include_answer=True,
                    # search_depth="basic",
                    # use_search_context=True,
                )
            ],
            description=dedent(
                """
            # ROLE
            You are a multimodal AI assistant specialized in assisting blind users by providing comprehensive answers using text and vision-based modalities.

            # ADDITIONAL INFORMATION
            ## Remember important details about users and reference them naturally, while always respecting privacy and data security.
            ## Maintain a warm, empathetic, and positive tone, using clear, accessible language that is easy to understand.
            ## When appropriate, refer back to previous conversations and memories to offer personalized support, but only if it enhances the current interaction.
            ## Always be truthful about what you remember or do not remember, and clearly state any limitations in your knowledge.
            ## Use all available tools to provide accurate, up-to-date information and detailed descriptions of visual content when necessary.
            ## If you are unsure about an answer, communicate your uncertainty clearly and offer alternative suggestions or guidance.
            ## Ensure that every response is designed with accessibility in mind, offering thorough explanations and descriptions to support users who rely on non-visual information.
            ## Provide helpful suggestions and guidance tailored to the unique needs of blind users.

            # TOOLS
            ## Tavily: Use this tool to search for real-time information and answer queries accurately, including providing detailed descriptions for visual content when applicable.
            """
            ),
            # prevent_hallucinations=True,
            # reasoning=True,
            add_datetime_to_instructions=True,
            # show_tool_calls=True,
        )

    def run_conversation(
        self,
        messages: List[Message] = [],
        stream: Literal[True] | Literal[False] = False,
    ) -> RunResponse | Iterator[RunResponse]:
        # convert our messages to phi messages
        phi_messages = [
            PhiMessage(role=msg.role, content=msg.content) for msg in messages
        ]

        return self.agent.run(messages=phi_messages, stream=stream)
