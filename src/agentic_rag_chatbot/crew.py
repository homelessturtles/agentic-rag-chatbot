from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
#from crewai_tools import QdrantVectorSearchTool
import os
from dotenv import load_dotenv
from openai import OpenAI
from agentic_rag_chatbot.tools.custom_tool import QdrantVectorSearchTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

load_dotenv()

@CrewBase
class AgenticRagChatbot():
    """AgenticRagChatbot crew"""

    vector_search_tool = QdrantVectorSearchTool(
        collection_name="aparavi_knowledge",
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY'),
    )

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[self.vector_search_tool],
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'],
            verbose=True
        )

    @agent
    def coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['coordinator'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task']
        )
    
    @task
    def coordination_task(self) -> Task:
        return Task(
            config=self.tasks_config['coordination_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AgenticRagChatbot crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
