from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import qdrant_client

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, query: str) -> str:
        # Implementation goes here
        search_result = qdrant_client.query(
        collection_name="aparavi_knowledge",
        query_text=query)

        results = []
        # Extract the list of ScoredPoint objects from the tuple
        for point in search_result:
            result = {
                "metadata": point[1][0].payload.get("metadata", {}),
                "context": point[1][0].payload.get("text", ""),
                "distance": point[1][0].score,
            }
            results.append(result)

        return search_result
