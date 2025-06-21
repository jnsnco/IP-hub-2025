from os import environ
from typing import List
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, FunctionTool, BaseTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

from flask import Flask, request, make_response
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = environ["OPENAI_API_KEY"]

app = Flask(__name__)


def main():
    llm = OpenAI(
        model="gpt-4",
    )

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/internal_db"
        )
        internal_db = load_index_from_storage(storage_context)

    except:
        # load data
        internal_docs = SimpleDirectoryReader(input_dir="./internal_docs").load_data()

        # build index
        internal_db = VectorStoreIndex.from_documents(internal_docs, show_progress=True)

        # persist index
        internal_db.storage_context.persist(persist_dir="./storage/internal_db")

    internal_db_engine = internal_db.as_query_engine(similarity_top_k=3, llm=llm)

    internal_db_tool = QueryEngineTool(
        query_engine=internal_db_engine,
        metadata=ToolMetadata(
            name="internal_db",
            description=(
                "Provides internal documents on various patents. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    )

    # def internal_db_patent_number_fn(patent_number: str) -> str:
    #     if not str.isalnum(patent_number):
    #         return "ERROR: patent or publication number invalid."
    #     with open(f"./internal_docs/{patent_number}.md") as file:
    #         return file.read()

    # internal_db_patent_number_tool = FunctionTool.from_defaults(
    #     fn=internal_db_patent_number_fn,
    #     name="search_by_number",
    #     description="Searches for a patent by its patent or publication number. The input should only contain the patent or publication number. The response will be the markdown-formatted data on the patent.",
    # )
    # query_engine_tools: list[BaseTool] = [
    #     QueryEngineTool(
    #         query_engine=internal_db_engine,
    #         metadata=ToolMetadata(
    #             name="internal_db",
    #             description=(
    #                 "Provides internal documents on various patents. "
    #                 "Use a detailed plain text question as input to the tool."
    #             ),
    #         ),
    #     ),
    #     # FunctionTool.from_defaults(
    #     #     async_fn=query_external_db,
    #     #     tool_metadata=ToolMetadata(
    #     #         name="external_db_search",
    #     #         description=( # TODO: better description
    #     #             "Allows for searching the US patent database. "
    #     #             "Use a search query as input to the tool."
    #     #         ),
    #     #     ),
    #     # ),
    # ]

    @app.route("/", methods=["POST"])
    def main_route():
        json = request.get_json()
        if "query" not in json:
            return make_response({"error": "no query specified"}, 400)

        agent = ReActAgent.from_tools(
            [
                internal_db_tool,
                # internal_db_patent_number_tool,
            ],
            llm=llm,
            verbose=True,
            max_turns=10,
        )
        response = agent.chat(
            """Return your FINAL answer in markdown format. Include an overall summary and a list of relevant
    or related patents to the user's query.
    For each patent, include the patent number (or publication number if not available),
    the title, and a description containing a brief summary and its relevance to the user's query.
    This can include aspects of the previous conversation history.

    If there are interesting relevant details found in the patents you discover,
    investigate further by asking more questions to tools in order to find more relevant patents.

    Example:
    ```
    # Summary
    This is where the summary should go. Include details on the related patent landscape
    and make recommendations for which patents the user might want to look at first.

    # Patents
    ## US1234567 - THIS IS THE PATENT TITLE
    This is the patent summary. 

    ## US643216 - THIS IS THE SECOND PATENT TITLE
    This is the second patent's summary.
    ```"""
            + json["query"]
        )
        return make_response({"response": str(response)}, 200)


main()
