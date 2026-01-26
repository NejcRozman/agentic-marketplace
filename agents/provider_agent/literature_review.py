"""Literature Review Agent for the agentic marketplace."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Annotated, Sequence, Literal, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

try:
    from ..config import config
except ImportError:
    from config import Config
    config = Config()

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class LiteratureReviewAgent:
    """
    Literature Review Agent that performs RAG-based literature reviews.
    
    This agent:
    - Loads PDFs provided by clients
    - Creates vector database for semantic search
    - Performs literature reviews based on prompts
    - Provides properly cited responses
    """
    
    # System prompts for different quality tiers
    SYSTEM_PROMPTS = {
        "detailed": """You are a literature review assistant specialized in analyzing research papers.

Your responsibilities:
1. Answer questions about research papers accurately and comprehensively
2. ALWAYS cite your sources using the format [Author, 'Title'] after each statement
3. When comparing papers, clearly indicate which paper each finding comes from
4. Summarize papers with proper citations
5. Generate literature reviews with consistent citation format
6. Provide detailed analysis with multiple perspectives
7. Include methodological details and context

Citation Rules:
- Every factual statement must be followed by a citation
- Use the exact citation format provided by the search_literature tool
- When multiple sources support a statement, list all citations
- Be precise about which paper each piece of information comes from
- Cross-reference related findings across papers

Quality Standards:
- Provide comprehensive answers with depth
- Include nuanced analysis
- Consider limitations and alternative interpretations
- Structure responses with clear sections

Maintain professional academic tone and provide detailed, well-structured responses.""",

        "standard": """You are a literature review assistant specialized in analyzing research papers.

Your responsibilities:
1. Answer questions about research papers accurately and comprehensively
2. ALWAYS cite your sources using the format [Author, 'Title'] after each statement
3. When comparing papers, clearly indicate which paper each finding comes from
4. Summarize papers with proper citations
5. Generate literature reviews with consistent citation format

Citation Rules:
- Every factual statement must be followed by a citation
- Use the exact citation format provided by the search_literature tool
- When multiple sources support a statement, list all citations
- Be precise about which paper each piece of information comes from

Maintain professional academic tone and provide detailed, well-structured responses.""",

        "minimal": """You are a literature review assistant.

Your tasks:
1. Answer questions about research papers
2. Cite sources using [Author, 'Title'] format
3. Provide summaries

Keep responses concise and direct."""
    }
    
    def __init__(
        self,
        agent_id: str,
        workspace_dir: Optional[str] = None
    ):
        """
        Initialize the Literature Review Agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            workspace_dir: Directory for storing RAG databases (optional)
        """
        self.agent_id = agent_id
        self.workspace_dir = Path(workspace_dir or config.workspace_dir) / agent_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM and embeddings
        if not config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Use config-based parameters (supports quality profiles)
        self.llm = ChatOpenAI(
            model="xiaomi/mimo-v2-flash:free",
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            temperature=config.rag_temperature
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url
        )
        
        # RAG parameters from config (quality profile)
        self.chunk_size = config.rag_chunk_size
        self.chunk_overlap = config.rag_chunk_overlap
        self.retrieval_k = config.rag_retrieval_k
        self.system_prompt_type = config.rag_system_prompt_type
        
        # RAG components (initialized when PDFs are loaded)
        self.vectorstore = None
        self.retriever = None
        self.current_pdf_directory = None
        
        # LangGraph workflow
        self.graph = None
        self.tools = []
        
        logger.info(f"Initialized LiteratureReviewAgent: {agent_id} (quality: {config.quality_profile}, "
                   f"temp={config.rag_temperature}, k={config.rag_retrieval_k})")
    
    def _extract_paper_metadata(self, document: Any) -> Dict[str, str]:
        """Extract paper title and first author from filename format: Author-Title.pdf"""
        metadata = document.metadata
        source_file = os.path.basename(metadata.get('source', ''))
        
        # Remove .pdf extension
        filename = source_file.replace('.pdf', '')
        
        # Split by first hyphen to get author and title
        if '-' in filename:
            parts = filename.split('-', 1)
            first_author = parts[0].strip()
            title = parts[1].strip().replace('_', ' ')
        else:
            first_author = 'Unknown Author'
            title = filename.replace('_', ' ')
        
        return {
            'title': title,
            'first_author': first_author,
            'source_file': source_file
        }
    
    def load_pdfs(self, pdf_directory: str, force_rebuild: bool = False) -> bool:
        """
        Load PDFs and create/update vector database.
        
        Args:
            pdf_directory: Path to directory containing PDF files
            force_rebuild: Force rebuild of vector database even if it exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_dir = Path(pdf_directory)
            if not pdf_dir.exists():
                logger.error(f"PDF directory does not exist: {pdf_directory}")
                return False
            
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                logger.error(f"No PDF files found in {pdf_directory}")
                return False
            
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            # Set up persist directory for this PDF collection
            collection_name = f"literature_review_{self.agent_id}"
            persist_directory = self.workspace_dir / "RAGfiles" / pdf_dir.name
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Check if we need to rebuild
            needs_rebuild = force_rebuild or not (persist_directory / "chroma.sqlite3").exists()
            
            if needs_rebuild:
                logger.info("Building vector database from PDFs...")
                
                # Load PDFs
                loader = PyPDFDirectoryLoader(str(pdf_dir))
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} pages from PDFs")
                
                # Extract and add metadata
                for doc in documents:
                    paper_metadata = self._extract_paper_metadata(doc)
                    doc.metadata.update(paper_metadata)
                
                # Split documents using config-based parameters
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                splits = text_splitter.split_documents(documents)
                logger.info(f"Created {len(splits)} text chunks")
                
                # Create vector database
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    collection_name=collection_name,
                    persist_directory=str(persist_directory)
                )
                logger.info("Vector database created successfully")
            else:
                logger.info("Loading existing vector database...")
                self.vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(persist_directory)
                )
                logger.info("Loaded existing vector database")
            
            # Create retriever using config-based parameters
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_k}
            )
            
            self.current_pdf_directory = pdf_directory
            
            # Rebuild LangGraph with the search tool
            self._build_graph()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}", exc_info=True)
            return False
    
    def _create_search_tool(self):
        """Create the search_literature tool with access to the retriever."""
        retriever = self.retriever
        
        @tool
        def search_literature(query: str) -> str:
            """
            Search through research papers to find relevant information.
            Returns content with citations indicating which paper each piece of information comes from.
            Use this tool to answer questions about research papers, find specific information,
            compare findings, or gather information for literature reviews.
            """
            docs = retriever.invoke(query)
            
            if not docs:
                return "No relevant information found in the research papers."
            
            # Format results with citations
            results = []
            for i, doc in enumerate(docs, 1):
                title = doc.metadata.get('title', 'Unknown Title')
                author = doc.metadata.get('first_author', 'Unknown Author')
                
                citation = f"[{author}, '{title}']"
                content = doc.page_content.strip()
                
                results.append(f"Source {i} {citation}:\n{content}\n")
            
            return "\n---\n".join(results)
        
        return search_literature
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        if not self.retriever:
            logger.warning("Cannot build graph without retriever. Load PDFs first.")
            return
        
        # Create tools
        search_tool = self._create_search_tool()
        self.tools = [search_tool]
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Define graph nodes
        def call_llm(state: AgentState) -> AgentState:
            """Call the LLM with tools and system prompt."""
            messages = state['messages']
            
            # Add system prompt if not already present (use quality-specific prompt)
            if not messages or not isinstance(messages[0], SystemMessage):
                system_prompt = self.SYSTEM_PROMPTS.get(self.system_prompt_type, self.SYSTEM_PROMPTS["standard"])
                messages = [SystemMessage(content=system_prompt)] + messages
            
            response = llm_with_tools.invoke(messages)
            return {'messages': [response]}
        
        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """Determine if we should use tools or end."""
            last_message = state['messages'][-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return "end"
        
        def call_tools(state: AgentState) -> AgentState:
            """Execute tool calls and return results."""
            last_message = state['messages'][-1]
            tool_calls = last_message.tool_calls
            
            tools_dict = {tool.name: tool for tool in self.tools}
            
            results = []
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                if tool_name not in tools_dict:
                    result = f"Error: Tool {tool_name} not found"
                else:
                    tool = tools_dict[tool_name]
                    result = tool.invoke(tool_args)
                
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id'],
                    name=tool_name
                )
                results.append(tool_message)
            
            return {'messages': results}
        
        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_llm)
        workflow.add_node("tools", call_tools)
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")
        
        self.graph = workflow.compile()
        logger.info("LangGraph workflow built successfully")
    
    def perform_review(
        self,
        pdf_directory: str,
        prompts: List[str],
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Perform a literature review based on provided PDFs and prompts.
        
        Args:
            pdf_directory: Path to directory containing PDF files
            prompts: List of questions/prompts for the literature review
            force_rebuild: Force rebuild of vector database
            
        Returns:
            Dictionary containing review results:
            {
                "success": bool,
                "agent_id": str,
                "pdf_directory": str,
                "prompts": List[str],
                "responses": List[Dict[str, str]],  # [{"prompt": "...", "response": "..."}]
                "error": Optional[str]
            }
        """
        result = {
            "success": False,
            "agent_id": self.agent_id,
            "pdf_directory": pdf_directory,
            "prompts": prompts,
            "responses": [],
            "error": None
        }
        
        try:
            # Load PDFs if not already loaded or if directory changed
            if self.current_pdf_directory != pdf_directory or force_rebuild:
                if not self.load_pdfs(pdf_directory, force_rebuild):
                    result["error"] = "Failed to load PDFs"
                    return result
            
            # IMPORTANT: Rebuild graph for each service to prevent context accumulation
            self._build_graph()
            
            if not self.graph:
                result["error"] = "Graph not initialized. Load PDFs first."
                return result
            
            # Process each prompt
            for prompt in prompts:
                logger.info(f"Processing prompt: {prompt[:100]}...")
                
                # Create initial state with the prompt
                initial_state = {"messages": [HumanMessage(content=prompt)]}
                
                # Run the graph
                graph_result = self.graph.invoke(initial_state)
                
                # Extract the final response
                final_message = graph_result['messages'][-1]
                
                # Handle response content
                if isinstance(final_message.content, list):
                    response_text = final_message.content[0].get("text", str(final_message.content))
                else:
                    response_text = final_message.content
                
                result["responses"].append({
                    "prompt": prompt,
                    "response": response_text
                })
            
            result["success"] = True
            logger.info(f"Literature review completed successfully with {len(prompts)} prompts")
            
        except Exception as e:
            logger.error(f"Error performing review: {e}", exc_info=True)
            result["error"] = str(e)
        
        return result
