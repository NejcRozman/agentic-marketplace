from setuptools import setup, find_packages

setup(
    name="agentic-marketplace",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "web3>=6.0.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.9.0",
        "pyyaml>=6.0",
        "eth-account>=0.10.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-google-genai>=0.0.6",
        "langgraph>=0.0.20",
    ],
)
