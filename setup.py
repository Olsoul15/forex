from setuptools import setup, find_packages

# Define core runtime dependencies explicitly
# These are the minimum packages required for forex_ai to be imported and run its core API/library logic.
core_requirements = [
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "pydantic>=2.5.3",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.1",
    "python-multipart>=0.0.9",
    "jinja2>=3.1.2",  # If API serves HTML templates
    "websockets>=12.0",
    "httpx>=0.27.0",
    "requests>=2.31.0",
    "aiohttp>=3.9.3",
    "python-jose>=3.3.0",
    "pyjwt>=2.6.0",
    "passlib>=1.7.4",
    "bcrypt>=4.0.1",
    "sqlalchemy>=2.0.28",
    "alembic>=1.13.1",
    "psycopg2-binary>=2.9.9",
    "redis>=5.0.2",
    "asyncpg>=0.29.0",
    # "databases>=0.8.0", # Consider if this is core or for specific db integrations
    # "ormar>=0.12.2",    # Consider if this is core or for specific db integrations
    "supabase>=2.13.0",
    "qdrant-client>=1.1.1",  # If vector search is core to the API/library
    "pgvector>=0.2.5",  # If vector search with PG is core
    "openai>=1.0.0",  # Assuming OpenAI is a core LLM provider
    "langchain>=0.1.12",  # Core Langchain components
    "langchain-community>=0.0.28",
    "langchain-core>=0.1.29",
    "tiktoken>=0.5.2",
    "transformers>=4.39.0",  # If core models rely on HuggingFace transformers
    "torch>=2.0.0",  # If core models rely on PyTorch
    # "tensorflow>=2.12.0", # Only if TF is a non-optional core dependency
    # "anthropic>=0.20.0",  # Add if Anthropic is a core, non-optional LLM provider
    # "langgraph>=0.0.32",
    # "pydantic-ai>=0.1.0",
    # "langchainhub>=0.1.14",
    # "azure-ai-openai>=1.0.0",
    "numpy>=1.26.4",
    "scipy>=1.13.0",  # Often a core data processing dep
    "pandas>=2.2.2",
    # "scikit-learn>=1.4.1", # If core, not just for specific models/features
    # "statsmodels>=0.14.1",
    # "polars>=0.20.10",
    # "pyarrow>=14.0.2",
    # "yfinance>=0.2.36", # If core data source
    "TA-Lib==0.4.30",  # Version from Dockerfile, ensure consistency
    # "arch>=5.3.0",
    "eventlet>=0.33.3",
    "pyyaml>=6.0.1",
    "loguru>=0.7.2",
    "tenacity>=8.2.3",
    "pytz>=2024.1",
    "arrow>=1.3.0",
    "click>=8.1.7",  # For CLI
    # "tqdm>=4.66.2", # More of a dev/scripting utility
    "orjson>=3.9.15",  # Faster JSON parsing
    "uuid>=1.30",
]

# Development, testing, and other optional dependencies
extras_require = {
    "dev": [
        "black>=24.2.0",
        "isort>=5.13.2",
        "mypy>=1.7.1",
        "flake8>=7.0.0",
        "pre-commit>=3.6.2",
        "docker>=7.0.0",
        "tqdm>=4.66.2",  # Moved here
    ],
    "test": [
        "pytest>=8.0.2",
        "pytest-asyncio>=0.23.4",
        "pytest-cov>=4.1.0",
        "qodo-cover>=1.0.0",  # Or direct git link if needed
        "playwright>=1.43.0",
    ],
    "ui": [
        "streamlit>=1.32.0",
    ],
    "visualization": [
        "plotly>=5.15.0",
        "matplotlib>=3.8.3",
        "seaborn>=0.13.2",
    ],
    # Add other optional groups if forex_ai supports multiple backends/integrations
    # that are not always needed, e.g.:
    "tensorflow_deps": ["tensorflow>=2.12.0"],
    "anthropic_deps": ["anthropic>=0.20.0"],
    "full_ml": [  # Example of a group that pulls in more ML libs
        "tensorflow>=2.12.0",
        "anthropic>=0.20.0",
        "langgraph>=0.0.32",
        "pydantic-ai>=0.1.0",
        "langchainhub>=0.1.14",
        "azure-ai-openai>=1.0.0",
        "scikit-learn>=1.4.1",
        "statsmodels>=0.14.1",
        "polars>=0.20.10",
        "pyarrow>=14.0.2",
        "yfinance>=0.2.36",
        "arch>=5.3.0",
    ],
    "all_db_drivers": [
        "databases>=0.8.0",
        "ormar>=0.12.2",
    ],
}

# Combine all extras for a 'full' install option if desired
extras_require["full"] = sum(extras_require.values(), [])

setup(
    name="forex_ai",
    version="0.1.0",
    description="AI-Powered Forex Trading System",
    author="Forex AI Team",
    author_email="info@forexai.example.com",
    url="https://github.com/yourusername/forex_ai",
    packages=find_packages(),
    include_package_data=True,  # Important for MANIFEST.in or other data files
    install_requires=core_requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "forex-ai=forex_ai.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
)
