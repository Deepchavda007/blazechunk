"""Adapters that plug blazechunk into other frameworks.

Each submodule imports its target framework at load time, so import only the one
you need — and install the matching extra:

    pip install "blazechunk[langchain]"   # blazechunk.integrations.langchain
    pip install "blazechunk[agno]"        # blazechunk.integrations.agno

This package itself imports nothing framework-specific, so importing
``blazechunk.integrations`` never pulls in LangChain or Agno.
"""
