#!/usr/bin/env python
import sys
import os
import asyncio

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from langchain_openai import ChatOpenAI
try:
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings

async def test():
    print(f"API Key (first 20 chars): {settings.openai_api_key[:20]}...")
    print(f"Base URL: {settings.openai_base_url}")
    print(f"Model: {settings.llm_model}")
    
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.llm_model,
        temperature=0.1,
    )
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say 'hello' in one word."),
        ]
        print("\nCalling LLM...")
        resp = await llm.ainvoke(messages)
        print("\nSUCCESS! Response:")
        print(resp)
    except Exception as e:
        print("\nERROR:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
