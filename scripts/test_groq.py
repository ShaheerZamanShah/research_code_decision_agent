import asyncio
import traceback
from langchain_openai import ChatOpenAI
try:
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    from langchain_core.messages import HumanMessage, SystemMessage
from app.config.settings import settings

async def main():
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.llm_model,
        temperature=0.1,
    )
    try:
        messages = [SystemMessage(content="Test"), HumanMessage(content="Say hello")]
        resp = await llm.ainvoke(messages)
        print('RESPONSE:', resp)
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
