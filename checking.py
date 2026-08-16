# КРОК 1 (продовження). Імпорти та налаштування
import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict, Optional
from operator import add as op_add

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool, StructuredTool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_chroma import Chroma

import numexpr as ne

# Налаштування логування
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

# Завантаження API-ключа
load_dotenv()
assert os.getenv(
    "OPENAI_API_KEY"), "Встановіть OPENAI_API_KEY у .env або змінних середовища"

# Ініціалізація моделі
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    max_tokens=1024,
    timeout=30,
)
print("✅ Середовище підготовлено. Модель:", llm.model)
