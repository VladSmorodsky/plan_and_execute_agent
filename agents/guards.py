"""Захисти: ліміти прогону (кроки, час, токени, повтори) і політика ризику на рівні виклику."""

import json
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from pydantic import ValidationError

from agents.react import SYSTEM_PROMPT, llm_with_tools, tool_node
from core.schemas import RISK_ORDER, RiskLevel
from core.tools import ALL_TOOLS, READ_TOOLS

_ARGS_SCHEMAS = {tool.name: tool.args_schema for tool in READ_TOOLS}

MAX_STEPS = 8
TIMEOUT_SEC = 60
MAX_TOKENS = 20_000


class GuardedState(TypedDict):
    """
    Стан, який читають захисти.

    `messages` несе редьюсер `add_messages`, тож обидва вузли до нього
    дописують. Решта полів без анотації, тому перезаписуються — саме це й
    потрібно лічильнику, і пише їх лише вузол агента.

    Кожен захист читає стан і більше нічого. Саме це робить їх
    демонстровними: `fresh_state` може подати графу вже вичерпаний лічильник
    або час старту в минулому, і захист зобов'язаний спрацювати. Захист, який
    можна перевірити тільки в бою, — це захист, якого ніхто не перевіряв.
    """
    messages: Annotated[list, add_messages]
    step_count: int
    start_time: float
    total_tokens: int
    last_tool_calls: list


def _stop(reason: str, step: int, **keep: Any) -> dict:
    """Завершити прогін звичайною відповіддю, а не винятком."""
    return {"messages": [AIMessage(content=reason)], "step_count": step, **keep}


def _fingerprint(response: AIMessage) -> list:
    """
    Що означає «той самий виклик ще раз».

    Виклик — це його назва плюс аргументи, і обидві нормалізації тут існують
    тому, що один виклик має багато написань. `sort_keys=True` розбирається з
    порядком: {"room": "kitchen", "name": null}, записаний навпаки, — це один
    виклик і два різні рядки. Проганяння аргументів через власну args_schema
    інструмента розбирається зі значеннями за замовчуванням: модель зазвичай
    пропускає `name` зовсім, а іноді шле його як null, і це той самий запит,
    який інакше виглядав би як два різні.

    Аргументи, що не пройшли валідацію, беруться як є. Це не той повтор, який
    варто ловити: ToolNode саме зараз віддасть моделі помилку валідації, і
    наступний виклик буде виправленим.
    """
    fingerprints = []
    for call in (response.tool_calls or []):
        args = call["args"]
        schema = _ARGS_SCHEMAS.get(call["name"])
        if schema is not None:
            try:
                args = schema(**args).model_dump(mode="json")
            except ValidationError:
                pass
        fingerprints.append((call["name"], json.dumps(args, sort_keys=True)))
    return fingerprints


def guarded_agent_node(state: GuardedState) -> dict:
    """
    Викликати модель, якщо жоден із захистів не каже, що прогін закінчено.

    Дешеві перевірки стоять до виклику моделі: коли бюджет вичерпано, платити
    за ще один запит немає сенсу. Повтор можна перевірити лише після виклику,
    бо щоб побачити, що модель повторюється, потрібна її наступна відповідь.

    `start_time` протягується наскрізь навмисно. Перерахунок його щовитка на
    щасливому шляху непомітний — нічого не падає, агент відповідає, а таймаут
    просто мертвий, бо витрачений час обнуляється на кожній ітерації.
    """
    step = state.get("step_count", 0) + 1
    start = state.get("start_time") or time.time()
    tokens = state.get("total_tokens", 0)
    elapsed = time.time() - start

    if step > MAX_STEPS:
        return _stop(f"⚠️ Досягнуто ліміт у {MAX_STEPS} кроків. "
                     f"Відповідаю з того, що вже зібрав.", step)

    if elapsed > TIMEOUT_SEC:
        return _stop(f"⚠️ Перевищено таймаут ({TIMEOUT_SEC}с, минуло "
                     f"{elapsed:.0f}с). Завершую з наявними даними.", step)

    if tokens > MAX_TOKENS:
        return _stop(f"⚠️ Вичерпано бюджет у {MAX_TOKENS} токенів "
                     f"(витрачено {tokens}). Завершую.", step)

    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)

    called = _fingerprint(response)
    if called and called == state.get("last_tool_calls", []):
        return _stop("⚠️ Агент повторює той самий виклик. Завершую цикл.", step,
                     start_time=start, total_tokens=tokens)

    usage = response.usage_metadata or {}
    return {
        "messages": [response],
        "step_count": step,
        "start_time": start,
        "total_tokens": tokens + usage.get("total_tokens", 0),
        "last_tool_calls": called,
    }


guarded_graph = StateGraph(GuardedState)
guarded_graph.add_node("agent", guarded_agent_node)
guarded_graph.add_node("tools", tool_node)

guarded_graph.add_edge(START, "agent")
guarded_graph.add_conditional_edges("agent", tools_condition,
                                    {"tools": "tools", END: END})
guarded_graph.add_edge("tools", "agent")

guarded_agent = guarded_graph.compile()


TOOL_RISK: dict[str, RiskLevel] = {
    tool.name: (tool.extras or {}).get("risk", "risky") for tool in ALL_TOOLS
}

APPROVAL_REQUIRED: frozenset[RiskLevel] = frozenset({"risky"})


def risk_of(tool_name: str) -> RiskLevel:
    """
    Наскільки небезпечний один інструмент; за замовчуванням — найгірше.

    Невідомий або непозначений інструмент навмисно повертається як "risky".
    Ціна поблажливого замовчування — дія, якої ніхто не затверджував; ціна
    цього — зайве питання про підтвердження.
    """
    return TOOL_RISK.get(tool_name, "risky")


def call_risk(message: AIMessage) -> RiskLevel:
    """
    Ризик цілого ходу моделі: найгірший серед викликів, які вона просить.

    Одне повідомлення може нести кілька викликів, а граф маршрутизує
    повідомлення, не виклик. Саме максимум не дає ризиковому виклику
    проїхати в непідконтрольний вузол поруч із трьома нешкідливими.
    """
    levels = [risk_of(call["name"]) for call in (message.tool_calls or [])]
    return max(levels, key=RISK_ORDER.index) if levels else "safe"


def needs_approval(message: AIMessage) -> bool:
    """Чи має людина подивитись на цей хід до того, як він виконається."""
    return call_risk(message) in APPROVAL_REQUIRED


def refuse_unapproved(request: Any, execute: Any) -> ToolMessage:
    """
    Та сама політика рівнем нижче: на виклику, а не на вузлі.

    Маршрутизація вирішує, в який вузол піде хід; це вирішує, чи виклик узагалі
    має право виконатись, і сидить усередині вузла, який вважається безпечним.
    Ці дві перевірки надлишкові навмисно. Затвердження на рівні вузла залежить
    від того, чи правильно маршрутизатор прочитав список, а ціна одного промаху
    в тому списку — зміна в чиємусь домі. Тому сам виклик перевіряється ще раз,
    просто перед виконанням, тим самим `risk_of`.

    Відмова — це ToolMessage, а не виняток: модель дізнається, чому нічого не
    сталося, і може про це сказати, точно як з будь-яким іншим результатом.
    """
    name = request.tool_call["name"]
    if risk_of(name) in APPROVAL_REQUIRED:
        return ToolMessage(
            content=f"Відхилено політикою: {name} потребує підтвердження "
                    f"оператора і не може виконуватись у цьому вузлі.",
            tool_call_id=request.tool_call["id"],
            name=name,
        )
    return execute(request)


def fresh_state(query: str, **overrides: Any) -> dict:
    """Початковий стан, у якому будь-яке поле можна замінити на потрібне демо."""
    state: dict[str, Any] = {
        "messages": [HumanMessage(content=query)],
        "step_count": 0,
        "start_time": time.time(),
        "total_tokens": 0,
        "last_tool_calls": [],
    }
    state.update(overrides)
    return state


if __name__ == "__main__":
    print("=" * 60)
    print("📌 Демо 1: нормальний хід, захист мовчить")
    result = guarded_agent.invoke(fresh_state(
        "Порівняй температуру у вітальні та спальні."))
    print(f"✅ {result['messages'][-1].content}")
    print(f"   кроків: {result['step_count']}, "
          f"токенів: {result['total_tokens']}")

    print("\n" + "=" * 60)
    print("📌 Демо 2: спрацьовує ліміт кроків")
    result = guarded_agent.invoke(fresh_state(
        "Що показують усі датчики в будинку?", step_count=MAX_STEPS))
    print(f"🛑 {result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("📌 Демо 3: спрацьовує таймаут")
    result = guarded_agent.invoke(fresh_state(
        "Що заплановано у спальні?", start_time=time.time() - TIMEOUT_SEC - 1))
    print(f"🛑 {result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("📌 Демо 4: спрацьовує бюджет токенів")
    result = guarded_agent.invoke(fresh_state(
        "Скільки електрики пішло за тиждень?", total_tokens=MAX_TOKENS + 1))
    print(f"🛑 {result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("📌 Демо 5: спрацьовує детекція повторів")
    # Виклик, який модель зараз зробить, вписаний у стан так, ніби він уже
    # відбувся витком раніше. Записаний з явним name=null, який сама модель
    # пропускає, — збігається, бо обидва написання спершу проходять args_schema.
    already = [("sensor_read", json.dumps(
        {"target": {"room": "living_room", "name": None},
         "sensor_type": "temperature"}, sort_keys=True))]
    result = guarded_agent.invoke(fresh_state(
        "Яка температура у вітальні?", last_tool_calls=already))
    print(f"🛑 {result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("📌 Демо 6: останній рубіж — recursion_limit самого графа")
    try:
        guarded_agent.invoke(fresh_state("Яка температура у вітальні?"),
                             config={"recursion_limit": 2})
    except Exception as exc:
        print(f"💥 {type(exc).__name__}: обривається винятком, а не відповіддю")
