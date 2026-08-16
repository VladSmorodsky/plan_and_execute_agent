"""ReAct-агент у LangGraph: цикл модель ⇄ інструменти."""

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from core.tools import READ_TOOLS

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Встановіть OPENAI_API_KEY у .env"

llm = ChatOpenAI(model="gpt-4.1", temperature=0, timeout=30)
llm_with_tools = llm.bind_tools(READ_TOOLS)

SYSTEM_PROMPT = """Ти — асистент розумного будинку. Відповідай українською мовою.

Тобі доступні інструменти: sensor_read (температура, вологість, рух),
device_status (увімкнено/вимкнено та налаштування), energy_consumption
(споживання електрики за період), schedule_list (автоматизації кімнати).

ЗАВЖДИ бери дані про будинок з інструментів — НІКОЛИ не вигадуй показники,
стани пристроїв і розклади. Якщо питання стосується кількох кімнат або
кількох різних даних, виклич інструмент стільки разів, скільки потрібно.

Назви кімнат в інструментах англійські: кухня — kitchen, вітальня —
living_room, спальня — bedroom, ванна — bathroom, передпокій (коридор) — hall.
Поле name (назва пристрою) залишай порожнім, якщо питання про всю кімнату.

Якщо інструмент повернув помилку, прочитай її текст: він перелічує, що
насправді є в тій кімнаті. Виправ аргументи і спробуй ще раз; якщо потрібних
даних у будинку немає — скажи це користувачеві прямо.

Ти можеш лише читати стан будинку. Керувати пристроями ти поки не вмієш —
якщо просять щось увімкнути чи змінити, чесно про це попередь.

Відповідай стисло, називаючи конкретні числа та одиниці виміру."""


def agent_node(state: MessagesState) -> dict:
    """
    Викликати модель на розмові, яка вже є.

    Перевірка isinstance важлива, бо цей вузол виконується щоразу на витку
    циклу: без неї кожен виток дописував би на початок ще одну копію промпту.
    Дописане повідомлення лишається в локальній змінній і ніколи не потрапляє
    в стан, тож у стрічці лежить розмова, а не конфігурація.
    """
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_error_to_message(exc: ToolException) -> str:
    """
    Перетворити невдалий виклик інструмента на те, з чим модель може працювати.

    Саме через це ретельне формулювання ToolException у `resolve` має сенс:
    замість обриву прогону повідомлення «No device matching 'датчик диму' in
    kitchen. That room has: ceiling light, coffee maker.» повертається як
    звичайний результат інструмента, і наступний виток виправляє себе сам.

    Передається явно, бо стандартний обробник перетворює лише помилки
    валідації аргументів — виняток, кинутий у тілі інструмента, проходить
    наскрізь і кладе граф. Анотація тут є конфігурацією: ToolNode читає її, щоб
    вирішити, які винятки ловити, тож справжня помилка в шарі інструментів досі
    падає, а не подається моделі як порада. ToolInvocationError успадковує
    ToolException, тож порушення схеми теж повертаються моделі.
    """
    return str(exc)


tool_node = ToolNode(READ_TOOLS, handle_tool_errors=_tool_error_to_message)

react_graph = StateGraph(MessagesState)
react_graph.add_node("agent", agent_node)
react_graph.add_node("tools", tool_node)

react_graph.add_edge(START, "agent")
react_graph.add_conditional_edges(
    "agent",
    tools_condition,  # є tool_calls -> "tools", інакше -> END
    {"tools": "tools", END: END},
)
react_graph.add_edge("tools", "agent")  # цикл

react_agent = react_graph.compile()


if __name__ == "__main__":
    print("✅ ReAct-граф скомпільовано. Вузли:", list(react_graph.nodes))

    test_queries = [
        "Яка зараз температура у вітальні?",          # -> sensor_read
        "Світло на кухні ввімкнене?",                  # -> device_status
        "Скільки електрики будинок спожив за тиждень?",  # -> energy_consumption
        "Що заплановано у спальні?",                   # -> schedule_list
        # Дві кімнати й два різні вимірювання: одним витком не відповісти,
        # тож цикл мусить обернутись не один раз.
        "У ванній волого? І чи хтось рухається в коридорі?",
        # Пристрою не існує: цікавий тут другий виток, де модель читає помилку
        # і обирає справжній пристрій.
        "Що показує датчик диму на кухні?",
    ]

    from runtime.trajectory import attach

    config, trajectory_log = attach()

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"👤 Запит: {query}")

        result = react_agent.invoke({"messages": [HumanMessage(content=query)]},
                                    config=config)
        print(f"🤖 Відповідь: {result['messages'][-1].content}")

        used = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        if used:
            print(f"🔧 Інструменти: {[m.name for m in used]}")

    trajectory_log.save()
    print(f"\n{'=' * 60}")
    print(f"📊 Траєкторія: {trajectory_log.log_path} — "
          f"{json.dumps(trajectory_log.summary(), ensure_ascii=False)}")

    print(f"\n{'=' * 60}")
    print("📈 Схема графа (Mermaid):\n")
    print(react_agent.get_graph().draw_mermaid())
