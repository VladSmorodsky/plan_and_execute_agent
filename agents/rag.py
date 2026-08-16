"""Agentic RAG: пошук у базі знань як ще один інструмент агента."""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.react import SYSTEM_PROMPT, _tool_error_to_message, llm
from core.knowledge import index_stats, search
from core.schemas import KnowledgeHit, KnowledgeSearchInput, Room
from core.tools import READ_TOOLS
from runtime.trajectory import attach


@tool("knowledge_search", args_schema=KnowledgeSearchInput,
      response_format="content_and_artifact", extras={"risk": "safe"})
def knowledge_search(query: str,
                     room: Room | None = None) -> tuple[str, list[KnowledgeHit]]:
    """Шукати в базі знань про будинок: тарифи на електрику, норми комфорту для
    температури й вологості, правила дому та ранкові/вечірні сценарії,
    інструкції до пристроїв (термостат, світло, датчик руху, хаб) і поради з
    усунення несправностей.

    Використовуй щоразу, коли питання про те, чи нормальне значення, скільки
    щось коштує, чому автоматизація повелася саме так, або яке діє правило —
    нічого з цього інші інструменти виміряти не можуть."""
    hits = [
        KnowledgeHit(doc_id=doc.metadata["doc_id"], topic=doc.metadata["topic"],
                     source=doc.metadata["source"], room=doc.metadata["room"],
                     text=doc.page_content)
        for doc in search(query, room=room)
    ]

    if not hits:
        return (f"Nothing in the knowledge base matches {query!r}"
                + (f" for room {room}." if room else "."), [])

    content = "\n".join(f"[{h.topic} | source: {h.source}] {h.text}"
                        for h in hits)
    return content, hits


RAG_TOOLS = READ_TOOLS + [knowledge_search]
llm_with_rag_tools = llm.bind_tools(RAG_TOOLS)

RAG_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Крім показників будинку, ти маєш knowledge_search — пошук у базі знань про цей
будинок: тарифи на електрику, норми комфорту (температура, вологість), правила
дому та сценарії автоматизацій, інструкції до пристроїв і поради з усунення
несправностей.

Розділяй джерела: інструменти дають поточні числа, база знань — те, чого в
числах немає (чи це значення нормальне, скільки воно коштує, чому сценарій не
спрацював). Якщо питання про оцінку, причину, вартість або правило —
скористайся базою знань, а не здогадуйся.

НІКОЛИ не вигадуй тарифів, норм і правил дому. Якщо в базі знань потрібного
немає — скажи це прямо. Коли відповідь спирається на базу, назви джерело
(поле source)."""


def rag_agent_node(state: MessagesState) -> dict:
    """
    Вузол агента з agent.py, але з довшим промптом і довшим списком
    інструментів.

    Скопійований, а не імпортований, щоб цей файл читався окремо, — той самий
    вибір, що й у лекції. У справжньому проєкті обидва бралися б з однієї
    фабрики `build_agent(tools, system_prompt)`.
    """
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)] + messages

    response = llm_with_rag_tools.invoke(messages)
    return {"messages": [response]}


rag_graph = StateGraph(MessagesState)
rag_graph.add_node("agent", rag_agent_node)
rag_graph.add_node("tools", ToolNode(RAG_TOOLS,
                                     handle_tool_errors=_tool_error_to_message))

rag_graph.add_edge(START, "agent")
rag_graph.add_conditional_edges("agent", tools_condition,
                                {"tools": "tools", END: END})
rag_graph.add_edge("tools", "agent")

rag_agent = rag_graph.compile()


if __name__ == "__main__":
    print(f"📚 База знань: {index_stats['total']} документів "
          f"(переіндексовано цього разу: {index_stats['indexed']})")
    print(f"🔧 Інструментів у агента: {len(RAG_TOOLS)} "
          f"({', '.join(t.name for t in RAG_TOOLS)})")

    queries = [
        "Скільки грошей коштувала електрика, яку будинок спожив сьогодні?",
        "У ванній зараз волого? Це нормально і що з цим робити?",
        "Чому вранці не зварилась кава?",
        "У вітальні 20.4°C, а на термостаті виставлено 21. Він зламався?",
        "Яка зараз температура у спальні?",
    ]

    config, trajectory_log = attach()

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"👤 {query}")

        result = rag_agent.invoke({"messages": [HumanMessage(content=query)]},
                                  config=config)
        used = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]

        print(f"🔧 Інструменти: {used or '(жодного)'}")
        print(f"🤖 {result['messages'][-1].content}")

    trajectory_log.save()
    print(f"\n{'=' * 60}")
    print(f"📊 Траєкторія: {trajectory_log.log_path} — "
          f"{trajectory_log.summary()}")
    print("📌 В останньому запиті knowledge_search не викликано — "
          "рішення шукати ухвалює сам агент.")
