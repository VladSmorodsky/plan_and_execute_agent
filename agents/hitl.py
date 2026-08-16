"""Human-in-the-Loop: ризикові дії зупиняються через interrupt_before і чекають рішення оператора."""

import json
import sys
import uuid

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents.react import _tool_error_to_message, llm
from runtime.checkpointer import DB_PATH
from agents.guards import APPROVAL_REQUIRED, needs_approval, refuse_unapproved, risk_of
from core.home import DEVICES, resolve
from core.schemas import Target
from core.tools import ALL_TOOLS
from runtime.trajectory import attach

SAFE_TOOLS = [t for t in ALL_TOOLS if risk_of(t.name) not in APPROVAL_REQUIRED]
RISKY_TOOLS = [t for t in ALL_TOOLS if risk_of(t.name) in APPROVAL_REQUIRED]

llm_with_all_tools = llm.bind_tools(ALL_TOOLS)

HITL_SYSTEM_PROMPT = """Ти — асистент розумного будинку. Відповідай українською мовою.

Ти вмієш читати стан будинку: sensor_read (температура, вологість, рух),
device_status (увімкнено/вимкнено та налаштування), energy_consumption
(споживання електрики за період), schedule_list (автоматизації кімнати).

Ти також вмієш керувати будинком: light_set (світло та яскравість),
climate_set (цільова температура термостата), switch_set (побутові прилади).
Виклик climate_set і switch_set потребує підтвердження оператора — система
поставить його на паузу й покаже людині. Це нормальний хід речей, а не
помилка: просто виклич інструмент і дочекайся результату. Якщо інструмент
повернув результат — зміну вже виконано, повідом про неї у минулому часі й не
пиши, що чекаєш на підтвердження.

Якщо оператор відхилив операцію, НЕ повторюй її і не шукай обхідних шляхів.
Повідом користувача, що зміну не підтверджено, і зупинись.

ЗАВЖДИ бери дані про будинок з інструментів — НІКОЛИ не вигадуй показники,
стани пристроїв і розклади. Перед зміною, яка залежить від поточного стану,
спершу прочитай цей стан.

Розрізняй дві температури. sensor_read показує ПОТОЧНУ температуру повітря,
device_status — ЦІЛЬОВУ (target_c), виставлену на термостаті. Коли питають про
температуру в кімнаті, де є термостат, виклич обидва інструменти й назви обидва
числа. Після climate_set поточна не змінюється миттєво: кімната прогрівається
приблизно на 0.5°C за годину, і про це варто сказати.

Назви кімнат в інструментах англійські: кухня — kitchen, вітальня —
living_room, спальня — bedroom, ванна — bathroom, передпокій (коридор) — hall.
Поле name (назва пристрою) залишай порожнім, якщо йдеться про всю кімнату.

Відповідай стисло, називаючи конкретні числа та одиниці виміру."""


def hitl_agent_node(state: MessagesState) -> dict:
    """Вузол агента з agent.py, прив'язаний до всіх семи інструментів."""
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=HITL_SYSTEM_PROMPT)] + messages

    response = llm_with_all_tools.invoke(messages)
    return {"messages": [response]}


def route_by_risk(state: MessagesState) -> str:
    """
    Три виходи з вузла агента замість двох.

    Мітка "risky" навмисно не збігається з назвою вузла "risky_tools": ця
    функція повертає рішення, а куди воно веде — вирішує таблиця маршрутів.
    """
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    return "risky" if needs_approval(last) else "safe"


safe_tool_node = ToolNode(SAFE_TOOLS,
                          handle_tool_errors=_tool_error_to_message,
                          wrap_tool_call=refuse_unapproved)
risky_tool_node = ToolNode(RISKY_TOOLS,
                           handle_tool_errors=_tool_error_to_message)

hitl_graph = StateGraph(MessagesState)
hitl_graph.add_node("agent", hitl_agent_node)
hitl_graph.add_node("safe_tools", safe_tool_node)
hitl_graph.add_node("risky_tools", risky_tool_node)

hitl_graph.add_edge(START, "agent")
hitl_graph.add_conditional_edges(
    "agent", route_by_risk,
    {"safe": "safe_tools", "risky": "risky_tools", END: END},
)
hitl_graph.add_edge("safe_tools", "agent")
hitl_graph.add_edge("risky_tools", "agent")


def _pending(snapshot) -> AIMessage | None:
    """Виклик, перед яким граф спинився, або None, якщо він не спинявся."""
    if not snapshot.next:
        return None
    last = snapshot.values["messages"][-1]
    return last if getattr(last, "tool_calls", None) else None


def _show_request(message: AIMessage) -> None:
    """
    Показати оператору повністю, що зараз має статися.

    HITL вартий рівно стільки, наскільки читабельний цей вивід. Аргументи, які
    людина не може перевірити за дві секунди, перетворюють підтвердження на
    штампування не глядячи — стандартний спосіб, у який HITL не працює, маючи
    вигляд працюючого.
    """
    for call in message.tool_calls:
        print(f"   🔧 {call['name']}  (ризик: {risk_of(call['name'])})")
        print(f"   📝 {json.dumps(call['args'], ensure_ascii=False)}")


def _device_line(room: str, name: str) -> str:
    """Поточний стан одного пристрою, прямо з будинку."""
    device = resolve(Target(room=room, name=name))[0]
    return (f"{name} у {room}: "
            f"{'увімкнено' if device.is_on else 'вимкнено'}, {device.settings}")


def _house_snapshot() -> dict[str, tuple]:
    """Усе, що керувальні інструменти можуть зрушити, однією порівнянною
    структурою."""
    return {f"{d.name} у {d.room}": (d.is_on, dict(d.settings)) for d in DEVICES}


def _show_house_diff(before: dict[str, tuple]) -> None:
    """
    Що прогін насправді зробив із будинком.

    Матеріальна перевірка, і саме вона має значення: відповідь «готово» — це
    твердження моделі, а це — будинок. У лекції ту саму роль грає файл, що
    з'являється на диску; тут будинок живе в пам'яті, тож доказ треба знімати в
    межах процесу.
    """
    after = _house_snapshot()
    moved = [(name, before[name], state)
             for name, state in after.items() if before.get(name) != state]

    if not moved:
        print("   🏠 будинок не змінився")
        return

    for name, (was_on, was), (now_on, now) in moved:
        changes = [f"{'увімкнено' if now_on else 'вимкнено'} "
                   f"(було {'увімкнено' if was_on else 'вимкнено'})"] \
            if was_on != now_on else []
        changes += [f"{key} {was.get(key)!r} → {value!r}"
                    for key, value in now.items() if was.get(key) != value]
        print(f"   🏠 {name}: {', '.join(changes)}")


def _history(agent, config) -> None:
    """
    Слід аудиту, якого нікому не довелося писати.

    Колонка, що має значення, — `source`: `input` це новий запит, `loop` —
    граф, що йде сам, `update` — людина, яка втрутилася в стан. Відмова видно в
    нитці й через рік, як побічний продукт механіки, а не як журнал, який
    хтось не забув додати.
    """
    print("   історія (step / source / next):")
    for state in list(agent.get_state_history(config))[::-1]:
        print(f"      {state.metadata.get('step'):>3}  "
              f"{str(state.metadata.get('source')):<7} {state.next}")


def run(query: str, approve: bool = False, ask_operator: bool = False,
        thread_id: str | None = None) -> str:
    """
    Один запит: від паузи до рішення і до наслідку.

    Без `thread_id` кожен виклик — окремий діалог, і саме цього потребують
    скриптовані демо: вони не мусять пам'ятати одне одного. Той самий
    `thread_id`, переданий двічі, перетворює виклики на одну розмову, і друге
    питання може сказати «там», маючи на увазі кімнату з першого.

    Це працює і між процесами, і в межах одного, — тому з'єднання можна
    відкривати й закривати на кожен виклик: нитка живе у файлі, а не в цій
    функції. Будинок не живе: це стан модуля, і перезапуск інтерпретатора
    скидає його, тоді як діалог виживає.

    Повертає thread_id, щоб інтерактивна сесія могла використовувати його далі.
    """
    thread_id = thread_id or f"hitl-{uuid.uuid4().hex[:8]}"

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        agent = hitl_graph.compile(checkpointer=checkpointer,
                                   interrupt_before=["risky_tools"])
        config, trajectory_log = attach({"configurable": {"thread_id": thread_id}})
        before = _house_snapshot()

        print(f"👤 {query}")
        agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)

        snapshot = agent.get_state(config)
        request = _pending(snapshot)

        if request is None:
            print("   ▶️  паузи не було — жоден виклик не потребує підтвердження")
            print(f"   наступний вузол: {snapshot.next or '(граф завершено)'}")
            print(f"🤖 {snapshot.values['messages'][-1].content}")
            _show_house_diff(before)
            trajectory_log.save()
            print(f"   📊 траєкторія: {trajectory_log.log_path}")
            return thread_id

        print(f"\n⏸️  Пауза перед вузлом {snapshot.next}. Нічого ще не виконано.")
        _show_request(request)

        if ask_operator:
            answer = input("\n❓ Дозволити? [y/N]: ").strip().lower()
            approve = answer in ("y", "yes", "т", "так")

        if approve:
            print("\n✅ Оператор затвердив.")
            result = agent.invoke(None, config=config)
        else:
            print("\n❌ Оператор відхилив.")
            agent.update_state(
                config,
                {"messages": [ToolMessage(
                    content="Операцію відхилено оператором. Не повторюй її.",
                    tool_call_id=request.tool_calls[0]["id"],
                    name=request.tool_calls[0]["name"],
                )]},
                as_node="risky_tools",
            )
            after = agent.get_state(config)
            print(f"   .next після update_state: {after.next} "
                  f"(було {snapshot.next}) — ще до жодного invoke")
            result = agent.invoke(None, config=config)

        print(f"\n🤖 {result['messages'][-1].content}")
        _show_house_diff(before)

        final = agent.get_state(config)
        if final.next:
            print(f"   ⚠️ агент спробував ще раз, граф знову стоїть на {final.next}")
        _history(agent, config)
        trajectory_log.save()
        print(f"   📊 траєкторія: {trajectory_log.log_path}")
        return thread_id


def demo_call_level_guard() -> None:
    """
    Останній рубіж, показаний без моделі: ризиковий виклик, поданий просто в
    безпечний вузол.

    Це той випадок, який маршрутизатор має унеможливлювати, — і саме тому варто
    переконатися, що його зупиняє ще щось. Модель тут не бере участі: виклик
    написаний руками, точно таким, яким його зробив би зламаний маршрутизатор.

    Вузол запускається всередині графа з одного вузла, а не викликається
    напряму: ToolNode очікує рантайм LangGraph, а поза графом рантайму немає.
    Той одноразовий граф до того ж є чеснішою перевіркою — вузол працює так, як
    його веде рушій.
    """
    print("=" * 60)
    print("📌 Демо 4: перевірка на рівні виклику, без моделі")

    print(f"   {_device_line('living_room', 'thermostat')}")
    forged = AIMessage(content="", tool_calls=[{
        "name": "climate_set",
        "args": {"target": {"room": "living_room", "name": "thermostat"},
                 "target_c": 28.0},
        "id": "forged-1",
    }])

    probe = StateGraph(MessagesState)
    probe.add_node("safe_tools", safe_tool_node)
    probe.add_edge(START, "safe_tools")
    probe.add_edge("safe_tools", END)

    result = probe.compile().invoke({"messages": [forged]})

    print(f"   🛑 {result['messages'][-1].content}")
    print(f"   {_device_line('living_room', 'thermostat')}")


DEFAULT_QUERY = "Постав у спальні 24 градуси."

USAGE = """Використання:
  hitl.py                        чотири сценарії демо
  hitl.py --ask                  інтерактивно, запит за замовчуванням
  hitl.py --ask "свій запит"     інтерактивно, свій запит
  hitl.py "свій запит"           те саме: свій запит завжди інтерактивний"""


if __name__ == "__main__":
    print(f"🔧 без підтвердження: {[t.name for t in SAFE_TOOLS]}")
    print(f"🔒 з підтвердженням:  {[t.name for t in RISKY_TOOLS]}")

    words = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    query = " ".join(words)

    if query or "--ask" in sys.argv:
        print(f"\n{USAGE}\n")
        run(query or DEFAULT_QUERY, approve=False, ask_operator=True)
        sys.exit()

    print("\n" + "=" * 60)
    print("📌 Демо 1: відмова")
    print(f"   до: {_device_line('bedroom', 'thermostat')}")
    run(DEFAULT_QUERY, approve=False)

    print("\n" + "=" * 60)
    print("📌 Демо 2: затвердження — той самий запит, той самий механізм")
    run(DEFAULT_QUERY, approve=True)

    print("\n" + "=" * 60)
    print("📌 Демо 3: оборотна дія виконується без питань")
    print(f"   до: {_device_line('living_room', 'ceiling light')}")
    run("Вимкни світло у вітальні.", approve=False)

    print()
    demo_call_level_guard()
