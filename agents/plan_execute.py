"""Plan-and-Execute: planner → executor → replanner, маршрут вирішує код, а не модель."""

import time
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agents.react import llm
from agents.guards import fresh_state, guarded_agent
from agents.planner import make_plan
from core.schemas import PlanStep, ReplanDecision

MAX_PLAN_STEPS = 4
MAX_REPLANS = 2


class PlanExecuteState(TypedDict):
    """
    Стан зовнішнього циклу.

    `messages` тут немає. Зовнішній граф не веде розмови: кожен крок — це
    окремий прогін внутрішнього агента зі своєю короткою історією, а між ними
    переживає лише `completed`, у вигляді тексту.

    `response` водночас є прапорцем завершення: порожній під час роботи,
    заповнений, коли відповідь є.
    """
    task: str
    plan: list[PlanStep]
    completed: list[str]
    step_idx: int
    response: str
    replan_count: int


def _ask(call, attempts: int = 2):
    """
    Повторити структурований виклик, який повернувся непридатним.

    Це інший збій, ніж той, який ретраїть HTTP-клієнт. Клієнт розбирається з
    429, 5xx і таймаутами — з транспортом. Тут ідеться про цілком успішну
    відповідь, чий вміст порожній або не лягає у схему: на рівні протоколу не
    сталося нічого поганого, тож ніхто нижче це не повторить. Повертає None,
    коли спроби вичерпано, а що означає відсутня відповідь — вирішує той, хто
    викликав.
    """
    for _ in range(attempts):
        try:
            result = call()
        except Exception:
            continue
        if result is not None:
            return result
    return None


def planner_node(state: PlanExecuteState) -> dict:
    """
    Побудувати план. Викликається рівно один раз за прогін.

    Переплануванння сюди не повертається: переплановувач сам вписує новий план
    у стан. Повернення сюди означало б планування з нуля, без тих результатів,
    через які переплановування й знадобилось.

    Якщо плану побудувати не вдалося, саме завдання стає планом з одного кроку.
    Агент деградує до звичайного ReAct — це гірша відповідь, а не падіння.
    """
    task = state["task"]
    plan = _ask(lambda: make_plan(task, max_steps=MAX_PLAN_STEPS))

    if plan is None or not plan.steps:
        fallback = [PlanStep(step_id=1, description=task)]
        return {"plan": fallback, "step_idx": 0, "completed": [],
                "replan_count": 0}

    print(f"📋 Мета: {plan.goal}")
    for step in plan.steps:
        print(f"   {step.step_id}. {step.description} "
              f"[{step.tool_name or 'без інструмента'}]")

    return {"plan": plan.steps, "step_idx": 0, "completed": [],
            "replan_count": 0}


def executor_node(state: PlanExecuteState,
                  config: RunnableConfig | None = None) -> dict:
    """
    Виконати один крок усередині вузла, викликавши цілого іншого агента.

    Внутрішній агент стартує з порожньої історії: він бачить крок, плановий
    інструмент і стислий переказ того, що повернули попередні кроки, — і більше
    нічого. Завдяки цьому кожен крок коштує однаково, хоч би яким довгим був
    план, і втрачається все, що переказ відкинув. У багатоагентній системі
    питання ніколи не в тому, як передати керування, а в тому, що передати
    разом із ним: усе — потонеш у токенах, замало — виконавець працює наосліп.

    `fresh_state` дає кожному кроку власний бюджет кроків, часу й токенів, тож
    один дорогий крок не може виморити решту плану.
    """
    idx = state["step_idx"]
    plan = state["plan"]

    if idx >= len(plan):
        return {"response": "Усі кроки плану виконано."}

    step = plan[idx]
    context = ""
    if state["completed"]:
        context = "\n\nЩо вже відомо:\n" + "\n".join(
            f"- крок {i}: {result}"
            for i, result in enumerate(state["completed"], 1))

    hint = (f"\nПлановий інструмент: {step.tool_name}."
            if step.tool_name else
            "\nЦей крок не потребує інструментів — міркуй з того, що відомо.")

    prompt = (f"Виконай крок {idx + 1} з {len(plan)}: {step.description}"
              f"{hint}{context}\n\nПоверни стислий результат.")

    result = guarded_agent.invoke(fresh_state(prompt), config=config)
    answer = str(result["messages"][-1].content)
    print(f"   ✅ крок {idx + 1}: {answer[:90]}")

    return {"completed": state["completed"] + [answer], "step_idx": idx + 1}


replanner = llm.with_structured_output(ReplanDecision, method="json_schema")


def replanner_node(state: PlanExecuteState) -> dict:
    """
    Оцінити поступ. Саме оцінити, а не вирішити — маршрут визначає код.

    Порожній словник — легітимне повернення: «мені нема чого додати». Стан
    лишається таким, яким його залишив виконавець, а `should_continue`
    працює від лічильника кроків, тож модель, яка не відповіла, віддає
    керування звичайній логіці, а не зупиняє прогін.
    """
    done = "\n".join(f"крок {i}: {r}"
                     for i, r in enumerate(state["completed"], 1))
    remaining = state["plan"][state["step_idx"]:]
    left = "\n".join(f"крок {s.step_id}: {s.description}" for s in remaining)

    decision = _ask(lambda: replanner.invoke(
        f"Завдання користувача: {state['task']}\n\n"
        f"Виконані кроки:\n{done or '(жодного)'}\n\n"
        f"Кроки, що лишились:\n{left or '(немає)'}\n\n"
        f"Якщо зібраних даних достатньо — action='finish' і обов'язково "
        f"final_answer українською. Якщо план більше не веде до мети — "
        f"'replan' з updated_plan. Інакше — 'continue'."
    ))

    if decision is None:
        if remaining:
            return {}
        return {"response": "Завдання виконано (переплановувач не відповів)."}

    print(f"   🔀 {decision.action}: {decision.reasoning[:80]}")

    if decision.action == "finish":
        return {"response": decision.final_answer or "Завдання виконано."}

    if decision.action == "replan" and decision.updated_plan:
        return {
            "plan": decision.updated_plan,
            "step_idx": 0,
            "replan_count": state.get("replan_count", 0) + 1,
        }

    return {}


def should_continue(state: PlanExecuteState) -> str:
    """
    Куди йти далі.

    Переплановувач висловлює думку, а маршрут визначає ця функція — і може
    його переважити: після MAX_REPLANS прогін завершується незалежно від того,
    чого хоче модель. Модель — радник, код — диспетчер.
    """
    if state.get("response"):
        return "finish"
    if state.get("replan_count", 0) >= MAX_REPLANS:
        return "finish"
    if state["step_idx"] >= len(state["plan"]):
        return "finish"
    return "execute"


pe_graph = StateGraph(PlanExecuteState)
pe_graph.add_node("planner", planner_node)
pe_graph.add_node("executor", executor_node)
pe_graph.add_node("replanner", replanner_node)

pe_graph.add_edge(START, "planner")
pe_graph.add_edge("planner", "executor")
pe_graph.add_edge("executor", "replanner")
pe_graph.add_conditional_edges("replanner", should_continue,
                               {"execute": "executor", "finish": END})

pe_agent = pe_graph.compile()


def new_state(task: str) -> dict[str, Any]:
    """Початковий стан для одного завдання."""
    return {"task": task, "plan": [], "completed": [], "step_idx": 0,
            "response": "", "replan_count": 0}


if __name__ == "__main__":
    task = ("Перевір, чи не залишилось десь увімкнене світло, подивись "
            "температуру у спальні й скажи, скільки електрики пішло сьогодні.")

    from runtime.trajectory import attach

    print("=" * 60)
    config, trajectory_log = attach()
    started = time.perf_counter()
    result = pe_agent.invoke(new_state(task), config=config)
    elapsed = time.perf_counter() - started
    trajectory_log.save()

    print(f"\n🤖 Відповідь:\n{result['response']}")
    print(f"\n   кроків виконано: {len(result['completed'])}, "
          f"переплановувань: {result['replan_count']}, "
          f"час: {elapsed:.1f}с")
    print(f"📊 Траєкторія: {trajectory_log.log_path} — "
          f"{trajectory_log.summary()}")

    print("\n" + "=" * 60)
    print("📌 Те саме питання звичайним ReAct — для порівняння ціни")
    started = time.perf_counter()
    plain = guarded_agent.invoke(fresh_state(task))
    print(f"🤖 {plain['messages'][-1].content}")
    print(f"\n   кроків: {plain['step_count']}, "
          f"токенів: {plain['total_tokens']}, "
          f"час: {time.perf_counter() - started:.1f}с")

    print("\n" + "=" * 60)
    print("📌 Маршрутизатор: три виходи, жоден не питає модель")
    half_done = {"plan": [PlanStep(step_id=1, description="крок")] * 3,
                 "step_idx": 1, "response": "", "replan_count": 0}
    print(f"   є ще кроки            → {should_continue(half_done)}")
    print(f"   переплановувань {MAX_REPLANS}     → "
          f"{should_continue({**half_done, 'replan_count': MAX_REPLANS})}")
    print(f"   відповідь уже є       → "
          f"{should_continue({**half_done, 'response': 'готово'})}")

    print("\n📈 Схема графа (Mermaid):\n")
    print(pe_agent.get_graph().draw_mermaid())
