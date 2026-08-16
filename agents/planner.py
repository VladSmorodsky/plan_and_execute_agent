"""Планувальник: with_structured_output повертає об'єкт Plan, а не текст."""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.react import llm
from core.schemas import Plan

PLANNER_PROMPT = """Ти — планувальник агента розумного будинку. Твоя робота —
скласти план, а не виконати його. Пиши українською.

Доступні інструменти:
- sensor_read — температура, вологість або рух конкретної кімнати;
- device_status — чи ввімкнено пристрій та його налаштування;
- energy_consumption — споживання електрики за today / week / month;
- schedule_list — автоматизації однієї кімнати.

Один крок — одна підзадача, яку виконавець доводить до кінця самостійно. Він
теж агент і за потреби зробить кілька викликів, тож «перевірити світло в усіх
кімнатах» — це один крок, а не п'ять.

План мусить покривати ВСЕ, про що спитав користувач. Якщо підзадач більше, ніж
дозволено кроків, об'єднуй споріднені, але не викидай жодної частини питання.

Останній крок, який лише поєднує отримані дані у відповідь користувачеві,
залишай без інструмента (tool_name = null). Не вигадуй інструментів поза
переліком і не плануй керування пристроями — їх поки що немає.

Кімнати: kitchen, living_room, bedroom, bathroom, hall."""


planner = llm.with_structured_output(Plan, method="json_schema")


def make_plan(question: str, max_steps: int | None = None) -> Plan:
    """
    Скласти план викликів, які дадуть відповідь на `question`.

    `max_steps` — це обмеження глибини, а не побажання: у plan-and-execute
    кожен крок стає повним вкладеним прогоном агента, тож довжина плану і є
    ціна прогону. Задає її той, хто платить, — той, хто викликає.
    """
    request = question
    if max_steps is not None:
        request = f"{question}\n\n(Не більше {max_steps} кроків у плані.)"

    return planner.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=request),
    ])


if __name__ == "__main__":
    questions = [
        "Чи не задушливо у ванній і скільки електрики ми спалили за місяць?",
        "Порівняй температуру у вітальні та спальні й скажи, де тепліше.",
    ]

    for question in questions:
        plan = make_plan(question)
        print(f"\n{'=' * 60}")
        print(f"👤 Запит: {question}")
        print(f"🎯 Мета: {plan.goal}")
        for step in plan.steps:
            print(f"   {step.step_id}. {step.description} "
                  f"[{step.tool_name or 'без інструмента'}]")
        print(f"💭 Обґрунтування: {plan.reasoning}")
        print(f"   type(plan) = {type(plan).__name__}")
