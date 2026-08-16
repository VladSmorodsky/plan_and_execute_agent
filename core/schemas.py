"""Pydantic-схеми: контракт із моделлю — входи інструментів, типізовані результати, план і рівні ризику."""

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """
    Базова модель для строгого формату даних від LLM.

    extra="forbid" дає "additionalProperties": false у JSON Schema — умову
    строгого function calling в OpenAI і причину, чому вигадане поле кидає
    ValidationError, а не мовчки ігнорується.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


Room = Literal["kitchen", "living_room", "bedroom", "bathroom", "hall"]


class Target(StrictModel):
    """
    До яких пристроїв стосується виклик.

    Target — це запит, а не ідентифікатор: він може дати нуль, один або кілька
    пристроїв. `room` обов'язкова, бо тримає пошук малим; `name` звужує до
    одного пристрою, коли в кімнаті їх кілька.
    """
    room: Room
    name: str | None = Field(min_length=1, default=None,
                             description="The name of device. Omit or set to null to target all devices in the room.")


SensorType = Literal["temperature", "humidity", "motion"]
Period = Literal["today", "week", "month"]


class SensorReadInput(StrictModel):
    """
    Вхід sensor_read: один канал вимірювання цільових пристроїв.
    """
    target: Target = Field(description="Which device to read from.")
    sensor_type: SensorType = Field(
        description="Measurement to read: 'temperature' (°C), 'humidity' (%), "
                    "or 'motion' (detected / clear).")


class DeviceStatusInput(StrictModel):
    """
    Вхід device_status: поточний стан увімкнення та налаштування.
    """
    target: Target = Field(description="Which device to inspect.")


class EnergyConsumptionInput(StrictModel):
    """
    Вхід energy_consumption: споживання електрики, зведене за період.
    """
    period: Period = Field(
        description="Aggregation window: 'today', 'week' (last 7 days), "
                    "or 'month' (last 30 days).")


class ScheduleListInput(StrictModel):
    """
    Вхід schedule_list: автоматизації, заплановані для однієї кімнати.
    """
    room: Room = Field(description="Room whose automations to list.")


# Одиниця виміру — чиста функція від типу датчика, тому вона виводиться, а не
# зберігається: збережена могла б розійтися з типом, який описує.
SENSOR_UNITS: dict[SensorType, str | None] = {
    "temperature": "°C",
    "humidity": "%",
    "motion": None,
}


def _require_aware(value: datetime) -> datetime:
    """
    Відхилити дату без часового поясу.

    Наївна мітка часу робить лог траєкторії неоднозначним, а її порівняння з
    обізнаним `now` кидає TypeError десь глибоко у виконавці, а не тут, де
    причина ще видна.
    """
    if value.tzinfo is None:
        raise ValueError(
            "Timestamp must be timezone-aware, e.g. '2026-08-13T09:00:00+03:00'."
        )
    return value


class SensorReading(StrictModel):
    """
    Одне вимірювання, зняте з одного пристрою.
    """
    device: str
    room: Room
    sensor_type: SensorType
    value: bool | float
    measured_at: datetime

    @computed_field
    @property
    def unit(self) -> str | None:
        """Одиниця виміру `value`, виведена з `sensor_type`."""
        return SENSOR_UNITS[self.sensor_type]

    @field_validator("measured_at")
    @classmethod
    def _measured_at_is_aware(cls, v: datetime) -> datetime:
        return _require_aware(v)

    @model_validator(mode="after")
    def _value_matches_sensor_type(self) -> "SensorReading":
        """
        Показник руху — прапорець, решта — числа.

        Перевірка на bool стоїть першою навмисно: bool є підкласом int, тож
        інакше `True` пройшов би як цілком коректна температура.
        """
        if self.sensor_type == "motion":
            if not isinstance(self.value, bool):
                raise ValueError(
                    f"A 'motion' reading must be a bool (true = detected), "
                    f"got {self.value!r}."
                )
        else:
            if isinstance(self.value, bool):
                raise ValueError(
                    f"A '{self.sensor_type}' reading must be a number, got a bool."
                )
            if self.sensor_type == "humidity" and not 0 <= self.value <= 100:
                raise ValueError(
                    f"Humidity must be within 0..100 %, got {self.value}."
                )
        return self


class DeviceStatus(StrictModel):
    """
    Поточний стан одного пристрою.

    `settings` навмисно без типу: яскравість, цільова температура тощо різні
    для різних типів пристроїв, і фіксувати їх тут означало б заводити схему
    на кожен тип ще до появи керувальних інструментів.
    """
    device: str
    room: Room
    is_on: bool
    settings: dict[str, Any] = Field(default_factory=dict)


class EnergyReport(StrictModel):
    """
    Споживання електрики за один період.
    """
    period: Period
    total_kwh: float = Field(ge=0)
    by_room: dict[Room, float] = Field(default_factory=dict)

    @field_validator("by_room")
    @classmethod
    def _rooms_are_non_negative(cls, v: dict[str, float]) -> dict[str, float]:
        negative = sorted(room for room, kwh in v.items() if kwh < 0)
        if negative:
            raise ValueError(f"Consumption cannot be negative: {negative}.")
        return v

    @model_validator(mode="after")
    def _breakdown_fits_total(self) -> "EnergyReport":
        """
        Розбивка по кімнатах може бути неповною, але не більшою за тотал.

        Порівняння з невеликим епсилоном: сума float-ів, отриманих поділом
        одного числа, рідко відтворює його біт у біт.
        """
        breakdown = sum(self.by_room.values())
        if breakdown > self.total_kwh + 1e-9:
            raise ValueError(
                f"Per-room consumption ({breakdown:g} kWh) exceeds the total "
                f"({self.total_kwh:g} kWh)."
            )
        return self


class ScheduleEntry(StrictModel):
    """
    Одна автоматизація, запланована для кімнати.
    """
    schedule_id: str
    room: Room
    description: str
    next_run: datetime
    enabled: bool = True

    @field_validator("next_run")
    @classmethod
    def _next_run_is_aware(cls, v: datetime) -> datetime:
        return _require_aware(v)


RiskLevel = Literal["safe", "reversible", "risky"]

RISK_ORDER: list[RiskLevel] = ["safe", "reversible", "risky"]


class LightSetInput(StrictModel):
    """
    Вхід light_set: перемкнути світло і, за бажанням, задати яскравість.
    """
    target: Target = Field(description="Which light to switch.")
    is_on: bool = Field(description="true to turn on, false to turn off.")
    brightness_pct: int | None = Field(
        default=None, ge=1, le=100,
        description="Brightness in percent, 1..100. Omit to keep the current "
                    "level. Ignored when turning off.")


class ClimateSetInput(StrictModel):
    """
    Вхід climate_set: змінити цільову температуру термостата.

    Межі є частиною схеми, тож «постав 40 градусів» падає помилкою валідації,
    яку модель прочитає і виправить, і до будинку не доходить. Захист, що живе
    в схемі, нічого не коштує в рантаймі, і про нього неможливо забути в тілі
    інструмента.
    """
    target: Target = Field(description="Which thermostat to change.")
    target_c: float = Field(ge=5, le=30,
                            description="Target temperature in °C, 5..30.")


class SwitchSetInput(StrictModel):
    """
    Вхід switch_set: увімкнути або вимкнути прилад (кавоварку тощо).
    """
    target: Target = Field(description="Which appliance to switch.")
    is_on: bool = Field(description="true to turn on, false to turn off.")


class ControlResult(StrictModel):
    """
    Що саме один керувальний виклик змінив на одному пристрої.

    `previous` стоїть поруч із `changed`, бо саме ця пара дозволяє переглянути
    дію заднім числом: це інструкція для відкату, а в лозі траєкторії — різниця
    між «агент виставив 24°C» і «агент підняв з 18.5 до 24».
    """
    device: str
    room: Room
    changed: dict[str, Any] = Field(default_factory=dict)
    previous: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchInput(StrictModel):
    """
    Вхід knowledge_search: запит, який модель пише для бази документів.

    `query` — не повідомлення користувача. Модель переформульовує його під
    пошук, ділить надвоє, коли питання має дві половини, і пробує іншими
    словами, якщо перша спроба промахнулась. У цьому й різниця між agentic RAG
    і конвеєром, що штовхає сирий текст користувача в retriever.

    `room` — фільтр за метаданими, а не частина запиту: кімната, про яку
    питають («чи не волого у ванній»), і так відома моделі, а в ембедингу вона
    лише розмиває сенс.
    """
    query: str = Field(min_length=3,
                       description="What to look for, in Ukrainian. Write it "
                                   "as a phrase about the subject, not as a "
                                   "keyword.")
    room: Room | None = Field(
        default=None,
        description="Narrow the search to documents about one room. House-wide "
                    "documents (tariff, comfort norms, manuals) are returned "
                    "anyway. Omit unless the question is clearly about a "
                    "single room.")


class KnowledgeHit(StrictModel):
    """
    Один знайдений документ разом із метаданими, які дозволяють на нього
    послатися.
    """
    doc_id: str
    topic: str
    source: str
    room: str
    text: str


ToolName = Literal[
    "sensor_read", "device_status", "energy_consumption", "schedule_list"
]


class PlanStep(StrictModel):
    """
    Один крок плану.

    `tool_name` необов'язковий, бо останній крок більшості планів — «звести
    зібране у відповідь», і він не викликає нічого. Обов'язкове поле змусило б
    модель назвати інструмент і для такого кроку, а вона радше вигадає його,
    ніж лишить схему незадоволеною.
    """
    step_id: int = Field(ge=1, description="Step number, starting from 1.")
    description: str = Field(max_length=500,
                             description="What this step does, in one sentence.")
    tool_name: ToolName | None = Field(
        default=None,
        description="Tool this step calls, or null if the step only reasons "
                    "over what earlier steps returned.")


class Plan(StrictModel):
    """
    Як агент має намір відповісти на одне питання.

    `min_length=1` потрапляє в JSON Schema, тож модель знає про недопустимість
    порожнього плану ще до генерації, а pydantic перевіряє це після — одне й те
    саме обмеження з обох боків виклику. Верхня межа мала, бо в цьому будинку
    чотири читальні інструменти й п'ять кімнат; довший план — це симптом, а не
    план.
    """
    goal: str = Field(
        description="The user's request, restated in one sentence.")
    steps: list[PlanStep] = Field(min_length=1, max_length=10,
                                  description="Steps in execution order.")
    reasoning: str = Field(description="Why the plan is shaped this way.")


class ReplanDecision(StrictModel):
    """
    Що робити після виконання чергового кроку плану.

    `action` — це Literal, тож «можливо» просто неможливо висловити: дозволені
    дієслова доїжджають до моделі всередині схеми, а не перевіряються потім.

    Яке з двох інших полів має значення, залежить від цього дієслова, і
    декларативно виразити таку залежність pydantic не вміє. Тому вона живе в
    описах полів, які модель читає, — і ловиться в маршрутизаторі, який їй не
    довіряє. Опис тут є частиною контракту, а не документацією до нього.
    """
    action: Literal["continue", "replan", "finish"] = Field(
        description="'continue' to run the next step as planned, 'replan' to "
                    "replace the remaining steps, 'finish' when the question "
                    "is already answered.")
    updated_plan: list[PlanStep] | None = Field(
        default=None,
        description="The new list of steps. Required when action='replan', "
                    "null otherwise.")
    final_answer: str | None = Field(
        default=None,
        description="The answer for the user, in Ukrainian. Required when "
                    "action='finish', null otherwise.")
    reasoning: str = Field(description="Why this decision.")
