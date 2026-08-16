"""Стан фейкового будинку: пристрої, розклади, пошук за назвою й кімнатою та єдина функція, що цей стан змінює."""

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import Field

from core.schemas import Room, SensorType, StrictModel, Target

DeviceType = Literal["light", "climate", "sensor", "switch"]


class Device(StrictModel):
    """
    Один пристрій будинку з його поточним станом.

    `readings` — канали вимірювання, які пристрій віддає. Термостат показує
    температуру, яку відчуває, і водночас несе цільову температуру в
    `settings`, тому це два різні поля, а не одна купа.
    """
    name: str
    room: Room
    type: DeviceType
    is_on: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)
    readings: dict[SensorType, bool | float] = Field(default_factory=dict)
    kwh_today: float = Field(default=0.0, ge=0)


DEVICES: list[Device] = [
    Device(name="ceiling light", room="kitchen", type="light",
           is_on=False, settings={"brightness_pct": 0}, kwh_today=0.4),
    Device(name="coffee maker", room="kitchen", type="switch",
           is_on=False, kwh_today=0.9),

    Device(name="ceiling light", room="living_room", type="light",
           is_on=True, settings={"brightness_pct": 70}, kwh_today=0.6),
    Device(name="floor lamp", room="living_room", type="light",
           is_on=False, settings={"brightness_pct": 0}, kwh_today=0.2),
    Device(name="thermostat", room="living_room", type="climate",
           is_on=True, settings={"target_c": 21.0, "mode": "heat"},
           readings={"temperature": 20.4}, kwh_today=3.1),
    Device(name="climate sensor", room="living_room", type="sensor",
           is_on=True, readings={"temperature": 20.4, "humidity": 44.0}),

    Device(name="ceiling light", room="bedroom", type="light",
           is_on=False, settings={"brightness_pct": 0}, kwh_today=0.3),
    Device(name="thermostat", room="bedroom", type="climate",
           is_on=True, settings={"target_c": 18.5, "mode": "heat"},
           readings={"temperature": 18.1}, kwh_today=2.4),

    Device(name="humidity sensor", room="bathroom", type="sensor",
           is_on=True, readings={"humidity": 61.0}),

    Device(name="motion sensor", room="hall", type="sensor",
           is_on=True, readings={"motion": False}),
]


class ScheduleSpec(StrictModel):
    """
    Повторювана автоматизація, збережена як час доби, а не як дата.

    Абсолютний `next_run`, вписаний у цей список, був би в минулому вже на
    момент читання; конкретна дата рахується під час виклику.
    """
    schedule_id: str
    room: Room
    description: str
    at_hour: int = Field(ge=0, le=23)
    at_minute: int = Field(default=0, ge=0, le=59)
    enabled: bool = True


SCHEDULES: list[ScheduleSpec] = [
    ScheduleSpec(schedule_id="sch-1", room="kitchen", at_hour=7,
                 description="Brew morning coffee"),
    ScheduleSpec(schedule_id="sch-2", room="living_room", at_hour=19, at_minute=30,
                 description="Turn on the evening lights"),
    ScheduleSpec(schedule_id="sch-3", room="bedroom", at_hour=23,
                 description="Lower the heating for the night"),
    ScheduleSpec(schedule_id="sch-4", room="bathroom", at_hour=8,
                 description="Ventilate after the morning shower", enabled=False),
    ScheduleSpec(schedule_id="sch-5", room="living_room", at_hour=6, at_minute=45,
                 description="Warm the room up before wake-up"),
]


def next_run_after(spec: ScheduleSpec, now: datetime) -> datetime:
    """Найближче майбутнє спрацювання `spec`, у часовому поясі `now`."""
    candidate = now.replace(hour=spec.at_hour, minute=spec.at_minute,
                            second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


class DeviceNotFound(LookupError):
    """
    Target не збігся з жодним пристроєм.

    Навмисно помилка, а не порожній результат: порожній список агент прочитав
    би як «такого пристрою в кімнаті немає, це й є відповідь» і повідомив би це
    користувачеві як факт. Текст помилки завжди перелічує, що в кімнаті таки є,
    тож модель може виправитись наступним викликом.
    """


def resolve(target: Target) -> list[Device]:
    """
    Перетворити Target на пристрої, про які йдеться.

    Пошук у два проходи, бо модель переказує назви своїми словами: скаже
    "lamp" замість "floor lamp" і "the floor lamp" про те саме. Точний збіг
    виграє першим, щоб повна назва не потрапила у ширший збіг за підрядком.
    """
    in_room = [d for d in DEVICES if d.room == target.room]
    if not in_room:
        raise DeviceNotFound(f"Room {target.room!r} holds no devices.")

    if target.name is None:
        return in_room

    needle = target.name.casefold()

    exact = [d for d in in_room if d.name.casefold() == needle]
    if exact:
        return exact

    partial = [d for d in in_room
               if needle in d.name.casefold() or d.name.casefold() in needle]
    if partial:
        return partial

    available = ", ".join(sorted(d.name for d in in_room))
    raise DeviceNotFound(
        f"No device matching {target.name!r} in {target.room}. "
        f"That room has: {available}."
    )


def apply_change(device: Device, *, is_on: bool | None = None,
                 **settings: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Змінити один пристрій і повідомити, що зрушило й чим воно було раніше.

    Єдине місце в коді, де будинок перестає бути тільки для читання. Тому на
    питання «що цей агент узагалі може зробити з моїм домом» відповідає пошук
    викликів однієї функції, а не читання всіх інструментів.

    Значення, рівне поточному, не є зміною: воно не потрапляє в жоден зі
    словників, і виклик, який нічого не змінив, повертає два порожні. Для
    оператора, якого просять затвердити дію, це суттєво — «вже вимкнено» і
    «вимкнено щойно» різні відповіді, і розрізняє їх лише діф. `None` означає
    «не чіпати», а не «записати null»: саме це дозволяє інструменту передати
    пропущений необов'язковий аргумент напряму.
    """
    changed: dict[str, Any] = {}
    previous: dict[str, Any] = {}

    if is_on is not None and device.is_on != is_on:
        previous["is_on"] = device.is_on
        changed["is_on"] = is_on
        device.is_on = is_on

    for key, value in settings.items():
        if value is None or device.settings.get(key) == value:
            continue
        previous[key] = device.settings.get(key)
        changed[key] = value
        device.settings[key] = value

    return changed, previous
