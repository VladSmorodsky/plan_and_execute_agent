"""Інструменти агента: чотири читають будинок, три ним керують; рівень ризику оголошено на кожному."""

from datetime import datetime

from langchain_core.tools import ToolException, tool

from core.home import (
    DEVICES,
    SCHEDULES,
    DeviceNotFound,
    apply_change,
    next_run_after,
    resolve,
)
from core.schemas import (
    ClimateSetInput,
    ControlResult,
    DeviceStatus,
    DeviceStatusInput,
    EnergyConsumptionInput,
    EnergyReport,
    LightSetInput,
    Period,
    Room,
    ScheduleEntry,
    ScheduleListInput,
    SensorReading,
    SensorReadInput,
    SensorType,
    SwitchSetInput,
    Target,
)

PERIOD_DAYS: dict[Period, int] = {"today": 1, "week": 7, "month": 30}


def _now() -> datetime:
    """Поточний місцевий час із поясом — вихідні схеми не приймають наївного."""
    return datetime.now().astimezone()


@tool("sensor_read", args_schema=SensorReadInput,
      response_format="content_and_artifact", extras={"risk": "safe"})
def sensor_read(target: Target,
                sensor_type: SensorType) -> tuple[str, list[SensorReading]]:
    """Зчитати температуру, вологість або рух із пристроїв, які називає target."""
    try:
        devices = resolve(target)
    except DeviceNotFound as exc:
        raise ToolException(str(exc)) from exc

    measuring = [d for d in devices if sensor_type in d.readings]
    if not measuring:
        offered = sorted({s for d in devices for s in d.readings})
        raise ToolException(
            f"No device in {target.room} reports {sensor_type!r}. "
            f"Available there: {', '.join(offered) or 'nothing'}."
        )

    now = _now()
    readings = [
        SensorReading(device=d.name, room=d.room, sensor_type=sensor_type,
                      value=d.readings[sensor_type], measured_at=now)
        for d in measuring
    ]
    content = "; ".join(f"{r.device}: {r.value}{r.unit or ''}" for r in readings)
    return content, readings


@tool("device_status", args_schema=DeviceStatusInput,
      response_format="content_and_artifact", extras={"risk": "safe"})
def device_status(target: Target) -> tuple[str, list[DeviceStatus]]:
    """Повідомити, чи ввімкнені цільові пристрої, та їхні поточні налаштування."""
    try:
        devices = resolve(target)
    except DeviceNotFound as exc:
        raise ToolException(str(exc)) from exc

    statuses = [
        DeviceStatus(device=d.name, room=d.room, is_on=d.is_on, settings=d.settings)
        for d in devices
    ]
    content = "; ".join(
        f"{s.device} is {'on' if s.is_on else 'off'}"
        + (f" ({s.settings})" if s.settings else "")
        for s in statuses
    )
    return content, statuses


@tool("energy_consumption", args_schema=EnergyConsumptionInput,
      response_format="content_and_artifact", extras={"risk": "safe"})
def energy_consumption(period: Period) -> tuple[str, EnergyReport]:
    """Повідомити споживання електрики всім будинком, з розбивкою по кімнатах."""
    days = PERIOD_DAYS[period]

    by_room: dict[Room, float] = {}
    for device in DEVICES:
        by_room[device.room] = by_room.get(device.room, 0.0) + device.kwh_today * days

    by_room = {room: round(kwh, 2) for room, kwh in by_room.items() if kwh}
    report = EnergyReport(period=period,
                          total_kwh=round(sum(by_room.values()), 2),
                          by_room=by_room)

    content = f"{report.total_kwh} kWh over the {period}: " + ", ".join(
        f"{room} {kwh}" for room, kwh in sorted(report.by_room.items())
    )
    return content, report


@tool("schedule_list", args_schema=ScheduleListInput,
      response_format="content_and_artifact", extras={"risk": "safe"})
def schedule_list(room: Room) -> tuple[str, list[ScheduleEntry]]:
    """Перелічити автоматизації, заплановані для однієї кімнати."""
    now = _now()
    entries = [
        ScheduleEntry(schedule_id=spec.schedule_id, room=spec.room,
                      description=spec.description,
                      next_run=next_run_after(spec, now), enabled=spec.enabled)
        for spec in SCHEDULES if spec.room == room
    ]

    if not entries:
        return f"No automations scheduled for {room}.", []

    content = "; ".join(
        f"{e.description} at {e.next_run:%H:%M}"
        + ("" if e.enabled else " (disabled)")
        for e in entries
    )
    return content, entries


READ_TOOLS = [sensor_read, device_status, energy_consumption, schedule_list]


def _devices_of_type(target: Target, device_type: str, action: str) -> list:
    """
    Звести target до пристроїв, до яких дію справді можна застосувати.

    Обидві невдачі кидають ToolException, а не повертають порожній список: з
    усіх можливих результатів найгірший — керувальний інструмент, який тихо
    нічого не зробив, бо тоді модель звітує про успіх, а будинок не зрушив.
    Текст помилки називає, що в кімнаті таки є, тож наступний виклик може бути
    правильним.
    """
    try:
        devices = resolve(target)
    except DeviceNotFound as exc:
        raise ToolException(str(exc)) from exc

    matching = [d for d in devices if d.type == device_type]
    if not matching:
        offered = ", ".join(sorted(f"{d.name} ({d.type})" for d in devices))
        raise ToolException(
            f"Nothing to {action} here: no {device_type} among {offered}."
        )
    return matching


def _describe(results: list[ControlResult], noun: str) -> str:
    """Один рядок для моделі, який прямо каже, коли нічого не зрушило."""
    moved = [r for r in results if r.changed]
    if not moved:
        return f"No change: {noun} already in the requested state."
    return "; ".join(
        f"{r.device} in {r.room}: "
        + ", ".join(f"{key} {r.previous.get(key)!r} → {value!r}"
                    for key, value in r.changed.items())
        for r in moved
    )


DEFAULT_BRIGHTNESS_PCT = 70


@tool("light_set", args_schema=LightSetInput,
      response_format="content_and_artifact", extras={"risk": "reversible"})
def light_set(target: Target, is_on: bool,
              brightness_pct: int | None = None) -> tuple[str, list[ControlResult]]:
    """Увімкнути або вимкнути світло, за потреби із заданою яскравістю (1-100%)."""
    lights = _devices_of_type(target, "light", "switch")

    results = []
    for device in lights:
        if is_on:
            level = brightness_pct or device.settings.get("brightness_pct") \
                    or DEFAULT_BRIGHTNESS_PCT
        else:
            level = 0
        changed, previous = apply_change(device, is_on=is_on,
                                         brightness_pct=level)
        results.append(ControlResult(device=device.name, room=device.room,
                                     changed=changed, previous=previous))

    return _describe(results, "light"), results


@tool("climate_set", args_schema=ClimateSetInput,
      response_format="content_and_artifact", extras={"risk": "risky"})
def climate_set(target: Target,
                target_c: float) -> tuple[str, list[ControlResult]]:
    """Виставити цільову температуру термостата, у °C."""
    thermostats = _devices_of_type(target, "climate", "heat")

    results = []
    for device in thermostats:
        changed, previous = apply_change(device, is_on=True, target_c=target_c)
        results.append(ControlResult(device=device.name, room=device.room,
                                     changed=changed, previous=previous))

    return _describe(results, "thermostat"), results


@tool("switch_set", args_schema=SwitchSetInput,
      response_format="content_and_artifact", extras={"risk": "risky"})
def switch_set(target: Target, is_on: bool) -> tuple[str, list[ControlResult]]:
    """Увімкнути або вимкнути побутовий прилад (кавоварку тощо)."""
    switches = _devices_of_type(target, "switch", "switch")

    results = []
    for device in switches:
        changed, previous = apply_change(device, is_on=is_on)
        results.append(ControlResult(device=device.name, room=device.room,
                                     changed=changed, previous=previous))

    return _describe(results, "appliance"), results


CONTROL_TOOLS = [light_set, climate_set, switch_set]
ALL_TOOLS = READ_TOOLS + CONTROL_TOOLS
