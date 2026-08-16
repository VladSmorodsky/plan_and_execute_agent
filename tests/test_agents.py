"""
Тести агента.

    .pae_agent/bin/python -m pytest test_agents.py -v
    RUN_LIVE_TESTS=1 .pae_agent/bin/python -m pytest test_agents.py -v
"""

import os
from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from core.home import DEVICES, DeviceNotFound, apply_change, resolve
from core.schemas import (
    ClimateSetInput,
    EnergyReport,
    LightSetInput,
    Plan,
    ReplanDecision,
    SensorReading,
    SensorReadInput,
    Target,
)
from core.tools import (
    climate_set,
    device_status,
    energy_consumption,
    light_set,
    schedule_list,
    sensor_read,
    switch_set,
)

load_dotenv()

needs_client = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Модулі агента будують клієнт OpenAI на імпорті (викликів не роблять)")

live = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Живий виклик LLM: запускати з RUN_LIVE_TESTS=1")


@pytest.fixture(autouse=True)
def restore_house():
    """
    Повернути будинок у початковий стан після кожного тесту.

    Керувальні інструменти змінюють стан модуля, тож без цього тест, який
    увімкнув кавоварку, змінює початкові умови для всього, що піде далі, — і
    набір починає зеленіти або червоніти залежно від порядку, який обрав
    pytest.
    """
    saved = [(device, device.is_on, dict(device.settings)) for device in DEVICES]
    yield
    for device, is_on, settings in saved:
        device.is_on = is_on
        device.settings = settings


def _aware(hour: int = 9) -> datetime:
    """Мітка часу з поясом — єдина, яку приймають схеми."""
    return datetime(2026, 8, 15, hour, tzinfo=timezone.utc)


class TestSchemas:
    """Контракт із моделлю: що їй дозволено надсилати."""

    def test_target_rejects_unknown_room(self):
        with pytest.raises(ValidationError):
            Target(room="garage")

    def test_target_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            Target(room="kitchen", name="")

    def test_target_without_name_means_whole_room(self):
        assert Target(room="kitchen").name is None

    def test_extra_field_is_forbidden(self):
        with pytest.raises(ValidationError):
            SensorReadInput(target={"room": "kitchen"}, sensor_type="temperature",
                            precision="high")

    def test_unit_is_derived_from_sensor_type(self):
        reading = SensorReading(device="climate sensor", room="living_room",
                                sensor_type="temperature", value=20.4,
                                measured_at=_aware())
        assert reading.unit == "°C"

    def test_motion_reading_must_be_bool(self):
        with pytest.raises(ValidationError):
            SensorReading(device="motion sensor", room="hall",
                          sensor_type="motion", value=1.0, measured_at=_aware())

    def test_temperature_reading_must_not_be_bool(self):
        with pytest.raises(ValidationError):
            SensorReading(device="thermostat", room="bedroom",
                          sensor_type="temperature", value=True,
                          measured_at=_aware())

    def test_humidity_out_of_range_is_rejected(self):
        with pytest.raises(ValidationError):
            SensorReading(device="humidity sensor", room="bathroom",
                          sensor_type="humidity", value=140.0,
                          measured_at=_aware())

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises(ValidationError):
            SensorReading(device="thermostat", room="bedroom",
                          sensor_type="temperature", value=18.1,
                          measured_at=datetime(2026, 8, 15, 9))

    def test_energy_breakdown_may_not_exceed_total(self):
        with pytest.raises(ValidationError):
            EnergyReport(period="today", total_kwh=5.0,
                         by_room={"kitchen": 3.0, "bedroom": 3.0})

    def test_energy_breakdown_may_be_partial(self):
        report = EnergyReport(period="today", total_kwh=5.0,
                              by_room={"kitchen": 3.0})
        assert report.total_kwh == 5.0

    def test_negative_consumption_is_rejected(self):
        with pytest.raises(ValidationError):
            EnergyReport(period="today", total_kwh=5.0, by_room={"hall": -1.0})

    def test_climate_target_out_of_range_is_rejected(self):
        with pytest.raises(ValidationError):
            ClimateSetInput(target={"room": "bedroom"}, target_c=40)

    def test_climate_target_within_range_is_accepted(self):
        assert ClimateSetInput(target={"room": "bedroom"}, target_c=21).target_c == 21

    def test_brightness_zero_is_rejected(self):
        with pytest.raises(ValidationError):
            LightSetInput(target={"room": "kitchen"}, is_on=True, brightness_pct=0)

    def test_plan_may_not_be_empty(self):
        with pytest.raises(ValidationError):
            Plan(goal="ціль", steps=[], reasoning="бо так")

    def test_replan_action_is_a_closed_set(self):
        with pytest.raises(ValidationError):
            ReplanDecision(action="maybe", reasoning="не впевнений")


class TestResolver:
    """Перетворення «назва + кімната» на пристрої — так, як їх адресує модель."""

    def test_exact_match_wins_over_substring(self):
        devices = resolve(Target(room="living_room", name="ceiling light"))
        assert [d.name for d in devices] == ["ceiling light"]

    def test_partial_match_is_allowed(self):
        devices = resolve(Target(room="living_room", name="lamp"))
        assert [d.name for d in devices] == ["floor lamp"]

    def test_no_name_returns_the_whole_room(self):
        assert len(resolve(Target(room="kitchen"))) == 2

    def test_unknown_device_names_what_the_room_has(self):
        with pytest.raises(DeviceNotFound) as exc:
            resolve(Target(room="kitchen", name="датчик диму"))
        assert "ceiling light" in str(exc.value)

    def test_apply_change_reports_before_and_after(self):
        lamp = resolve(Target(room="living_room", name="floor lamp"))[0]
        changed, previous = apply_change(lamp, is_on=True, brightness_pct=50)
        assert changed == {"is_on": True, "brightness_pct": 50}
        assert previous == {"is_on": False, "brightness_pct": 0}

    def test_apply_change_ignores_a_no_op(self):
        lamp = resolve(Target(room="living_room", name="floor lamp"))[0]
        changed, previous = apply_change(lamp, is_on=False)
        assert changed == {} and previous == {}

    def test_apply_change_leaves_none_alone(self):
        thermostat = resolve(Target(room="bedroom", name="thermostat"))[0]
        apply_change(thermostat, is_on=None, target_c=None)
        assert thermostat.settings["target_c"] == 18.5


class TestReadTools:
    """Чотири інструменти, які лише дивляться на будинок."""

    def test_sensor_read_returns_typed_artifact(self):
        message = sensor_read.invoke({
            "name": "sensor_read", "id": "1", "type": "tool_call",
            "args": {"target": {"room": "bedroom"}, "sensor_type": "temperature"},
        })
        assert message.artifact[0].value == 18.1
        assert message.artifact[0].unit == "°C"

    def test_sensor_read_rejects_a_channel_the_room_lacks(self):
        with pytest.raises(ToolException):
            sensor_read.invoke({"target": {"room": "kitchen"},
                                "sensor_type": "temperature"})

    def test_device_status_reports_settings(self):
        content = device_status.invoke({
            "target": {"room": "living_room", "name": "thermostat"}})
        assert "target_c" in content

    def test_energy_week_is_seven_days_of_today(self):
        today = energy_consumption.invoke({
            "name": "energy_consumption", "id": "1", "type": "tool_call",
            "args": {"period": "today"}}).artifact
        week = energy_consumption.invoke({
            "name": "energy_consumption", "id": "2", "type": "tool_call",
            "args": {"period": "week"}}).artifact
        assert round(week.total_kwh, 2) == round(today.total_kwh * 7, 2)

    def test_energy_total_matches_the_breakdown(self):
        report = energy_consumption.invoke({
            "name": "energy_consumption", "id": "1", "type": "tool_call",
            "args": {"period": "today"}}).artifact
        assert round(sum(report.by_room.values()), 2) == report.total_kwh

    def test_schedule_next_run_is_in_the_future(self):
        entries = schedule_list.invoke({
            "name": "schedule_list", "id": "1", "type": "tool_call",
            "args": {"room": "bedroom"}}).artifact
        assert entries[0].next_run > datetime.now().astimezone()
        assert entries[0].next_run < datetime.now().astimezone() + timedelta(days=1)

    def test_schedule_list_marks_a_disabled_automation(self):
        content = schedule_list.invoke({"room": "bathroom"})
        assert "disabled" in content

    def test_empty_room_is_not_an_error(self):
        content = schedule_list.invoke({"room": "hall"})
        assert "No automations" in content


class TestControlTools:
    """Три інструменти, які змінюють будинок."""

    def test_light_set_turns_on_with_a_default_brightness(self):
        light_set.invoke({"target": {"room": "living_room", "name": "floor lamp"},
                          "is_on": True})
        lamp = resolve(Target(room="living_room", name="floor lamp"))[0]
        assert lamp.is_on and lamp.settings["brightness_pct"] == 70

    def test_light_set_off_zeroes_the_brightness(self):
        light_set.invoke({"target": {"room": "living_room",
                                     "name": "ceiling light"}, "is_on": False})
        lamp = resolve(Target(room="living_room", name="ceiling light"))[0]
        assert not lamp.is_on and lamp.settings["brightness_pct"] == 0

    def test_light_set_refuses_a_device_that_is_not_a_light(self):
        with pytest.raises(ToolException):
            light_set.invoke({"target": {"room": "kitchen",
                                         "name": "coffee maker"}, "is_on": True})

    def test_climate_set_changes_the_target_not_the_reading(self):
        thermostat = resolve(Target(room="bedroom", name="thermostat"))[0]
        climate_set.invoke({"target": {"room": "bedroom"}, "target_c": 21.0})
        assert thermostat.settings["target_c"] == 21.0
        assert thermostat.readings["temperature"] == 18.1

    def test_climate_set_validates_through_the_tool(self):
        with pytest.raises(ValidationError):
            climate_set.invoke({"target": {"room": "bedroom"}, "target_c": 40})

    def test_control_result_carries_the_previous_value(self):
        message = climate_set.invoke({
            "name": "climate_set", "id": "1", "type": "tool_call",
            "args": {"target": {"room": "bedroom"}, "target_c": 21.0}})
        assert message.artifact[0].previous["target_c"] == 18.5
        assert message.artifact[0].changed["target_c"] == 21.0

    def test_switch_set_reports_a_no_op(self):
        switch_set.invoke({"target": {"room": "kitchen",
                                      "name": "coffee maker"}, "is_on": True})
        content = switch_set.invoke({"target": {"room": "kitchen",
                                                "name": "coffee maker"},
                                     "is_on": True})
        assert "No change" in content


@needs_client
class TestRiskPolicy:
    """guards.py: що дозволено виконати без людини."""

    def test_read_tools_are_safe(self):
        from agents.guards import risk_of
        assert risk_of("sensor_read") == "safe"

    def test_light_is_reversible_and_climate_is_risky(self):
        from agents.guards import risk_of
        assert risk_of("light_set") == "reversible"
        assert risk_of("climate_set") == "risky"

    def test_unknown_tool_defaults_to_risky(self):
        from agents.guards import risk_of
        assert risk_of("http_post") == "risky"

    def test_turn_risk_is_the_worst_of_its_calls(self):
        from agents.guards import call_risk, needs_approval
        message = AIMessage(content="", tool_calls=[
            {"name": "sensor_read", "args": {}, "id": "1"},
            {"name": "switch_set", "args": {}, "id": "2"},
        ])
        assert call_risk(message) == "risky"
        assert needs_approval(message)

    def test_a_turn_without_calls_needs_nobody(self):
        from agents.guards import needs_approval
        assert not needs_approval(AIMessage(content="готово"))

    def test_refuse_unapproved_never_executes(self):
        from agents.guards import refuse_unapproved

        class Request:
            tool_call = {"name": "climate_set", "args": {}, "id": "x"}

        def execute(_request):
            raise AssertionError("ризиковий виклик не мав виконатись")

        message = refuse_unapproved(Request(), execute)
        assert isinstance(message, ToolMessage)
        assert "підтвердження" in message.content

    def test_refuse_unapproved_lets_a_safe_call_through(self):
        from agents.guards import refuse_unapproved

        class Request:
            tool_call = {"name": "sensor_read", "args": {}, "id": "x"}

        assert refuse_unapproved(Request(), lambda _r: "виконано") == "виконано"


@needs_client
class TestGuards:
    """
    Захисти рівня прогону, перевірені поданням графу вичерпаного бюджету.

    Вони викликають справжній граф і все одно нічого не коштують: кожен захист
    перевіряється до виклику моделі, тож прогін завершується, а запит нікуди не
    йде.
    """

    def test_step_limit_stops_the_run(self):
        from agents.guards import MAX_STEPS, fresh_state, guarded_agent
        result = guarded_agent.invoke(
            fresh_state("Яка температура у спальні?", step_count=MAX_STEPS))
        assert "ліміт" in result["messages"][-1].content.lower()

    def test_timeout_stops_the_run(self):
        import time

        from agents.guards import TIMEOUT_SEC, fresh_state, guarded_agent
        result = guarded_agent.invoke(fresh_state(
            "Яка температура у спальні?",
            start_time=time.time() - TIMEOUT_SEC - 1))
        assert "таймаут" in result["messages"][-1].content.lower()

    def test_token_budget_stops_the_run(self):
        from agents.guards import MAX_TOKENS, fresh_state, guarded_agent
        result = guarded_agent.invoke(fresh_state(
            "Яка температура у спальні?", total_tokens=MAX_TOKENS + 1))
        assert "бюджет" in result["messages"][-1].content.lower()

    def test_a_guard_ends_the_run_instead_of_raising(self):
        from agents.guards import MAX_STEPS, fresh_state, guarded_agent
        result = guarded_agent.invoke(
            fresh_state("Яка температура у спальні?", step_count=MAX_STEPS))
        assert not result["messages"][-1].tool_calls


@needs_client
class TestPlanExecuteRouter:
    """Маршрутизатор кроку 6 вирішує в коді, хоч би що радила модель."""

    def test_more_steps_means_execute(self):
        from agents.plan_execute import should_continue
        from core.schemas import PlanStep
        state = {"plan": [PlanStep(step_id=1, description="крок")] * 3,
                 "step_idx": 1, "response": "", "replan_count": 0}
        assert should_continue(state) == "execute"

    def test_replan_limit_overrules_the_model(self):
        from agents.plan_execute import MAX_REPLANS, should_continue
        from core.schemas import PlanStep
        state = {"plan": [PlanStep(step_id=1, description="крок")] * 3,
                 "step_idx": 1, "response": "", "replan_count": MAX_REPLANS}
        assert should_continue(state) == "finish"

    def test_an_answer_finishes(self):
        from agents.plan_execute import should_continue
        from core.schemas import PlanStep
        state = {"plan": [PlanStep(step_id=1, description="крок")],
                 "step_idx": 0, "response": "готово", "replan_count": 0}
        assert should_continue(state) == "finish"


@needs_client
class TestHitlRouting:
    """Поділ на два вузли інструментів і те, куди відправляється хід."""

    def test_the_split_follows_the_policy_not_a_name_list(self):
        from agents.guards import APPROVAL_REQUIRED, risk_of
        from agents.hitl import RISKY_TOOLS, SAFE_TOOLS
        assert {t.name for t in RISKY_TOOLS} == {"climate_set", "switch_set"}
        assert all(risk_of(t.name) not in APPROVAL_REQUIRED for t in SAFE_TOOLS)

    def test_no_calls_ends_the_graph(self):
        from langgraph.graph import END

        from agents.hitl import route_by_risk
        assert route_by_risk({"messages": [AIMessage(content="готово")]}) == END

    def test_a_safe_call_goes_to_the_safe_node(self):
        from agents.hitl import route_by_risk
        message = AIMessage(content="", tool_calls=[
            {"name": "sensor_read", "args": {}, "id": "1"}])
        assert route_by_risk({"messages": [message]}) == "safe"

    def test_a_risky_call_goes_to_the_approval_node(self):
        from agents.hitl import route_by_risk
        message = AIMessage(content="", tool_calls=[
            {"name": "climate_set", "args": {}, "id": "1"}])
        assert route_by_risk({"messages": [message]}) == "risky"

    def test_a_mixed_turn_goes_to_the_approval_node(self):
        from agents.hitl import route_by_risk
        message = AIMessage(content="", tool_calls=[
            {"name": "sensor_read", "args": {}, "id": "1"},
            {"name": "climate_set", "args": {}, "id": "2"},
        ])
        assert route_by_risk({"messages": [message]}) == "risky"

    def test_the_safe_node_refuses_a_risky_call_it_should_never_receive(self):
        from langgraph.graph import END, START, MessagesState, StateGraph

        from agents.hitl import safe_tool_node
        forged = AIMessage(content="", tool_calls=[{
            "name": "climate_set",
            "args": {"target": {"room": "bedroom"}, "target_c": 28.0},
            "id": "forged-1"}])

        probe = StateGraph(MessagesState)
        probe.add_node("safe_tools", safe_tool_node)
        probe.add_edge(START, "safe_tools")
        probe.add_edge("safe_tools", END)
        result = probe.compile().invoke({"messages": [forged]})

        assert "Відхилено політикою" in result["messages"][-1].content
        thermostat = resolve(Target(room="bedroom", name="thermostat"))[0]
        assert thermostat.settings["target_c"] == 18.5


class FakeModel:
    """
    Модель, яка відповідає за сценарієм.

    У цьому й суть тестів графа: із підставленою моделлю цикл, маршрутизація та
    пауза стають такими ж детермінованими, як будь-який інший код, і їх можна
    перевіряти assert-ами, не платячи за жоден токен.
    """

    def __init__(self, *responses: AIMessage):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


@needs_client
class TestGraphTopology:
    """Цикл і пауза, які веде модель зі сценарієм."""

    def test_react_loop_returns_to_the_agent_after_a_tool(self, monkeypatch):
        import agents.react as agent_module
        fake = FakeModel(
            AIMessage(content="", tool_calls=[{
                "name": "sensor_read",
                "args": {"target": {"room": "bedroom"},
                         "sensor_type": "temperature"},
                "id": "call-1"}]),
            AIMessage(content="У спальні 18.1°C."),
        )
        monkeypatch.setattr(agent_module, "llm_with_tools", fake)

        result = agent_module.react_agent.invoke(
            {"messages": [HumanMessage(content="Яка температура у спальні?")]})

        assert fake.calls == 2, "після інструмента граф має повернутись в agent"
        assert [m.name for m in result["messages"]
                if isinstance(m, ToolMessage)] == ["sensor_read"]
        assert result["messages"][-1].content == "У спальні 18.1°C."

    def test_a_risky_call_pauses_before_the_node(self, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        import agents.hitl as hitl_module
        fake = FakeModel(AIMessage(content="", tool_calls=[{
            "name": "climate_set",
            "args": {"target": {"room": "bedroom"}, "target_c": 24.0},
            "id": "call-1"}]))
        monkeypatch.setattr(hitl_module, "llm_with_all_tools", fake)

        app = hitl_module.hitl_graph.compile(checkpointer=MemorySaver(),
                                             interrupt_before=["risky_tools"])
        config = {"configurable": {"thread_id": "test-pause"}}
        app.invoke({"messages": [HumanMessage(content="Постав 24 у спальні.")]},
                   config=config)

        snapshot = app.get_state(config)
        assert snapshot.next == ("risky_tools",)
        thermostat = resolve(Target(room="bedroom", name="thermostat"))[0]
        assert thermostat.settings["target_c"] == 18.5, "нічого не мало виконатись"

    def test_rejection_replaces_the_node_result(self, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        import agents.hitl as hitl_module
        fake = FakeModel(
            AIMessage(content="", tool_calls=[{
                "name": "climate_set",
                "args": {"target": {"room": "bedroom"}, "target_c": 24.0},
                "id": "call-1"}]),
            AIMessage(content="Оператор відхилив зміну."),
        )
        monkeypatch.setattr(hitl_module, "llm_with_all_tools", fake)

        app = hitl_module.hitl_graph.compile(checkpointer=MemorySaver(),
                                             interrupt_before=["risky_tools"])
        config = {"configurable": {"thread_id": "test-reject"}}
        app.invoke({"messages": [HumanMessage(content="Постав 24 у спальні.")]},
                   config=config)

        app.update_state(
            config,
            {"messages": [ToolMessage(content="Операцію відхилено оператором.",
                                      tool_call_id="call-1", name="climate_set")]},
            as_node="risky_tools")
        assert app.get_state(config).next == ("agent",)

        result = app.invoke(None, config=config)

        thermostat = resolve(Target(room="bedroom", name="thermostat"))[0]
        assert thermostat.settings["target_c"] == 18.5, "climate_set не мав виконатись"
        assert result["messages"][-1].content == "Оператор відхилив зміну."

    def test_approval_executes_the_same_pending_call(self, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        import agents.hitl as hitl_module
        fake = FakeModel(
            AIMessage(content="", tool_calls=[{
                "name": "climate_set",
                "args": {"target": {"room": "bedroom"}, "target_c": 24.0},
                "id": "call-1"}]),
            AIMessage(content="Готово."),
        )
        monkeypatch.setattr(hitl_module, "llm_with_all_tools", fake)

        app = hitl_module.hitl_graph.compile(checkpointer=MemorySaver(),
                                             interrupt_before=["risky_tools"])
        config = {"configurable": {"thread_id": "test-approve"}}
        app.invoke({"messages": [HumanMessage(content="Постав 24 у спальні.")]},
                   config=config)
        app.invoke(None, config=config)

        thermostat = resolve(Target(room="bedroom", name="thermostat"))[0]
        assert thermostat.settings["target_c"] == 24.0

    def test_a_thread_remembers_and_another_does_not(self, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        import agents.react as agent_module
        monkeypatch.setattr(agent_module, "llm_with_tools",
                            FakeModel(AIMessage(content="Записав."),
                                      AIMessage(content="Записав.")))

        checkpointer = MemorySaver()
        app = agent_module.react_graph.compile(checkpointer=checkpointer)
        mine = {"configurable": {"thread_id": "mine"}}
        other = {"configurable": {"thread_id": "other"}}

        app.invoke({"messages": [HumanMessage(content="Мене цікавить спальня.")]},
                   config=mine)
        app.invoke({"messages": [HumanMessage(content="А там?")]}, config=other)

        assert len(app.get_state(mine).values["messages"]) == 2
        assert len(app.get_state(other).values["messages"]) == 2
        assert app.get_state(mine).next == ()


@live
class TestLive:
    """
    Справжні виклики моделі. Перевіряється траєкторія, а не формулювання.
    """

    def test_react_reads_a_sensor(self):
        from agents.react import react_agent
        result = react_agent.invoke({"messages": [
            HumanMessage(content="Яка зараз температура у вітальні?")]})
        used = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
        assert "sensor_read" in used
        assert result["messages"][-1].content

    def test_rag_searches_only_when_the_answer_is_in_the_base(self):
        from agents.rag import rag_agent

        priced = rag_agent.invoke({"messages": [
            HumanMessage(content="Скільки коштує кіловат-година вдень?")]})
        assert "knowledge_search" in [m.name for m in priced["messages"]
                                      if isinstance(m, ToolMessage)]

        measured = rag_agent.invoke({"messages": [
            HumanMessage(content="Яка зараз температура у спальні?")]})
        assert "knowledge_search" not in [m.name for m in measured["messages"]
                                          if isinstance(m, ToolMessage)]

    def test_rejected_change_never_reaches_the_house(self):
        from agents.hitl import run
        run("Постав у спальні 24 градуси.", approve=False,
            thread_id="test-live-reject")
        thermostat = resolve(Target(room="bedroom", name="thermostat"))[0]
        assert thermostat.settings["target_c"] == 18.5

    def test_knowledge_base_answers_a_paraphrase(self):
        from core.knowledge import search
        topics = [d.metadata["topic"] for d in search("як зберегти тепло вночі")]
        assert "night_mode" in topics
