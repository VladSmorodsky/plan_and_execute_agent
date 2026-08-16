"""JSON-лог траєкторії: кожен виклик моделі й інструмента як запис, таймінги — з callbacks."""

import json
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel


def _jsonable(value: Any) -> Any:
    """Зробити артефакт інструмента серіалізовним, не розплющивши його в текст."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class TrajectoryLogger:
    """
    Збирає один прогін як список записів і потім записує його на диск.

    Текст обрізається, бо повідомлення може бути як завгодно довгим, а лог,
    який ніхто не може відкрити, марний. Артефакти не обрізаються: це типізовані
    результати, які інструменти вже побудували, вони малі за конструкцією і це
    єдина частина логу, яку можна не читати, а звірити з будинком.
    """

    def __init__(self, log_path: str = "trajectory.json"):
        self.log_path = log_path
        self.entries: list[dict[str, Any]] = []

    def log_step(self, node: str, output: str, duration_sec: float,
                 tool_name: str | None = None, artifact: Any = None,
                 error: str | None = None) -> None:
        self.entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": len(self.entries) + 1,
            "node": node,
            "tool": tool_name,
            "output": output[:500],
            "artifact": _jsonable(artifact),
            "error": error,
            "duration_sec": round(duration_sec, 3),
        })

    def save(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def summary(self) -> dict[str, Any]:
        """
        Зерно метрик агента.

        `tools_used` зберігає повтори, бо інструмент, викликаний п'ять разів
        поспіль, — це цикл; `unique_tools` їх прибирає, бо це покриття. Вони
        відповідають на різні питання, і жодне не замінює інше.

        Рахується лише за записами вузла `tools`. Запис агента теж називає
        інструмент, але то той, який модель попросила, і рахувати обидва
        означало б зарахувати кожен виклик двічі — як намір і як виконання.
        """
        tools_used = [e["tool"] for e in self.entries
                      if e["tool"] and e["node"] == "tools"]
        return {
            "total_steps": len(self.entries),
            "total_time_sec": round(sum(e["duration_sec"] for e in self.entries), 3),
            "tools_used": tools_used,
            "unique_tools": sorted(set(tools_used)),
            "errors": sum(1 for e in self.entries if e["error"]),
        }


class TrajectoryCallbackHandler(BaseCallbackHandler):
    """
    Чотири хуки двома парами: старт/кінець моделі та старт/кінець інструмента.

    Хуки помилок тут тому, що невдалий виклик — це саме той крок, який варто
    зберегти. `*_end` для нього не спрацьовує, тож обробник без них запише
    траєкторію, в якій збою не було. А в цьому агенті збої — робоча рутина:
    target, який не збігся з жодним пристроєм, це і є спосіб, у який модель
    дізнається, що в кімнаті є.
    """

    def __init__(self, logger: TrajectoryLogger):
        self.logger = logger
        self.started_at: dict[UUID, float] = {}
        self.tool_names: dict[UUID, str | None] = {}

    def _elapsed(self, run_id: UUID) -> float:
        return time.perf_counter() - self.started_at.pop(run_id, time.perf_counter())

    def on_chat_model_start(self, serialized: dict[str, Any],
                            messages: list[list[Any]], *, run_id: UUID,
                            **kwargs: Any) -> None:
        self.started_at[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID,
                   **kwargs: Any) -> None:
        message = getattr(response.generations[0][0], "message", None)
        requested = [call["name"]
                     for call in getattr(message, "tool_calls", [])]
        text = str(getattr(message, "content", "") or "")

        self.logger.log_step(
            node="agent",
            # Порожня відповідь — це не мовчання: модель витратила виток на
            # запит інструментів, і лог має сказати, яких саме.
            output=text or (f"→ {', '.join(requested)}" if requested else ""),
            duration_sec=self._elapsed(run_id),
            tool_name=requested[0] if requested else None,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                     **kwargs: Any) -> None:
        self.logger.log_step(node="agent", output="", error=repr(error),
                             duration_sec=self._elapsed(run_id))

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *,
                      run_id: UUID, **kwargs: Any) -> None:
        self.started_at[run_id] = time.perf_counter()
        self.tool_names[run_id] = serialized.get("name")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name = self.tool_names.pop(run_id, None)
        if isinstance(output, ToolMessage):
            self.logger.log_step(
                node="tools", tool_name=output.name or name,
                output=str(output.content), artifact=output.artifact,
                duration_sec=self._elapsed(run_id),
            )
        else:
            self.logger.log_step(node="tools", tool_name=name,
                                 output=str(output),
                                 duration_sec=self._elapsed(run_id))

    def on_tool_error(self, error: BaseException, *, run_id: UUID,
                      **kwargs: Any) -> None:
        self.logger.log_step(node="tools", output="", error=str(error),
                             tool_name=self.tool_names.pop(run_id, None),
                             duration_sec=self._elapsed(run_id))


def attach(config: dict | None = None,
           log_path: str = "trajectory.json") -> tuple[dict, TrajectoryLogger]:
    """
    Додати хендлер траєкторії до конфігу виклику.

    Логування лишається властивістю виклику, а не агента: хендлер домішується
    в конфіг, який той, хто викликає, і так збирався передати, тож жоден вузол,
    ребро чи інструмент не дізнається, що за ним спостерігають. Колбеки, які
    вже були в конфізі, зберігаються.

    Повертає конфіг для `invoke` і логер, якому наприкінці прогону треба
    викликати `save()`.
    """
    logger = TrajectoryLogger(log_path)
    merged = dict(config or {})
    merged["callbacks"] = (list(merged.get("callbacks") or [])
                           + [TrajectoryCallbackHandler(logger)])
    return merged, logger


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    from agents.react import react_agent

    query = ("Порівняй температуру у вітальні та спальні, "
             "і заразом скажи, що показує датчик диму на кухні.")

    logger = TrajectoryLogger("trajectory.json")
    started = time.perf_counter()
    result = react_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"callbacks": [TrajectoryCallbackHandler(logger)]},
    )
    wall_clock = time.perf_counter() - started
    logger.save()

    print(f"👤 Запит: {query}")
    print(f"🤖 Відповідь: {result['messages'][-1].content}\n")

    for entry in logger.entries:
        mark = "❌" if entry["error"] else "•"
        print(f"{mark} {entry['step']}. {entry['node']:<6} "
              f"{entry['duration_sec']:>6.2f}s  {entry['tool'] or ''} "
              f"{entry['error'] or entry['output'][:60]}")

    print(f"\n📊 Підсумок: {json.dumps(logger.summary(), ensure_ascii=False)}")
    print(f"⏱  Сума кроків {logger.summary()['total_time_sec']}s "
          f"проти {wall_clock:.3f}s усього прогону")
    print(f"💾 Траєкторія: {logger.log_path}")
