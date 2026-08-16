# Агент розумного будинку на LangGraph

Автономний агент, який читає стан будинку й керує ним: датчики, світло,
термостати, побутові прилади, розклади автоматизацій і база знань про сам
будинок (тарифи, норми комфорту, правила дому).

Зібраний покроково — від простого ReAct-циклу до plan-and-execute з
переплануванням, персистентністю, RAG і підтвердженням ризикових дій людиною.
Будинок фейковий (`home.py`), але межа з ним оформлена так, щоб на її місце
став справжній Home Assistant без переписування агента.

## Швидкий старт

```bash
python3 -m venv .pae_agent
.pae_agent/bin/pip install -r requirements.txt

cp .env.example .env          # вписати свій OPENAI_API_KEY
.pae_agent/bin/python checking.py     # перевірка, що ключ живий
```

Модель — OpenAI `gpt-4.1`, ембединги — `text-embedding-3-small`.
Інших ключів не потрібно.

## Структура

```
core/         домен: те, що знає про будинок
agents/       графи: те, що вміє думати
runtime/      обв'язка: те, що спостерігає й зберігає
tests/        тести
```

| Модуль | Що показує |
|---|---|
| `checking.py` | середовище, ключ, ініціалізація моделі |
| `core/schemas.py` | Pydantic v2: входи інструментів, типізовані результати, план, рівні ризику |
| `core/home.py` | стан будинку: 10 пристроїв у 5 кімнатах, 5 розкладів, пошук за назвою + кімнатою |
| `core/tools.py` | 7 інструментів: 4 читальні, 3 керувальні |
| `core/knowledge.py` | 12 документів → ChromaDB → семантичний пошук |
| `agents/react.py` | ReAct-агент: цикл модель ⇄ інструменти |
| `agents/planner.py` | `with_structured_output(Plan)` — план як об'єкт, а не текст |
| `agents/guards.py` | ліміти прогону + політика ризику на рівні виклику |
| `agents/plan_execute.py` | planner → executor → replanner |
| `agents/rag.py` | agentic RAG: пошук як ще один інструмент |
| `agents/hitl.py` | human-in-the-loop через `interrupt_before` |
| `runtime/trajectory.py` | JSON-лог траєкторії через callbacks |
| `runtime/checkpointer.py` | SqliteSaver: пам'ять між запитами й відновлення |
| `tests/test_agents.py` | 68 тестів (64 офлайн + 4 живих) |

Залежності односторонні й саме тому поділені так: `core` не знає про `agents`,
`agents` не знають про `runtime`.

## Запуск

Кожен модуль — самостійне демо. Запускати з кореня проєкту, через `-m`:

```bash
.pae_agent/bin/python -m agents.react          # ReAct: 6 запитів, зокрема виправлення після помилки
.pae_agent/bin/python -m agents.planner        # план як типізований об'єкт
.pae_agent/bin/python -m runtime.trajectory    # лог траєкторії з таймінгами
.pae_agent/bin/python -m agents.guards         # 6 демо: ліміти, таймаут, повтори, recursion_limit
.pae_agent/bin/python -m agents.plan_execute   # план → виконання → переплануванння + порівняння ціни з ReAct
.pae_agent/bin/python -m runtime.checkpointer  # пам'ять, знімок стану, відновлення, ізоляція ниток
.pae_agent/bin/python -m core.knowledge        # індексація та «голий» пошук без агента
.pae_agent/bin/python -m agents.rag            # agentic RAG: 5 запитів, останній — без пошуку
.pae_agent/bin/python -m agents.hitl           # 4 сценарії: відмова, затвердження, оборотна дія, бекстоп
```

Якщо запускаєте кілька демо підряд, робіть паузу між важкими (`plan_execute`,
`rag`, `hitl`): OpenAI має ліміт токенів за хвилину, і на щільному прогоні
прилітає 429.

### Власні запити

HITL приймає запит аргументом і питає підтвердження в консолі:

```bash
.pae_agent/bin/python -m agents.hitl "Постав у спальні 21 градус."
.pae_agent/bin/python -m agents.hitl "Вимкни світло на кухні."   # оборотна дія — паузи не буде
```

Решта агентів — через інтерактивну сесію:

```bash
.pae_agent/bin/python -i
```

```python
from langchain_core.messages import HumanMessage, ToolMessage
from agents.react import react_agent
r = react_agent.invoke({"messages": [HumanMessage(content="ВАШ ЗАПИТ")]})
print([m.name for m in r["messages"] if isinstance(m, ToolMessage)])
print(r["messages"][-1].content)
```

```python
from agents.hitl import run
s = "my-session"
run("Яка температура у спальні?", thread_id=s)
run("А постав там 21 градус.", ask_operator=True, thread_id=s)   # спитає y/N
```

Однаковий `thread_id` перетворює виклики на одну розмову; діалог лежить у
`checkpoints.db` і переживає перезапуск процесу.

## Інструменти

Ризик оголошено на самому інструменті (`extras={"risk": ...}`), а не списком
небезпечних імен десь збоку. Інструмент без мітки вважається ризиковим.

| Інструмент | Ризик | Що робить |
|---|---|---|
| `sensor_read` | safe | температура, вологість, рух |
| `device_status` | safe | увімкнено/вимкнено та налаштування |
| `energy_consumption` | safe | споживання за today / week / month |
| `schedule_list` | safe | автоматизації кімнати |
| `knowledge_search` | safe | пошук у базі знань про будинок |
| `light_set` | reversible | світло та яскравість — виконується без підтвердження |
| `climate_set` | risky | цільова температура — **потребує оператора** |
| `switch_set` | risky | побутові прилади — **потребує оператора** |

`light_set` оборотний: увімкнене світло вимикається назад. Кавоварка — це
нагрівальний елемент, а термостат коштує грошей, тому вони за іншим боком межі.

## Захисні механізми

Два рівні, які відповідають на різні питання:

- **прогін** (`guards.py`): `MAX_STEPS=8`, `TIMEOUT_SEC=60`, `MAX_TOKENS=20_000`,
  детекція повторюваних викликів. Жоден не кидає виняток — усі завершують
  прогін звичайною відповіддю, тож користувач бачить «я вичерпав ліміт, ось що
  встиг», а не traceback;
- **виклик** (`risk_of` + `refuse_unapproved`): чи можна виконати саме цю дію
  без людини. Перевіряється двічі — маршрутизатором графа і обгорткою
  `wrap_tool_call` уже всередині вузла.

Останній рубіж під усім цим — `recursion_limit` самого графа.

## База знань

12 документів у `knowledge.py` → ChromaDB (`chroma_db/`): тарифи, норми
комфорту, правила дому, інструкції до пристроїв, усунення несправностей.
Індексація ідемпотентна за контрольною сумою — повторний запуск переіндексує
лише змінені документи й видалить ті, що зникли з файлу.

Пошук — інструмент, а не етап конвеєра: рішення шукати ухвалює агент. На
питання «яка температура у спальні» база знань не чіпається взагалі.

## Тести

```bash
.pae_agent/bin/python -m pytest tests/ -v                 # 64 passed, 4 skipped, ~1 с
RUN_LIVE_TESTS=1 .pae_agent/bin/python -m pytest tests/ -v   # + живі виклики
```

Швидкий набір не робить жодного запиту до API: топологія графів перевіряється
підставним `FakeModel`, який відповідає за сценарієм. Так детерміновано
перевірено і повернення в `agent` після інструмента, і паузу перед ризиковим
вузлом, і те, що після відмови `climate_set` не виконується.

Живі тести стверджують траєкторію (які інструменти викликано, чи зрушив
будинок), а не формулювання відповіді.

## Лог траєкторії

Кожен агент пише `trajectory.json`: по запису на виклик моделі й інструмента, з
тривалістю, помилками та типізованим артефактом результату.

```json
{"step": 4, "node": "tools", "tool": "climate_set",
 "output": "thermostat in bedroom: target_c 18.5 → 22.0",
 "artifact": [{"device": "thermostat", "room": "bedroom",
               "changed": {"target_c": 22.0}, "previous": {"target_c": 18.5}}],
 "duration_sec": 0.001}
```

Таймінги беруться з callback-хуків, а не діляться порівну між повідомленнями.
Файл перезаписується кожним прогоном; свій шлях задається через
`attach(log_path="...")`.
