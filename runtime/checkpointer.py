"""Checkpointer (SqliteSaver): пам'ять між запитами й відновлення стану."""

import time

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.react import react_graph
from agents.guards import (
    MAX_STEPS,
    MAX_TOKENS,
    TIMEOUT_SEC,
    fresh_state,
    guarded_graph,
)

DB_PATH = "checkpoints.db"

THREAD = "demo-house-001"
OTHER_THREAD = "demo-house-002"
GUARDED_THREAD = "demo-house-guarded"


def _config(thread_id: str) -> dict:
    """Ключ, під яким зберігає checkpointer; передається в кожен invoke."""
    return {"configurable": {"thread_id": thread_id}}


def _turn(agent, query: str, config: dict) -> dict:
    """
    Один хід персистентного діалогу.

    Надсилається лише нове повідомлення. Рушій піднімає останній чекпоінт
    нитки, і редьюсер `add_messages` дописує повідомлення до відновленої
    стрічки, тож модель отримує всю історію — зібрану тут, а не запам'ятовану
    там. Пам'яті модель не має в жодному разі: у кожному HTTP-запиті їде вся
    стрічка, і саме тому довга сесія дорожчає й зрештою потребує обрізання чи
    підсумовування.
    """
    result = agent.invoke({"messages": [HumanMessage(content=query)]},
                          config=config)
    print(f"👤 {query}\n🤖 {result['messages'][-1].content}\n")
    return result


def dialogue_with_memory(checkpointer) -> None:
    """
    Доказ: уточнювальне питання, яке має сенс лише з історією.

    «А світло там увімкнене?» не називає кімнати. Інструменти її вимагають, тож
    агент без попереднього ходу не може викликати нічого — доводиться
    перепитувати.
    """
    print("=" * 60)
    print("📌 Демо 1: пам'ять між запитами")

    agent = react_graph.compile(checkpointer=checkpointer)
    config = _config(THREAD)

    _turn(agent, "Мене цікавить вітальня. Яка там зараз температура?", config)
    _turn(agent, "А світло там увімкнене?", config)


def show_snapshot(checkpointer) -> None:
    """
    Що насправді тримає checkpointer.

    `.next` — поле, яке варто запам'ятати: порожнє після нормального
    завершення, і саме там крок HITL покаже `('risky_tools',)`. Так і
    впізнається «граф чекає на людину».

    Чекпоінтів більше, ніж повідомлень, бо запис робиться після **кожного**
    вузла, включно з `tools`. На цій самій історії працює подорож у часі:
    запустити invoke із конфігу старого чекпоінта — і граф піде іншою гілкою.
    """
    print("=" * 60)
    print("📌 Демо 2: знімок стану")

    agent = react_graph.compile(checkpointer=checkpointer)
    config = _config(THREAD)

    snapshot = agent.get_state(config)
    history = list(agent.get_state_history(config))

    print(f"   thread_id: {THREAD}")
    print(f"   повідомлень у стрічці: {len(snapshot.values.get('messages', []))}")
    print(f"   наступний вузол: {snapshot.next or '(граф завершено)'}")
    print(f"   чекпоінт: {snapshot.config['configurable']['checkpoint_id']}")
    print(f"   чекпоінтів у гілці: {len(history)}\n")


def restore_and_isolate(checkpointer) -> None:
    """
    Новий об'єкт агента на тій самій нитці — і той самий об'єкт на іншій.

    Уся суть у цій парі: агент не володіє станом. Будь-який екземпляр, що
    підхопив нитку, продовжує діалог, а той самий екземпляр на іншому ключі не
    знає нічого — пам'ять є властивістю ключа.

    У межах одного процесу це імітація відновлення. Чесний варіант — два
    окремі запуски інтерпретатора, він описаний у README.
    """
    print("=" * 60)
    print("📌 Демо 3: відновлення новим екземпляром")

    restored = react_graph.compile(checkpointer=checkpointer)
    _turn(restored,
          "Нагадай, про яку кімнату ми говорили, і скажи, що там заплановано.",
          _config(THREAD))

    print("📌 Демо 4: інший thread_id — спільної пам'яті немає")
    _turn(restored, "А світло там увімкнене?", _config(OTHER_THREAD))


def _guarded_turn(agent, query: str, config: dict, *, reset: bool) -> dict:
    """
    Один хід захищеного агента: бюджет або обнуляється, або тягнеться далі.

    Уся різниця в тому, що пишеться в стан: `fresh_state` подає всі чотири
    поля захистів, голе повідомлення — жодного.
    """
    state = fresh_state(query) if reset else {
        "messages": [HumanMessage(content=query)]}

    result = agent.invoke(state, config=config)
    elapsed = time.time() - result["start_time"]

    print(f"👤 {query}\n🤖 {result['messages'][-1].content}")
    print(f"   кроків {result['step_count']}/{MAX_STEPS}, "
          f"токенів {result['total_tokens']}/{MAX_TOKENS}, "
          f"на годиннику {elapsed:.0f}с з {TIMEOUT_SEC}\n")
    return result


def guards_under_persistence(checkpointer) -> None:
    """
    Що персистентність робить із захистами — частина, специфічна саме для цього
    агента, а не для лекції.

    Лічильники з guards.py не мають редьюсера, тож вони перезаписуються, а не
    зливаються; але вони все одно зберігаються, і хід, який у них нічого не
    пише, успадковує значення попереднього. `step_count` росте далі,
    `start_time` і далі вказує на перше питання сесії, а бюджет «на прогін»
    непомітно стає бюджетом «на сесію»: четверте питання довгої розмови
    відхиляється через роботу, зроблену в першому.

    Жодна з двох поведінок не є помилковою, і в цьому суть — з checkpointer-ом
    вибір перестає бути деталлю реалізації. `fresh_state` пише всі чотири поля,
    тож передати його означає «кожне питання має власний бюджет»; передати саме
    лише повідомлення означає «вся сесія ділить один».
    """
    print("=" * 60)
    print("📌 Демо 5: лічильники теж зберігаються")

    agent = guarded_graph.compile(checkpointer=checkpointer)
    config = _config(GUARDED_THREAD)

    print("   — хід 1: fresh_state, бюджет свій")
    _guarded_turn(agent, "Скільки електрики будинок спожив сьогодні?",
                  config, reset=True)

    print("   — хід 2: лише повідомлення, бюджет успадкований")
    _guarded_turn(agent, "А за тиждень?", config, reset=False)

    print("   — хід 3: знову fresh_state, лічильники обнулено")
    _guarded_turn(agent, "І що заплановано на кухні?", config, reset=True)


def main() -> None:
    """
    Усе всередині одного `with`, бо в ньому живе з'єднання з базою.

    Спершу нитки видаляються. Без цього другий запуск скрипта виглядає
    зламаним: агент відповідає про вітальню ще до того, як про неї згадали, бо
    checkpoints.db лежить на диску з минулого разу. Це не костиль демо, а сама
    суть: стан переживає процес. `delete_thread` — штатний спосіб прибрати
    сесію, той самий виклик, що стоїть за кнопкою «очистити історію чату».
    """
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        for thread in (THREAD, OTHER_THREAD, GUARDED_THREAD):
            checkpointer.delete_thread(thread)

        dialogue_with_memory(checkpointer)
        show_snapshot(checkpointer)
        restore_and_isolate(checkpointer)
        guards_under_persistence(checkpointer)

    print("=" * 60)
    print(f"💾 Стан лежить у {DB_PATH} і переживе цей процес.")
    print(f'   sqlite3 {DB_PATH} "select count(*) from checkpoints;"')


if __name__ == "__main__":
    main()
