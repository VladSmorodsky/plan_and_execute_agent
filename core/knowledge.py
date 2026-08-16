"""База знань про будинок: документи → ChromaDB → семантичний пошук."""

import hashlib
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Встановіть OPENAI_API_KEY у .env"

# Сховище лежить у корені проєкту, а не поруч із цим файлом: інакше після
# переїзду модуля база «переїхала» б разом із ним і переіндексувалась заново.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(ROOT, "chroma_db")
COLLECTION = "house_knowledge"

ANY_ROOM = "any"


def _doc(doc_id: str, topic: str, source: str, room: str, text: str) -> Document:
    """
    Один документ бази знань разом із метаданими, які повертає пошук.

    Текст перескладається по пробілах, щоб його можна було писати тут
    відступленим блоком і все одно індексувати одним рядком: модель ембедингів
    бачить і форматування, тож документ, що відрізняється від іншого лише
    переносами рядків, опинився б в іншому місці векторного простору.
    """
    content = " ".join(text.split())
    return Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "topic": topic,
            "source": source,
            "room": room,
            "checksum": hashlib.sha256(content.encode()).hexdigest()[:16],
        },
    )


DOCUMENTS: list[Document] = [
    _doc("thermostat-modes", "thermostat", "manual/thermostat", ANY_ROOM, """
        Термостат має режими heat, cool і auto та цільову температуру
        target_c. Нагрівання вмикається, коли температура опускається на
        0.5°C нижче цільової, і вимикається при її досягненні. Тому
        показник на 0.2–0.4°C нижчий за ціль — це нормальна робота, а не
        несправність.
    """),
    _doc("night-mode-bedroom", "night_mode", "house_rules", "bedroom", """
        Нічний режим спальні: з 23:00 до 6:30 цільова температура 18.5°C.
        Це навмисно нижче денної норми — прохолодніше спиться і менший
        рахунок. За зниження відповідає автоматизація sch-3 о 23:00.
    """),
    _doc("electricity-tariff", "tariff", "tariff_2026", ANY_ROOM, """
        Двозонний тариф на електрику: 4.32 грн за кВт·год з 07:00 до 23:00
        і 2.16 грн за кВт·год уночі. Лічильник агента не розділяє зони, тож
        для оцінки вартості множ спожиті кВт·год на денний тариф і кажи, що
        це верхня межа.
    """),
    _doc("comfort-norms", "comfort", "comfort_guide", ANY_ROOM, """
        Норми комфорту: вдень 20–22°C, вночі 18–19°C, відносна вологість
        40–60%. Понад 60% — ризик конденсату і цвілі, потрібне провітрювання
        чи витяжка. Нижче 30% — надто сухо, страждають слизові й дерев'яні
        меблі.
    """),
    _doc("bathroom-ventilation", "ventilation", "house_rules", "bathroom", """
        Витяжка у ванній вмикається, коли вологість перевищує 65%, і працює
        ще 10 хвилин після падіння нижче. Ранкову автоматизацію sch-4
        вимкнули вручну через шум, тому після душу вологість спадає повільно
        і значення близько 60% зранку — очікувані.
    """),
    _doc("kitchen-coffee-scenario", "coffee", "house_rules", "kitchen", """
        Автоматизація sch-1 вмикає кавоварку о 07:00 лише в будні. Сценарій
        скасовується, якщо датчик руху в передпокої не фіксував руху з 06:00
        — це ознака, що вдома нікого немає. Тож невзварена кава найчастіше
        означає вихідний або відсутність руху, а не поламану кавоварку.
    """),
    _doc("motion-sensor", "motion", "manual/motion", "hall", """
        Датчик руху в передпокої після спрацювання витримує паузу 90 секунд.
        Значення clear (руху немає) означає лише те, що за останні 90 секунд
        руху не було, і не є доказом, що в будинку нікого немає.
    """),
    _doc("lights-dimming", "lights", "manual/lights", ANY_ROOM, """
        Стельові світильники й торшер підтримують яскравість brightness_pct
        від 1 до 100. Значення 0 разом із вимкненим станом — це норма, а не
        помилка: пристрій запам'ятовує 0 після вимкнення.
    """),
    _doc("energy-baseline", "energy", "energy_guide", ANY_ROOM, """
        Типове добове споживання будинку — 6–9 кВт·год у міжсезоння. Понад
        12 кВт·год за добу — привід перевірити обігрів. Взимку термостати
        дають до 70% рахунку, тому саме цільова температура впливає на нього
        найсильніше.
    """),
    _doc("troubleshooting-heating", "troubleshooting", "troubleshooting",
         ANY_ROOM, """
        Якщо термостат у режимі heat, а температура не росте: перевір
        відчинене вікно, перекритий радіатор і фільтр (міняти раз на пів
        року). Нормальна швидкість прогріву — приблизно 0.5°C за годину,
        тож одразу після ввімкнення різниця з ціллю ще нічого не означає.
    """),
    _doc("living-room-evening", "evening_scene", "house_rules", "living_room",
         """
        Вечірній сценарій sch-2 о 19:30 вмикає стельове світло у вітальні на
        70% яскравості — це штатний режим, а не забуте світло. Увімкнене
        світло після 00:30 вважається забутим, і про нього варто повідомити.
    """),
    _doc("hub-recovery", "hub", "manual/hub", ANY_ROOM, """
        Кожен пристрій адресується назвою та кімнатою, назви унікальні лише в
        межах кімнати. Якщо пристрій не відповідає або зник зі списку,
        перезавантаж хаб утримуванням кнопки 10 секунд; пристрої повертаються
        протягом двох хвилин.
    """),
]


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)


def sync_index() -> dict[str, int]:
    """
    Привести сховище у відповідність до DOCUMENTS, зробивши якнайменше роботи.

    Ембединг — платний виклик API, тож що надсилати, вирішують контрольні суми:
    документ, чий текст не змінився, не чіпають; змінений перезаписують під тим
    самим `doc_id`; id, якого в цьому файлі більше немає, видаляють. Запуск
    двічі поспіль нічого не робить, і в цьому вся суть: індексація — операція зі
    станом, і саме на другому запуску наївна версія тихо плодить дублікати.
    """
    stored = store.get(include=["metadatas"])
    known = {
        doc_id: (metadata or {}).get("checksum")
        for doc_id, metadata in zip(stored["ids"], stored["metadatas"])
    }

    stale = [d for d in DOCUMENTS
             if known.get(d.metadata["doc_id"]) != d.metadata["checksum"]]
    if stale:
        store.add_documents(stale, ids=[d.metadata["doc_id"] for d in stale])

    current = {d.metadata["doc_id"] for d in DOCUMENTS}
    removed = [doc_id for doc_id in known if doc_id not in current]
    if removed:
        store.delete(ids=removed)

    return {"indexed": len(stale), "removed": len(removed),
            "total": len(DOCUMENTS)}


def search(query: str, room: str | None = None, k: int = 3) -> list[Document]:
    """
    Найближчі до `query` документи, за потреби звужені до однієї кімнати.

    Фільтр лишає всередині загальнобудинкові документи: коли питають про ванну,
    тариф і норми комфорту досі є відповіддю на половину питань. Саме це
    виражає `$in` — фільтр на рівність кімнаті сховав би їх.

    `k=3` за замовчуванням: менше — і потрібний документ може випасти з вікна,
    більше — і промпт наповнюється текстом, який лише виглядає доречним. Під
    капотом працює геометрія, а не розуміння: запит ембедиться тією самою
    моделлю і повертаються найближчі сусіди. Тому точні терміни, заперечення й
    числа — слабке місце такого пошуку.
    """
    where = {"room": {"$in": [room, ANY_ROOM]}} if room else None
    return store.similarity_search(query, k=k, filter=where)


index_stats = sync_index()


if __name__ == "__main__":
    print(f"📚 Колекція '{COLLECTION}': {index_stats['total']} документів "
          f"(переіндексовано {index_stats['indexed']}, "
          f"видалено {index_stats['removed']})")
    print(f"   Сховище: {CHROMA_DIR}")

    print("\n🔎 Семантика, а не ключові слова.")
    print("   Запит: 'як зберегти тепло вночі' — слова 'нічний режим' у ньому немає")
    for doc in search("як зберегти тепло вночі"):
        print(f"   • [{doc.metadata['topic']}] {doc.page_content[:70]}...")

    print("\n   Запит: 'чому вранці не було кави'")
    for doc in search("чому вранці не було кави"):
        print(f"   • [{doc.metadata['topic']}] {doc.page_content[:70]}...")

    print("\n🎯 Фільтр за кімнатою: 'волога' + room='bathroom'")
    for doc in search("волога", room="bathroom"):
        print(f"   • [{doc.metadata['topic']}] room={doc.metadata['room']} "
              f"| {doc.page_content[:60]}...")

    print("\n♻️  Повторний sync_index():", sync_index())
