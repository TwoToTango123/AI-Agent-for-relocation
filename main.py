import json
import logging
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tools import get_climate_data, get_country_data

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

load_dotenv()
console = Console()
logging.basicConfig(level=logging.ERROR)
logging.getLogger("tools").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

DISCLAIMER = "Данные носят оценочный характер. Для принятия решений проверяйте актуальные источники."
API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def init_llm():
    if not ChatOpenAI or not API_KEY or API_KEY == "your_groq_key_here":
        return None
    try:
        return ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2,
        )
    except Exception as exc:
        logger.error("LLM init failed: %s", exc)
        return None


def parse_json_result(text: str):
    payload = text.replace("[КЭШ] ", "", 1) if text.startswith("[КЭШ] ") else text
    try:
        return json.loads(payload), None
    except json.JSONDecodeError:
        return None, payload


def tool_value(result: dict, *keys, default="N/A"):
    for key in keys:
        value = result.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def climate_score(data: dict, profile: str) -> float:
    temp = data.get("avg_temperature", 0) or 0
    rain = data.get("precipitation", 0) or 0
    profile_text = profile.lower()
    warm = any(word in profile_text for word in ["тепл", "жар", "солн", "пляж"])
    cold = any(word in profile_text for word in ["холод", "прохлад", "снег", "зима"])
    quiet = any(word in profile_text for word in ["спокой", "сем", "тих", "дет"])
    remote = any(word in profile_text for word in ["remote", "удал", "фриланс", "digital", "номад"])
    currency = str(data.get("currency", "")).upper()
    strong_currency = {"USD", "EUR", "GBP", "CHF", "CAD", "AUD"}
    strong_score = 2.0 if currency in strong_currency else 0.5
    if warm:
        return temp * 1.7 - rain * 0.4 + strong_score
    if cold:
        return -abs(temp - 5) * 1.4 - rain * 0.2 + strong_score
    if quiet:
        pop = float(data.get("city_population", data.get("population", 0)) or 0)
        return -pop / 1_000_000_000 - rain * 0.2 + (25 - abs(temp - 20)) * 0.6 + strong_score
    if remote:
        return strong_score + (25 - abs(temp - 20)) * 0.5 - rain * 0.2
    return strong_score + (25 - abs(temp - 18)) * 0.4 - rain * 0.3


def recommendation(city_a: dict, city_b: dict, profile: str) -> str:
    score_a = climate_score(city_a, profile)
    score_b = climate_score(city_b, profile)
    better = city_a if score_a >= score_b else city_b
    other = city_b if better is city_a else city_a
    profile_text = profile.strip().lower()
    known = any(
        key in profile_text
        for key in [
            "тепл", "жар", "солн", "пляж", "холод", "прохлад", "снег", "зима",
            "спокой", "сем", "тих", "дет", "remote", "удал", "фриланс", "digital", "номад",
        ]
    )
    if not profile_text:
        return (
            f"Профиль не указан, применен нейтральный режим. Предпочтительнее {better['city']} "
            f"(баланс климата и валюты выше, чем у {other['city']})."
        )
    if not known:
        return (
            f"Профиль распознан не полностью, применен нейтральный режим. Сейчас лидирует {better['city']} "
            f"по суммарному баллу относительно {other['city']}."
        )
    return (
        f"По вашему профилю больше подходит {better['city']}. "
        f"У него более сильный суммарный баланс по климату и валюте, чем у {other['city']}."
    )


def compare_cities(city_a_name: str, city_b_name: str, profile: str = "", llm=None) -> tuple[str, list[str]]:
    tool_calls = []
    country_a_raw = get_country_data(city_a_name, llm=llm)
    country_b_raw = get_country_data(city_b_name, llm=llm)
    tool_calls.extend(["get_country_data", "get_country_data"])

    country_a, error_a = parse_json_result(country_a_raw)
    country_b, error_b = parse_json_result(country_b_raw)
    if error_a or error_b:
        return error_a or error_b or "Не удалось получить данные о стране.", tool_calls

    climate_a_raw = get_climate_data(city_a_name, tool_value(country_a, "country", default=city_a_name), llm=llm)
    climate_b_raw = get_climate_data(city_b_name, tool_value(country_b, "country", default=city_b_name), llm=llm)
    tool_calls.extend(["get_climate_data", "get_climate_data"])

    climate_a, error_ca = parse_json_result(climate_a_raw)
    climate_b, error_cb = parse_json_result(climate_b_raw)
    if error_ca or error_cb:
        return error_ca or error_cb or "Не удалось получить климатические данные.", tool_calls

    currency_a = str(tool_value(country_a, "currency", default="N/A")).upper()
    currency_b = str(tool_value(country_b, "currency", default="N/A")).upper()

    table = Table(title=f"Сравнение городов: {city_a_name} vs {city_b_name}", show_lines=True)
    table.add_column("Параметр", style="bold cyan")
    table.add_column(city_a_name, style="green")
    table.add_column(city_b_name, style="magenta")
    table.add_row("Страна", str(tool_value(country_a, "country")), str(tool_value(country_b, "country")))
    table.add_row("Регион", str(tool_value(country_a, "region")), str(tool_value(country_b, "region")))
    table.add_row("Валюта", currency_a, currency_b)
    table.add_row("Часовой пояс", str(tool_value(climate_a, "city_timezone", default=tool_value(country_a, "city_timezone"))), str(tool_value(climate_b, "city_timezone", default=tool_value(country_b, "city_timezone"))))
    table.add_row("Население города", str(tool_value(country_a, "city_population", default=tool_value(country_a, "country_population"))), str(tool_value(country_b, "city_population", default=tool_value(country_b, "country_population"))))
    table.add_row("Средняя температура", str(climate_a.get("avg_temperature", "N/A")), str(climate_b.get("avg_temperature", "N/A")))
    table.add_row("Осадки", str(climate_a.get("precipitation", "N/A")), str(climate_b.get("precipitation", "N/A")))
    console.print(table)

    rec = recommendation(
        {"city": city_a_name, **country_a, **climate_a},
        {"city": city_b_name, **country_b, **climate_b},
        profile,
    )
    profile_text = profile.strip() or "без уточненного профиля"
    panel = Panel(
        f"[bold]Профиль:[/bold] {profile_text}\n[bold]Рекомендация:[/bold] {rec}\n\n{DISCLAIMER}",
        title="[bold blue]Итог[/bold blue]",
        border_style="blue",
    )
    return panel.renderable, tool_calls


def render_single(label: str, data: dict):
    table = Table(title=label, show_header=True, header_style="bold cyan")
    table.add_column("Поле", style="bold")
    table.add_column("Значение")
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, ensure_ascii=False)
        else:
            value_str = str(value)
        table.add_row(str(key), value_str)
    console.print(table)


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"\{.*\}", text, flags=re.S)
    if not fenced:
        return None
    try:
        return json.loads(fenced.group(0))
    except Exception:
        return None


def parse_intent_heuristic(query: str) -> dict:
    q = query.strip()
    q_lower = q.lower()

    if q_lower in {"exit", "quit", "выход"}:
        return {"intent": "exit"}
    if q_lower in {"help", "помощь", "хелп"}:
        return {"intent": "help"}

    if ";" in q:
        parts = [p.strip() for p in q.split(";") if p.strip()]
        if len(parts) >= 2:
            if parts[0].lower().startswith("compare"):
                parts[0] = parts[0][7:].strip()
            if parts[0].lower().startswith("сравни"):
                parts[0] = parts[0][6:].strip()
            return {
                "intent": "compare",
                "city_a": parts[0],
                "city_b": parts[1],
                "profile": parts[2] if len(parts) > 2 else "",
            }

    compare_ru = re.search(r"(?:сравни|сравнить)\s+([A-Za-zА-Яа-яЁё\-\s']+?)\s+(?:и|с)\s+([A-Za-zА-Яа-яЁё\-\s']+?)(?:\s+для\s+(.+))?$", q, flags=re.I)
    if compare_ru:
        return {
            "intent": "compare",
            "city_a": compare_ru.group(1).strip(" ,.!?"),
            "city_b": compare_ru.group(2).strip(" ,.!?"),
            "profile": (compare_ru.group(3) or "").strip(),
        }

    compare_en = re.search(r"(?:compare)\s+([A-Za-zА-Яа-яЁё\-\s']+?)\s+(?:and|with|vs)\s+([A-Za-zА-Яа-яЁё\-\s']+?)(?:\s+for\s+(.+))?$", q, flags=re.I)
    if compare_en:
        return {
            "intent": "compare",
            "city_a": compare_en.group(1).strip(" ,.!?"),
            "city_b": compare_en.group(2).strip(" ,.!?"),
            "profile": (compare_en.group(3) or "").strip(),
        }

    if any(word in q_lower for word in ["сравни", "compare", "лучше", "или", "vs"]):
        cities = re.findall(r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-']+", q)
        cities = [c for c in cities if c.lower() not in {"сравни", "сравнить", "compare"}]
        if len(cities) >= 2:
            return {"intent": "compare", "city_a": cities[0], "city_b": cities[1], "profile": q}

    if any(word in q_lower for word in ["климат", "погода", "температур", "осадки"]):
        cities = re.findall(r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-']+", q)
        if cities:
            city = cities[0]
            return {"intent": "climate", "city": city, "country": ""}

    if any(word in q_lower for word in ["валюта", "страна", "часовой пояс", "население"]):
        cities = re.findall(r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-']+", q)
        if cities:
            return {"intent": "country", "city": cities[0]}

    return {"intent": "general", "query": q}


def parse_intent_with_llm(llm, query: str) -> dict:
    if not llm:
        return parse_intent_heuristic(query)

    system = (
        "Ты классификатор запросов. Верни ТОЛЬКО JSON без пояснений. "
        "Формат: "
        "{\"intent\":\"compare|country|climate|general|exit|help\","
        "\"city_a\":\"\",\"city_b\":\"\",\"city\":\"\",\"country\":\"\"," 
        "\"profile\":\"\",\"query\":\"\"}. "
        "Если запрос не про климат/сравнение/страну - intent=general. "
        "Для compare выделяй 2 города."
    )
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
        parsed = extract_json(str(response.content))
        if parsed and parsed.get("intent"):
            return parsed
    except Exception as exc:
        logger.error("Intent LLM failed: %s", exc)
    return parse_intent_heuristic(query)


def answer_general(llm, query: str) -> str:
    if not llm:
        return "Для общих вопросов нужен настроенный GROQ_API_KEY в .env."
    prompt = (
        "Отвечай по сути на русском, даже если вопрос задан краткой фразой. "
        "Не отвечай отговорками. Если есть неопределенность, укажи ее в конце одной фразой."
    )
    try:
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
        return str(response.content).strip()
    except Exception as exc:
        logger.error("General LLM failed: %s", exc)
        return "Не удалось получить ответ LLM."


def log_trace(user_input: str, tool_calls: list[str], output_preview: str):
    record = {
        "timestamp": datetime.now().isoformat(),
        "input": user_input,
        "tool_calls": tool_calls,
        "output_preview": output_preview[:200],
    }
    try:
        with open("agent_trace.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Trace log failed: %s", exc)


def process_query(llm, query: str):
    intent = parse_intent_with_llm(llm, query)
    action = str(intent.get("intent", "general")).lower()

    if action == "exit":
        return "exit", [], "До встречи!"
    if action == "help":
        text = (
            "Пиши свободно, например:\n"
            "- Сравни Москву и Берлин для удаленки\n"
            "- Где теплее: Париж или Мадрид\n"
            "- Какая валюта в Японии\n"
            "- Любой общий вопрос: отвечу через LLM"
        )
        return "help", [], text

    if action == "compare":
        city_a = (intent.get("city_a") or "").strip()
        city_b = (intent.get("city_b") or "").strip()
        profile = str(intent.get("profile") or "").strip()
        if not city_a or not city_b:
            h = parse_intent_heuristic(query)
            city_a = city_a or h.get("city_a", "")
            city_b = city_b or h.get("city_b", "")
            profile = profile or h.get("profile", "")
        if not city_a or not city_b:
            return "error", [], "Не смог выделить два города. Пример: Сравни Москву и Берлин для удаленки."
        panel_or_text, tool_calls = compare_cities(city_a, city_b, profile, llm=llm)
        return "compare", tool_calls, panel_or_text

    if action == "country":
        city = (intent.get("city") or "").strip()
        if not city:
            h = parse_intent_heuristic(query)
            city = h.get("city", "")
        if not city:
            return "error", [], "Не смог определить город для запроса о стране/валюте."
        raw = get_country_data(city, llm=llm)
        tool_calls = ["get_country_data"]
        parsed, err = parse_json_result(raw)
        return "country", tool_calls, err if err else parsed

    if action == "climate":
        city = (intent.get("city") or "").strip()
        country = (intent.get("country") or "").strip()
        if not city:
            h = parse_intent_heuristic(query)
            city = h.get("city", "")
        if not city:
            return "error", [], "Не смог определить город для климатического запроса."
        if not country:
            country_raw = get_country_data(city, llm=llm)
            c_parsed, _ = parse_json_result(country_raw)
            country = tool_value(c_parsed or {}, "country", default=city)
            tool_calls = ["get_country_data", "get_climate_data"]
        else:
            tool_calls = ["get_climate_data"]
        raw = get_climate_data(city, country, llm=llm)
        parsed, err = parse_json_result(raw)
        return "climate", tool_calls, err if err else parsed

    answer = answer_general(llm, query)
    return "general", [], answer


def main():
    llm = init_llm()
    console.print(Panel("[bold cyan]AI-агент для сравнения городов по релокации[/bold cyan]", subtitle="свободный ввод | help | exit"))
    console.print("[cyan]Пиши свободным языком: сравни города, спроси про климат/валюту или задай общий вопрос.[/cyan]")

    while True:
        try:
            query = input("\nТы: ").strip()
        except KeyboardInterrupt:
            console.print("\n[cyan]До встречи![/cyan]")
            return
        except EOFError:
            console.print("[red]stdin закрыт. Запускай из интерактивного терминала.[/red]")
            return

        if not query:
            continue

        action, tool_calls, output = process_query(llm, query)
        if action == "exit":
            console.print("[cyan]До встречи![/cyan]")
            return

        log_trace(query, tool_calls, str(output))

        if isinstance(output, dict):
            render_single(action.capitalize(), output)
        elif isinstance(output, str):
            console.print(Panel(output, title="[bold blue]Агент[/bold blue]", border_style="blue"))
        else:
            console.print(output)


if __name__ == "__main__":
    main()
