import json
import time
import requests
from datetime import datetime
from pydantic import BaseModel, Field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
CACHE_FILE = "cache.json"
CACHE_TTL = 86400  # 24 часа


def extract_city_country_with_llm(user_input: str, llm=None) -> dict:
    """
    Использует LLM для надежного извлечения города и страны из пользовательского ввода.
    Возвращает {"city": str, "country": str | None}
    Если LLM недоступен, возвращает исходный ввод как город.
    """
    if not llm or not user_input:
        return {"city": user_input.strip() if user_input else "", "country": None}
    
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return {"city": user_input.strip(), "country": None}
    
    system_prompt = """You are a city and country extractor. Your task is to identify which city and country the user is asking about.
Return ONLY valid JSON (no markdown, no code blocks) in exactly this format:
{"city": "City Name", "country": "Country Name"}

Rules:
- If input is ambiguous or missing country, set country to null
- Use proper English names for cities and countries
- Return the most famous/capital city if multiple matches exist
- Always put quotes around strings
"""
    
    human_msg = f'Extract city and country from: "{user_input}"'
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg)
        ])
        result_text = response.content.strip()
        
        # Убрать markdown блоки если есть
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        parsed = json.loads(result_text)
        if isinstance(parsed, dict) and "city" in parsed:
            return {
                "city": parsed.get("city", user_input.strip()) or user_input.strip(),
                "country": parsed.get("country")
            }
    except Exception as exc:
        logger.debug(f"LLM extraction failed for '{user_input}': {exc}")
    
    # Fallback: просто вернуть ввод как город
    return {"city": user_input.strip(), "country": None}

class CountryDataArgs(BaseModel):
    city: str = Field(description="Название города или страны для поиска")

class ClimateDataArgs(BaseModel):
    city: str = Field(description="Название города")
    country: str = Field(description="Название страны")

def load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Поврежденный или не-UTF-8 кэш не должен валить приложение.
        backup = Path(CACHE_FILE).with_suffix(f".corrupted.{int(time.time())}.json")
        try:
            Path(CACHE_FILE).rename(backup)
            logger.warning("Кэш поврежден, создан бэкап: %s (%s)", backup, exc)
        except Exception:
            logger.warning("Кэш поврежден, но не удалось сохранить бэкап (%s)", exc)
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_cached(key: str):
    cache = load_cache()
    if key in cache:
        entry = cache[key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            return entry["value"], True
    return None, False

def set_cached(key: str, value):
    cache = load_cache()
    cache[key] = {"value": value, "timestamp": time.time()}
    save_cache(cache)

def fetch_json(url, *, params=None, timeout=8):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()

def resolve_city_country(city: str, llm=None):
    """
    Определяет город и страну через Open-Meteo geocoding API.
    Сначала использует LLM для уточнения ввода, потом вызывает Open-Meteo.
    """
    extracted = extract_city_country_with_llm(city, llm)
    city_query = extracted.get("city", city).strip()
    
    try:
        geo = fetch_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_query, "count": 1, "language": "en", "format": "json"},
        )
        results = geo.get("results") or []
        if results:
            match = results[0]
            return {
                "city": match.get("name", city_query),
                "country": match.get("country", city_query),
                "country_code": match.get("country_code"),
                "latitude": match.get("latitude"),
                "longitude": match.get("longitude"),
                "timezone": match.get("timezone"),
                "population": match.get("population"),
            }
    except Exception as exc:
        logger.info("Geocoding failed for %s: %s", city_query, exc)
    
    return {"city": city_query, "country": city_query, "country_code": None, "latitude": None, "longitude": None, "timezone": None, "population": None}

def current_month_range():
    today = datetime.now().date()
    start = today.replace(day=1)
    return start.isoformat(), today.isoformat()

# Инструмент 1: Получение данных о стране
def get_country_data(city: str, llm=None) -> str:
    """Получает данные о стране из REST Countries API"""
    # Используем LLM для уточнения города и страны
    extracted = extract_city_country_with_llm(city, llm)
    city_clean = extracted.get("city", city).strip()
    country_from_llm = extracted.get("country")
    cache_key = f"country_v4_{city_clean.lower()}"
    
    cached_val, hit = get_cached(cache_key)
    if hit:
        return f"[КЭШ] {cached_val}"
    
    try:
        # Сначала пробуем найти координаты города через Open-Meteo
        resolved = resolve_city_country(city_clean, llm)
        country_query = country_from_llm or resolved["country"] or city_clean
        
        # Теперь ищем в REST Countries по имени страны
        country = None
        
        # Попробуем найти страну по имени из LLM или geocoding
        try:
            data = fetch_json(f"https://restcountries.com/v3.1/name/{country_query}")
            if isinstance(data, list) and data:
                country = data[0]
            elif isinstance(data, dict):
                country = data
        except Exception:
            country = None

        # Если не нашли по имени, ищем в полном списке стран
        if not country:
            try:
                all_countries = fetch_json("https://restcountries.com/v3.1/all")
                if isinstance(all_countries, list):
                    lowered = country_query.lower()
                    # Точный поиск по common name
                    country = next((item for item in all_countries if lowered == item.get("name", {}).get("common", "").lower()), None)
                    # Поиск по содержанию
                    if not country:
                        country = next((item for item in all_countries if lowered in item.get("name", {}).get("common", "").lower() or lowered in item.get("name", {}).get("official", "").lower()), None)
            except Exception:
                country = None

        if not country:
            return f"Не удалось найти страну для '{city}'. Попробуй указать город или страну иначе."
        
        result = {
            "input": city,
            "city": city_clean,
            "country": country.get("name", {}).get("common", ""),
            "currency": list(country.get("currencies", {}).keys())[0] if isinstance(country.get("currencies"), dict) and country.get("currencies") else "N/A",
            "region": country.get("region", ""),
            "city_timezone": resolved.get("timezone") or (country.get("timezones", ["N/A"])[0] if country.get("timezones") else "N/A"),
            "city_population": resolved.get("population"),
            "country_timezones": country.get("timezones", []),
            "country_population": country.get("population", 0),
            "capital": country.get("capital", ["N/A"])[0] if isinstance(country.get("capital"), list) and country.get("capital") else "N/A",
        }
        result_str = json.dumps(result, ensure_ascii=False)
        set_cached(cache_key, result_str)
        return result_str
    
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout при запросе к REST Countries для {city}")
        return f"Ошибка: истекло время ожидания при запросе к REST Countries для '{city}'"
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP ошибка для {city}: {e}")
        return f"Ошибка API REST Countries: {str(e)}"
    except json.JSONDecodeError:
        logger.error(f"Ошибка парсинга JSON для {city}")
        return f"Ошибка: не удалось распарсить ответ API REST Countries"
    except Exception as e:
        logger.error(f"Неожиданная ошибка в get_country_data: {e}")
        return f"Ошибка: {str(e)}"

# Инструмент 2: Получение климатических данных
def get_climate_data(city: str, country: str, llm=None) -> str:
    """Получает климатические данные через Open-Meteo"""
    extracted = extract_city_country_with_llm(city, llm)
    city_clean = extracted.get("city", city).strip()
    # Используем страну из параметра, либо из LLM информации
    country_clean = country or extracted.get("country", city).strip()
    cache_key = f"climate_v3_{city_clean.lower()}_{country_clean.lower()}"
    cached_val, hit = get_cached(cache_key)
    if hit:
        return f"[КЭШ] {cached_val}"
    
    try:
        # Геокодирование через Open-Meteo с полной информацией о городе и стране
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        query_name = f"{city_clean}, {country_clean}" if country_clean else city_clean
        params = {"name": query_name, "count": 1, "language": "en", "format": "json"}
        response = requests.get(geocode_url, params=params, timeout=12)
        response.raise_for_status()
        
        geo_data = response.json()
        if not geo_data.get("results"):
            return f"Город '{city}' не найден в базе Open-Meteo"
        
        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        city_tz = location.get("timezone")
        
        start_date, end_date = current_month_range()
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto",
        }
        try:
            weather_response = requests.get(weather_url, params=weather_params, timeout=20)
            weather_response.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError):
            fallback_url = "https://api.open-meteo.com/v1/forecast"
            fallback_params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 7,
            }
            fallback_response = requests.get(fallback_url, params=fallback_params, timeout=20)
            fallback_response.raise_for_status()
            weather = fallback_response.json()
            daily = weather.get("daily", {})
            temp_max = daily.get("temperature_2m_max", []) or []
            temp_min = daily.get("temperature_2m_min", []) or []
            precs = daily.get("precipitation_sum", []) or []
            if temp_max and temp_min:
                avg_temperature = round(sum((hi + lo) / 2 for hi, lo in zip(temp_max, temp_min)) / min(len(temp_max), len(temp_min)), 2)
            else:
                avg_temperature = None
            precipitation = round(sum(precs), 2) if precs else None
            result = {
                "city": city,
                "country": country,
                "latitude": lat,
                "longitude": lon,
                "city_timezone": city_tz,
                "period_start": start_date,
                "period_end": end_date,
                "avg_temperature": avg_temperature,
                "precipitation": precipitation,
                "source": "forecast_fallback",
            }
            result_str = json.dumps(result, ensure_ascii=False)
            set_cached(cache_key, result_str)
            return result_str

        weather = weather_response.json()
        daily = weather.get("daily", {})
        temps = daily.get("temperature_2m_mean", []) or []
        precs = daily.get("precipitation_sum", []) or []
        avg_temperature = round(sum(temps) / len(temps), 2) if temps else None
        precipitation = round(sum(precs), 2) if precs else None

        result = {
            "city": city,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "city_timezone": city_tz,
            "period_start": start_date,
            "period_end": end_date,
            "avg_temperature": avg_temperature,
            "precipitation": precipitation,
        }
        result_str = json.dumps(result, ensure_ascii=False)
        set_cached(cache_key, result_str)
        return result_str
    
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout при запросе к Open-Meteo для {city}")
        return f"Ошибка: истекло время ожидания при запросе климатических данных для '{city}'"
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP ошибка Open-Meteo для {city}: {e}")
        return f"Ошибка API Open-Meteo: {str(e)}"
    except json.JSONDecodeError:
        logger.error(f"Ошибка парсинга JSON Open-Meteo для {city}")
        return f"Ошибка: не удалось распарсить ответ Open-Meteo"
    except KeyError as e:
        logger.error(f"Отсутствует ожидаемое поле в ответе: {e}")
        return f"Ошибка: некорректный формат ответа API"
    except Exception as e:
        logger.error(f"Неожиданная ошибка в get_climate_data: {e}")
        return f"Ошибка: {str(e)}"

