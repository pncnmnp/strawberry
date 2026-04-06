from tools.weather import DEFINITION as _weather_def, get_weather
from tools.shutdown import DEFINITION as _shutdown_def, shutdown
from tools.reset import DEFINITION as _reset_def, reset_history
from tools.wikipedia import DEFINITION as _wikipedia_def, search_wikipedia

TOOLS = [_weather_def, _shutdown_def, _reset_def, _wikipedia_def]


def dispatch(tool_name: str, args: dict) -> str:
    if tool_name == "get_weather":
        return get_weather(**args)
    if tool_name == "shutdown":
        return shutdown()
    if tool_name == "reset_history":
        return reset_history()
    if tool_name == "search_wikipedia":
        return search_wikipedia(**args)
    raise ValueError(f"Unknown tool: {tool_name}")
