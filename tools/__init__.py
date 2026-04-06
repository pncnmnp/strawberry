from tools.weather import DEFINITION as _weather_def, get_weather
from tools.shutdown import DEFINITION as _shutdown_def, shutdown

TOOLS = [_weather_def, _shutdown_def]


def dispatch(tool_name: str, args: dict) -> str:
    if tool_name == "get_weather":
        return get_weather(**args)
    if tool_name == "shutdown":
        return shutdown()
    raise ValueError(f"Unknown tool: {tool_name}")
