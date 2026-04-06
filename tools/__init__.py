from tools.weather import DEFINITION as _weather_def, get_weather
from tools.shutdown import DEFINITION as _shutdown_def, shutdown
from tools.reset import DEFINITION as _reset_def, reset_history
from tools.search import DEFINITION as _search_def, search
from tools.notes import SAVE_DEFINITION as _save_note_def, RECALL_DEFINITION as _recall_notes_def, save_note, recall_notes

TOOLS = [_weather_def, _shutdown_def, _reset_def, _search_def, _save_note_def, _recall_notes_def]


def dispatch(tool_name: str, args: dict) -> str:
    if tool_name == "get_weather":
        return get_weather(**args)
    if tool_name == "shutdown":
        return shutdown()
    if tool_name == "reset_history":
        return reset_history()
    if tool_name == "search":
        return search(**args)
    if tool_name == "save_note":
        return save_note(**args)
    if tool_name == "recall_notes":
        return recall_notes(**args)
    raise ValueError(f"Unknown tool: {tool_name}")
