from tools.weather import get_weather
from tools.shutdown import shutdown
from tools.reset import reset_history
from tools.search import search
from tools.notes import save_note, recall_notes
from tools.mute import mute

TOOL_FUNCTIONS = [get_weather, shutdown, reset_history, search, save_note, recall_notes, mute]
