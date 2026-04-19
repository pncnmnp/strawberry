from tools.weather import get_weather
from tools.shutdown import shutdown
from tools.reset import reset_history
from tools.search import search
from tools.notes import save_note, recall_notes
from tools.mute import mute
from tools.date_time import get_current_datetime
from tools.music import music_play, music_control, music_set_volume, music_now_playing, music_search_library
from tools.thinking import deep_think

TOOL_FUNCTIONS = [get_weather, shutdown, reset_history, search, save_note, recall_notes, mute, get_current_datetime,
                  music_play, music_control, music_set_volume, music_now_playing, music_search_library, deep_think]
