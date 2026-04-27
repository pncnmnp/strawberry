from tools.weather import get_weather
from tools.shutdown import shutdown
from tools.powercycle import powercycle
from tools.reset import reset_history
from tools.search import search
from tools.notes import save_note, recall_notes, add_todo, recall_todos, complete_todo
from tools.mute import mute
from tools.date_time import get_current_datetime
from tools.music import music_play, music_control, music_set_volume, music_now_playing, music_search_library, music_rebuild_index
from tools.thinking import deep_think
from tools.code import run_python, install_package

TOOL_FUNCTIONS = [get_weather, shutdown, powercycle, reset_history, search, save_note, recall_notes, add_todo, recall_todos, complete_todo, mute, get_current_datetime,
                  music_play, music_control, music_set_volume, music_now_playing, music_search_library, music_rebuild_index, deep_think,
                  run_python, install_package]
