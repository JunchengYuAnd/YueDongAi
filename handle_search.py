from pywinauto import findwindows

# 列出所有窗口
windows = findwindows.find_elements()
for win in windows:
    print(f"窗口标题: {win.name}, 句柄: {win.handle}")
