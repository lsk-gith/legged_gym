# 在 play() 函数开始处，定义状态变量（可用列表以便在回调中修改）
cmd_state = {"x": 0.0, "y": 0.0, "yaw": 0.0}

def keyboard_callback(event):
    """Isaac Gym 键盘回调，event 包含 'inputEvent' 和 'key' 信息"""
    key = event['key']
    is_pressed = (event['inputEventType'] == 1)  # 1 = press, 0 = release
    max_speed = 0.8
    yaw_rate = 1.0
    
    if key == ord('W'):
        cmd_state["x"] = max_speed if is_pressed else 0.0
    elif key == ord('S'):
        cmd_state["x"] = -max_speed if is_pressed else 0.0
    elif key == ord('A'):
        cmd_state["y"] = max_speed if is_pressed else 0.0
    elif key == ord('D'):
        cmd_state["y"] = -max_speed if is_pressed else 0.0
    elif key == ord('Z'):
        cmd_state["yaw"] = yaw_rate if is_pressed else 0.0
    elif key == ord('X'):
        cmd_state["yaw"] = -yaw_rate if is_pressed else 0.0
    # 可选：按其他键复位指令
    elif key == ord(' '):  # 空格键复位
        cmd_state["x"] = 0.0
        cmd_state["y"] = 0.0
        cmd_state["yaw"] = 0.0

# 在环境创建之后，仿真循环之前，注册回调
env.gym.subscribe_viewer_keyboard_event(env.viewer, keyboard_callback)