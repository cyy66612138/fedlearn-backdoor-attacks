import os
import subprocess

# 1. 配置
SESSION_NAME = "EXP_QUEUE"
TASKS_FILE = "tasks.txt"
NUM_PARALLEL = 8  # 始终保持 8 个窗口并行

# 2. 读取所有命令
if not os.path.exists(TASKS_FILE):
    print(f"❌ 找不到 {TASKS_FILE}，请先生成该文件！")
    exit()

with open(TASKS_FILE, 'r') as f:
    all_cmds = [line.strip() for line in f if line.strip()]

print(f"统计：共发现 {len(all_cmds)} 个任务，准备分发到 {NUM_PARALLEL} 个 Tmux 窗格...")

# 3. 将任务分配给 8 个队列
queues = [all_cmds[i::NUM_PARALLEL] for i in range(NUM_PARALLEL)]

# 4. 创建 Tmux 会话并分屏
subprocess.run(f"tmux kill-session -t {SESSION_NAME}", shell=True, stderr=subprocess.DEVNULL)
subprocess.run(f"tmux new-session -d -s {SESSION_NAME}", shell=True)

for i in range(NUM_PARALLEL):
    # 如果不是第一个，就创建新窗格
    if i > 0:
        subprocess.run(f"tmux split-window -t {SESSION_NAME}", shell=True)
        subprocess.run(f"tmux select-layout -t {SESSION_NAME} tiled", shell=True)

    # 构建该窗格的串行命令流：cmd1 && cmd2 && cmd3...
    # 这样当 cmd1 跑完，该窗格会自动开始跑下一个，而不会占用新的显存
    pane_cmds = " && ".join(queues[i])

    # 发送到 Tmux 窗格执行
    subprocess.run(f"tmux send-keys -t {SESSION_NAME}.{i} \"{pane_cmds}\" C-m", shell=True)

print(f"✅ 任务分发完毕！")
print(f"👉 输入以下命令实时查看 8 个并发窗口：")
print(f"   tmux attach -t {SESSION_NAME}")