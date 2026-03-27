import os
import subprocess
import shutil
import sys
import time
import glob

# ===================== 配置区域 =====================
SESSION_NAME = "EXP_QUEUE_DYNAMIC"
TASKS_FILE = "tasks.txt"
NUM_PARALLEL = 8
TASKS_DIR = "dynamic_tasks"


# ===================================================

def run_worker():
    """Worker (打工人) 模式：在每个 Tmux 窗格中独立运行，不断抢任务"""
    todo_dir = os.path.join(TASKS_DIR, "todo")
    processing_dir = os.path.join(TASKS_DIR, "processing")
    done_dir = os.path.join(TASKS_DIR, "done")

    while True:
        # 1. 查找所有还没跑的任务
        tasks = sorted(glob.glob(os.path.join(todo_dir, "*.sh")))
        if not tasks:
            print(f"[{time.strftime('%H:%M:%S')}] 🎉 任务池已空，当前窗格结束工作并闲置。")
            break

        task_path = tasks[0]
        task_name = os.path.basename(task_path)
        processing_path = os.path.join(processing_dir, task_name)
        done_path = os.path.join(done_dir, task_name)

        try:
            # 2. 原子操作抢占任务：尝试将任务移入 processing 目录。
            # 只有抢成功（没报异常）的窗格，才真正拥有该任务的执行权。
            os.rename(task_path, processing_path)
        except (FileNotFoundError, FileExistsError):
            # 任务已经被其他速度更快的窗格抢走，继续下一次循环找新任务
            continue

        # 3. 读取命令
        with open(processing_path, 'r', encoding='utf-8') as f:
            cmd = f.read().strip()

        print(f"[{time.strftime('%H:%M:%S')}] 🚀 成功抢到任务: {task_name}, 开始执行...")
        print(f"💻 执行命令: {cmd}")

        # 4. 执行命令 (不论成功还是报错，Python 会捕捉状态并继续执行，不会卡死)
        subprocess.run(cmd, shell=True)

        # 5. 移入完成目录
        os.rename(processing_path, done_path)

        print(f"[{time.strftime('%H:%M:%S')}] ✅ 任务 {task_name} 处理完毕！")
        print('\n=======================================\n')

        # 6. 缓冲时间：释放显存等系统资源，防止连续启动爆显存
        time.sleep(2)


def main():
    """主程序模式：负责创建任务池、开启 Tmux 和分发 Worker"""
    # 如果检测到 --worker 参数，说明是被分配到 Tmux 窗格里运行的子进程
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker()
        return

    # -------- 以下为主控制程序的逻辑 --------
    if not os.path.exists(TASKS_FILE):
        print(f"❌ 找不到 {TASKS_FILE}，请先生成该文件！")
        exit(1)

    with open(TASKS_FILE, 'r', encoding='utf-8') as f:
        all_cmds = [line.strip() for line in f if line.strip()]

    print(f"统计：共发现 {len(all_cmds)} 个任务，准备动态分发到 {NUM_PARALLEL} 个 Tmux 窗格...")

    # 1. 初始化任务目录
    shutil.rmtree(TASKS_DIR, ignore_errors=True)
    os.makedirs(os.path.join(TASKS_DIR, "todo"))
    os.makedirs(os.path.join(TASKS_DIR, "processing"))
    os.makedirs(os.path.join(TASKS_DIR, "done"))

    # 2. 把每个命令变成一个独立的文件放入 todo 目录
    for i, cmd in enumerate(all_cmds):
        task_file = os.path.join(TASKS_DIR, "todo", f"task_{i:04d}.sh")
        with open(task_file, 'w', encoding='utf-8') as f:
            f.write(cmd)

    # 3. 创建 Tmux 会话并分屏
    subprocess.run(f"tmux kill-session -t {SESSION_NAME}", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f"tmux new-session -d -s {SESSION_NAME}", shell=True)

    current_script = os.path.abspath(__file__)

    for i in range(NUM_PARALLEL):
        if i > 0:
            subprocess.run(f"tmux split-window -t {SESSION_NAME}", shell=True)
            subprocess.run(f"tmux select-layout -t {SESSION_NAME} tiled", shell=True)

        # 4. 让每个窗格去运行本脚本的 "打工人模式"
        pane_cmd = f"python {current_script} --worker"
        subprocess.run(f"tmux send-keys -t {SESSION_NAME}.{i} \"{pane_cmd}\" C-m", shell=True)

    print(f"✅ 任务动态抢占机制已启动！")
    print(f"👉 输入以下命令实时查看 {NUM_PARALLEL} 个并发窗口（它们正在自动争抢任务）：")
    print(f"   tmux attach -t {SESSION_NAME}")


if __name__ == "__main__":
    main()