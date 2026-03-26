import os
import subprocess
import shutil
import time

# ===================== 配置区域 =====================
SESSION_NAME = "EXP_QUEUE_DYNAMIC"  # Tmux会话名
TASKS_FILE = "tasks.txt"             # 任务列表文件
NUM_PARALLEL = 8                     # 并行窗格数
TASKS_DIR = "dynamic_tasks"          # 待执行任务目录
COMPLETED_DIR = "dynamic_completed"  # 执行中/已完成任务目录
# ===================================================

def prepare_task_queue():
    """准备任务队列：将tasks.txt的每个命令拆分为单独的任务文件"""
    # 校验任务文件
    if not os.path.exists(TASKS_FILE):
        print(f"❌ 找不到 {TASKS_FILE}，请先生成该文件！")
        exit(1)

    # 清理旧的任务目录
    shutil.rmtree(TASKS_DIR, ignore_errors=True)
    shutil.rmtree(COMPLETED_DIR, ignore_errors=True)
    os.makedirs(TASKS_DIR)
    os.makedirs(COMPLETED_DIR)

    # 读取所有任务，写入单独的任务文件
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        all_cmds = [line.strip() for line in f if line.strip()]

    for idx, cmd in enumerate(all_cmds, 1):
        task_file = os.path.join(TASKS_DIR, f"task_{idx:03d}.sh")
        with open(task_file, "w", encoding="utf-8") as f:
            f.write(cmd + "\n")

    print(f"📋 统计：共发现 {len(all_cmds)} 个任务，准备启动 {NUM_PARALLEL} 个动态调度窗格...")
    return len(all_cmds)

def get_worker_code():
    """生成每个Tmux窗格运行的Worker代码（动态抢任务逻辑）"""
    return f'''
import os
import subprocess
import time

TASKS_DIR = "{TASKS_DIR}"
COMPLETED_DIR = "{COMPLETED_DIR}"

def run_worker():
    while True:
        # 1. 扫描待执行任务（按名称排序，保证大致顺序）
        task_files = sorted([
            f for f in os.listdir(TASKS_DIR)
            if f.startswith("task_") and f.endswith(".sh")
        ])

        if not task_files:
            # 没有待执行任务，检查是否有任务还在运行（避免提前退出）
            running_tasks = [f for f in os.listdir(COMPLETED_DIR) if f.endswith(".running")]
            if not running_tasks:
                print("✅ 所有任务已完成，本窗格退出。")
                break
            else:
                time.sleep(2)  # 稍等再检查
                continue

        # 2. 尝试“抢”任务（利用os.rename的原子性做锁）
        task_taken = False
        for task_file in task_files:
            src_path = os.path.join(TASKS_DIR, task_file)
            running_path = os.path.join(COMPLETED_DIR, task_file + ".running")
            try:
                os.rename(src_path, running_path)  # 原子操作：抢到任务
                task_taken = True
                break
            except OSError:
                continue  # 被其他窗格抢走了，试下一个

        if not task_taken:
            time.sleep(1)
            continue

        # 3. 执行抢到的任务
        print(f"🚀 窗格 {{os.getpid()}} 开始执行: {{task_file}}")
        with open(running_path, "r", encoding="utf-8") as f:
            cmd = f.read().strip()

        # 运行任务（继承当前环境变量，失败也继续）
        returncode = subprocess.call(cmd, shell=True)

        # 4. 标记任务完成
        done_path = os.path.join(COMPLETED_DIR, task_file + ".done")
        os.rename(running_path, done_path)
        print(f"✅ 窗格 {{os.getpid()}} 完成任务: {{task_file}} (返回码: {{returncode}})")

        # 5. 缓冲时间：回收显存，避免下一个任务立即启动报错
        print("⏳ 等待2秒，回收系统资源...\\n")
        time.sleep(2)

if __name__ == "__main__":
    run_worker()
'''

def main():
    # 1. 准备任务队列
    total_tasks = prepare_task_queue()

    # 2. 清理旧的Tmux会话
    subprocess.run(
        f"tmux kill-session -t {SESSION_NAME}",
        shell=True,
        stderr=subprocess.DEVNULL
    )

    # 3. 新建Tmux会话（后台运行）
    subprocess.run(f"tmux new-session -d -s {SESSION_NAME}", shell=True)

    # 4. 创建分屏并启动Worker
    worker_code = get_worker_code()
    for i in range(NUM_PARALLEL):
        if i > 0:
            subprocess.run(f"tmux split-window -t {SESSION_NAME}", shell=True)
            subprocess.run(f"tmux select-layout -t {SESSION_NAME} tiled", shell=True)

        # 发送Worker命令到窗格（用python -c执行内嵌代码）
        worker_cmd = f"python -c \"{worker_code}\""
        subprocess.run(
            f"tmux send-keys -t {SESSION_NAME}.{i} \"{worker_cmd}\" C-m",
            shell=True
        )

    # 5. 输出使用指引
    print(f"\\n✅ 动态任务调度已启动！")
    print(f"👉 查看实时运行: tmux attach -t {SESSION_NAME}")
    print(f"📂 任务状态查询:")
    print(f"   待执行: ls {TASKS_DIR}/")
    print(f"   执行中: ls {COMPLETED_DIR}/*.running")
    print(f"   已完成: ls {COMPLETED_DIR}/*.done")
    print(f"   统计进度: ls {COMPLETED_DIR}/*.done | wc -l （已完成数）/ {total_tasks}（总数）")

if __name__ == "__main__":
    main()