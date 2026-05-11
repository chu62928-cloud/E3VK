#!/bin/bash

# 配置部分
LOG_FILE="pipeline_run.log"  # 请确保你的日志文件名正确
INTERVAL=300                  # 每 5 分钟 (300秒) 检查一次

echo "===================================================="
echo "   scTenifold 运行进度实时监控启动 (按 Ctrl+C 停止)   "
echo "===================================================="

while true; do
    echo "----------------------------------------------------"
    echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 1. 检查日志最后 3 行
    echo "[日志最新进度]:"
    if [ -f "$LOG_FILE" ]; then
        tail -n 3 "$LOG_FILE"
    else
        echo "等待日志文件生成..."
    fi

    # 2. 检查 Ray 进程状态
    echo -e "\n[Ray 计算节点状态]:"
    RAY_PROCESSES=$(ps -eo pcpu,pmem,comm --sort=-pcpu | grep "ray::" | head -n 5)
    if [ -z "$RAY_PROCESSES" ]; then
        echo "警告: 未检测到 Ray 计算进程，可能已运行结束或出错。"
    else
        echo "CPU%  MEM%  COMMAND"
        echo "$RAY_PROCESSES"
    fi

    # 3. 统计当前存活的并行任务数
    TASK_COUNT=$(ps -ef | grep "ray::pc_net_par" | grep -v grep | wc -l)
    echo -e "\n当前活跃并行任务数: $TASK_COUNT"
    
    echo "----------------------------------------------------"
    sleep $INTERVAL
done
