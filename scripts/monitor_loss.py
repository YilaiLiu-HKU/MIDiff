#!/usr/bin/env python3
"""
实时监控训练损失并动态绘制损失曲线
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import argparse

def monitor_loss_csv(csv_path, update_interval=10):
    """
    实时监控CSV文件中的损失数据并动态绘制
    
    Args:
        csv_path: 损失CSV文件路径
        update_interval: 更新间隔（秒）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('实时训练损失监控', fontsize=16)
    
    # 初始化空数据
    steps = []
    losses = {'loss': [], 'mse': [], 'vb': [], 'aux_loss': [], 'cos': []}
    
    def update_plot(frame):
        try:
            if os.path.exists(csv_path):
                # 读取CSV数据
                df = pd.read_csv(csv_path)
                
                # 更新数据
                steps = df['step'].tolist()
                for key in losses.keys():
                    if key in df.columns:
                        losses[key] = df[key].tolist()
                
                # 清除之前的图
                for ax in axes.flat:
                    ax.clear()
                
                if len(steps) > 0:
                    # 总损失
                    axes[0, 0].plot(steps, losses['loss'], 'b-', linewidth=2, label='Total Loss')
                    axes[0, 0].set_title('总损失')
                    axes[0, 0].set_xlabel('步数')
                    axes[0, 0].set_ylabel('损失值')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
                    
                    # MSE损失
                    axes[0, 1].plot(steps, losses['mse'], 'r-', linewidth=2, label='MSE Loss')
                    axes[0, 1].set_title('MSE损失')
                    axes[0, 1].set_xlabel('步数')
                    axes[0, 1].set_ylabel('损失值')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
                    
                    # VB损失
                    axes[1, 0].plot(steps, losses['vb'], 'g-', linewidth=2, label='VB Loss')
                    axes[1, 0].set_title('VB损失')
                    axes[1, 0].set_xlabel('步数')
                    axes[1, 0].set_ylabel('损失值')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True)
                    
                    # 辅助损失
                    axes[1, 1].plot(steps, losses['aux_loss'], 'm-', linewidth=2, label='Aux Loss')
                    axes[1, 1].set_title('辅助损失')
                    axes[1, 1].set_xlabel('步数')
                    axes[1, 1].set_ylabel('损失值')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)
                    
                    # 添加最新损失值标注
                    if len(steps) > 0:
                        latest_step = steps[-1]
                        axes[0, 0].text(0.02, 0.98, f'最新步数: {latest_step}', 
                                       transform=axes[0, 0].transAxes, 
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        
                        for i, (key, values) in enumerate(losses.items()):
                            if len(values) > 0:
                                ax = axes[i//2, i%2]
                                latest_value = values[-1]
                                ax.text(0.02, 0.02, f'最新值: {latest_value:.6f}', 
                                       transform=ax.transAxes, 
                                       verticalalignment='bottom',
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                
                plt.tight_layout()
                
        except Exception as e:
            print(f"更新图表时出错: {e}")
    
    # 创建动画
    ani = FuncAnimation(fig, update_plot, interval=update_interval*1000, blit=False)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='实时监控训练损失')
    parser.add_argument('--csv_path', type=str, required=True, 
                       help='损失CSV文件路径')
    parser.add_argument('--update_interval', type=int, default=10,
                       help='更新间隔（秒）')
    
    args = parser.parse_args()
    
    print(f"开始监控损失文件: {args.csv_path}")
    print(f"更新间隔: {args.update_interval}秒")
    print("按 Ctrl+C 停止监控")
    
    try:
        monitor_loss_csv(args.csv_path, args.update_interval)
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    main() 