# train.py
from ultralytics import YOLO
import torch

def main():
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载模型
    model = YOLO('yolov11n.yaml') # 使用nano模型，最小最快
    
    # 训练参数（针对8GB显存优化）
    results = model.train(
        data='dataset.yaml',  # 数据集配置文件
        epochs=200,                     # 训练轮数
        batch=32,                        # 批次大小（8GB显存建议8-16）
        imgsz=640,                      # 图像尺寸
        device=device,                  # 设备
        
        # 显存优化策略
        workers=4,                      # 数据加载线程（减少显存占用）
        cache=False,                    # 不缓存图像（节省显存）
        rect=True,                      # 矩形训练（节省显存）
        
        # 优化器设置
        optimizer='auto',               # 自动选择优化器
        lr0=0.01,                       # 初始学习率
        lrf=0.01,                       # 最终学习率因子
        momentum=0.937,                 # SGD动量
        weight_decay=0.0005,            # 权重衰减
        
        # 数据增强（8GB显存建议适度使用）
        mosaic=0.0,                     # mosaic增强概率
        mixup=0.0,                      # mixup增强（显存紧张时可关闭）
        copy_paste=0.0,                 # copy-paste增强（显存紧张时可关闭）
        amp=True,                        # 自动混合精度（节省显存，提升速度）
        
        # 训练策略
        warmup_epochs=3,                # 预热轮数
        warmup_momentum=0.8,            # 预热动量
        warmup_bias_lr=0.1,             # 预热偏置学习率
        
        # 验证和保存
        val=True,                       # 每轮验证
        save=True,                      # 保存模型
        save_period=10,                 # 每10轮保存一次
        plots=True,                     # 绘制训练曲线
        project='runs/train',           # 保存目录
        name='yolov11n_baseline',         # 实验名称
        exist_ok=True,                  # 覆盖已有结果
        
        # 其他
        seed=42,                        # 随机种子
        verbose=True,                   # 显示详细信息
    )
    
    print("\n训练完成！")
    print(f"最佳模型保存在: runs/train/yolov11n_baseline/weights/best.pt")

if __name__ == '__main__':
    main()