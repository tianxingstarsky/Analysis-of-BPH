# 工程调试记录（NOTES）

本页记录研究过程中的调试细节与管线验证，供复现排障参考。结论性内容见 README。

## 1. 三个关键 bug

### 1.1 AMP 混合精度权重发散（RTX 5060 Ti / Blackwell）
- 现象：训练 loss 看似正常下降（0.1 级别），但推理输出全背景；检查 checkpoint 发现
  全部参数为 NaN（epoch 1 即有 2956 个 NaN，epoch 6 几乎全部）。
- 根因：fp16 autocast + RMSprop(momentum=0.999) 在 Blackwell 架构上数值不稳定。
- 修复：关闭 AMP（fp32 训练）。320px 输入 batch 4 显存占用约 5GB，可接受。

### 1.2 小批量 BatchNorm 统计量失效
- 现象：训练 loss 正常，但 eval 模式输出全背景（逐切片 hard Dice = 0），且
  训练进程内部的验证 Dice 恒为 ~0；用独立进程加载同一 checkpoint 评估却又正常——
  前后矛盾。
- 根因：batch=2 时 BatchNorm running statistics 被破坏（增强进一步放大差异），
  train 模式用 batch 统计、eval 模式用 running 统计，两者行为不一致。
- 修复：BatchNorm → InstanceNorm（affine=True，无 running stats，nnU-Net 同款做法）。
  修复后 train/eval 行为一致。

### 1.3 Windows DataLoader 多进程死锁
- 现象：num_workers=os.cpu_count() 时训练在第一个 epoch 前卡死，GPU 0% 但显存已分配。
- 修复：Windows 下 num_workers=0（小数据集本就不需要多进程）。

## 2. 评估口径问题

逐切片 2D Dice 与逐病例 3D Dice 差异巨大（同一模型：2D 0.404 vs 3D 0.766）。原因：
① 腺体底部/顶端薄层切片面积小，边界小偏差即造成单层 Dice 大幅下降；② 中心裁剪训练的
模型在全图推理时构图失配。**结论：分割报告必须注明评估粒度；本仓库以逐病例 3D Dice
为主口径。** 另注意：中心裁剪训练的模型，推理需使用相同的裁剪（评估脚本已内置，
以 GT 质心定位属理想假设，部署可换粗定位/滑窗）。

## 3. 管线验证（ISBI 2012 电镜膜分割）

在投入前列腺数据前，用 U-Net 的"娘家"数据集验证管线：30 张 512×512 电镜图，
50 epoch 训练，逐切片 Dice 0.941（figures/isbi_result_comparison.png）。
该实验同时暴露并驱动修复了 1.1/1.3 两个 bug。

## 4. MedSAM2 微调的梯度问题

SAM2ImagePredictor.predict() 与 _predict() 均带 @torch.no_grad() 装饰器，
外层 torch.enable_grad() 无法穿透。解法：用 `type(predictor)._predict.__wrapped__`
绕过装饰器，并用未装饰的 _prep_prompts 做提示变换（见 medsam2/medsam2_safetune.py）。
另外 SAM2 训练框架的 checkpoint 格式为 {"model": state_dict}，直接保存裸 state_dict
无法被 build_sam2_video_predictor 加载。

## 5. 数据下载

MSD Task05 官方/镜像下载速度不稳定时，可用 Range 分段多线程下载
（scripts/download_data.py，16 线程断点续传，实测单线程 0.3MB/s → 多线程 3-6MB/s）。
国内环境 HuggingFace 建议走 hf-mirror.com 镜像。
