# MedSAM2 部署与评估

1. 克隆官方仓库到本仓库同级目录（或任意位置后设置环境变量 MEDSAM2_DIR）：
   git clone https://github.com/bowang-lab/MedSAM2.git
2. 安装：cd MedSAM2 && SAM2_BUILD_CUDA=0 pip install -e . --no-build-isolation
3. 权重（国内镜像，149MB）：
   curl -L -o MedSAM2/checkpoints/MedSAM2_latest.pt \
     https://hf-mirror.com/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt
4. 零样本评估（boxall=每层框提示 / propagate=单框传播；jitter=框抖动鲁棒性）：
   python medsam2/medsam2_eval.py --box gt
5. 安全微调（冻结 encoder/memory，仅训 decoder）：python medsam2/medsam2_safetune.py
6. 可视化：python medsam2/medsam2_overlay.py
