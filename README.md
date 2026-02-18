# NVIDIA Earth-2 Climate Prediction Platform 환경 설정 가이드

## 시스템 환경
- **GPU**: NVIDIA RTX 5090
- **OS**: Ubuntu 24.04 LTS (권장)
- **CUDA**: 12.x
- **Python**: 3.10 이상

## 목차
1. [개요](#개요)
2. [시스템 요구사항](#시스템-요구사항)
3. [CUDA 및 cuDNN 설치](#cuda-및-cudnn-설치)
4. [Python 환경 설정](#python-환경-설정)
5. [NVIDIA Earth-2 설치](#nvidia-earth-2-설치)
6. [데이터셋 다운로드](#데이터셋-다운로드)
7. [모델 실행](#모델-실행)
8. [벤치마크 및 최적화](#벤치마크-및-최적화)
9. [시각화](#시각화)
10. [Docker 환경](#docker-환경-선택사항)
11. [트러블슈팅](#트러블슈팅)
12. [참고 자료](#참고-자료)

---

## 개요

NVIDIA Earth-2는 기후 및 날씨 예측을 위한 AI 기반 오픈소스 플랫폼입니다. FourCastNet, GraphCast 등의 최신 기후 예측 모델을 지원하며, RTX 5090과 같은 고성능 GPU에서 효율적으로 실행됩니다.

### 주요 특징
- **고해상도 기후 예측**: 글로벌 날씨 패턴 예측
- **빠른 추론 속도**: 전통적인 수치 예보 대비 수천 배 빠름
- **다양한 모델 지원**: FourCastNet, GraphCast, Pangu-Weather 등
- **확장 가능한 아키텍처**: 멀티 GPU 지원

---

## 시스템 요구사항

### 하드웨어
- **GPU**: RTX 5090 (24GB VRAM)
- **RAM**: 최소 32GB (권장 64GB)
- **저장공간**: 500GB 이상 (데이터셋 포함)

### 소프트웨어
- Ubuntu 24.04 LTS 또는 Ubuntu 22.04 LTS
- NVIDIA Driver 560.x 이상
- CUDA Toolkit 12.4 이상
- cuDNN 9.x
- Docker (선택사항)

---

## CUDA 및 cuDNN 설치

### 1. NVIDIA 드라이버 설치

```bash
# 기존 드라이버 제거
sudo apt-get purge nvidia-*
sudo apt-get autoremove

# 저장소 추가
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

# RTX 5090 지원 드라이버 설치
sudo apt-get install nvidia-driver-560
sudo reboot

# 설치 확인
nvidia-smi
```

### 2. CUDA Toolkit 설치

```bash
# CUDA 12.4 다운로드 및 설치
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
nvcc --version
```

### 3. cuDNN 설치

```bash
# cuDNN 다운로드 (NVIDIA 개발자 사이트에서 로그인 필요)
# https://developer.nvidia.com/cudnn

# 다운로드한 파일 설치
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.0.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn9-cuda-12
```

---

## Python 환경 설정

### 1. Miniconda 설치

```bash
# Miniconda 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. 가상환경 생성

```bash
# Earth-2 전용 환경 생성
conda create -n earth2 python=3.10
conda activate earth2

# 기본 패키지 설치
conda install -c conda-forge numpy scipy matplotlib pandas
```

### 3. PyTorch 설치 (CUDA 12.4 지원)

```bash
# PyTorch 2.2 이상 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

---

## NVIDIA Earth-2 설치

### 방법 1: GitHub에서 직접 설치

```bash
# Earth-2 저장소 클론
git clone https://github.com/NVIDIA/earth2mip.git
cd earth2mip

# 의존성 설치
pip install -e .

# 추가 패키지 설치
pip install xarray netCDF4 cfgrib eccodes zarr dask
```

### 방법 2: pip 설치

```bash
pip install earth2mip
```

### 검증

```bash
python -c "import earth2mip; print(earth2mip.__version__)"
```

---

## 데이터셋 다운로드

### 1. ERA5 재분석 데이터 다운로드

```bash
# CDS API 설치
pip install cdsapi

# API 키 설정 (~/.cdsapirc 파일 생성)
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_API_KEY_HERE
EOF

# ERA5 데이터 다운로드 스크립트
mkdir -p ~/earth2_data/era5
```

**ERA5 데이터 다운로드 Python 스크립트:**

```python
# download_era5.py
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '2m_temperature', '10m_u_component_of_wind',
            '10m_v_component_of_wind', 'mean_sea_level_pressure',
        ],
        'year': '2024',
        'month': '01',
        'day': ['01', '02', '03'],
        'time': [
            '00:00', '06:00', '12:00', '18:00',
        ],
        'format': 'netcdf',
    },
    'era5_data.nc')
```

### 2. 사전 학습된 모델 다운로드

```bash
# 모델 가중치 저장 디렉토리 생성
mkdir -p ~/earth2_models

# FourCastNet 모델 다운로드
python -c "
from earth2mip.networks import get_model
model = get_model('fcn')
"
```

---

## 모델 실행

### 1. 기본 추론 예제

**inference_example.py:**

```python
import torch
from earth2mip.networks import get_model
from earth2mip.inference_ensemble import run_basic_inference
import numpy as np

# 모델 로드
model = get_model('fcn', device='cuda:0')

# 초기 조건 설정 (예: 랜덤 데이터)
# 실제로는 ERA5 데이터 사용
initial_condition = torch.randn(1, 73, 721, 1440).cuda()

# 7일 예측 (6시간 간격, 28 timesteps)
with torch.no_grad():
    predictions = model(initial_condition, num_steps=28)

print(f"Prediction shape: {predictions.shape}")
```

### 2. 실행 스크립트

```bash
# RTX 5090 최적 설정으로 실행
export CUDA_VISIBLE_DEVICES=0
python inference_example.py
```

### 3. 배치 추론 (최대 처리량)

**batch_inference.py:**

```python
import torch
from earth2mip.networks import get_model

model = get_model('fcn', device='cuda:0')
model.eval()

# RTX 5090 24GB VRAM 최적화
batch_size = 4  # VRAM 사용량에 따라 조정

initial_conditions = torch.randn(batch_size, 73, 721, 1440).cuda()

with torch.cuda.amp.autocast():  # 혼합 정밀도 사용
    with torch.no_grad():
        predictions = model(initial_conditions, num_steps=28)

print(f"Batch prediction completed: {predictions.shape}")
```

### 4. ERA5 데이터를 사용한 실제 예측

**real_world_inference.py:**

```python
import torch
import xarray as xr
from earth2mip.networks import get_model
from earth2mip.initial_conditions import cds

# 모델 로드
model = get_model('fcn', device='cuda:0')

# ERA5 데이터에서 초기 조건 로드
time = "2024-01-01"
data_source = cds.DataSource()
initial_condition = data_source(time)

# GPU로 이동
initial_condition = initial_condition.to('cuda:0')

# 10일 예측
with torch.no_grad():
    predictions = model(initial_condition, num_steps=40)

# 결과 저장
torch.save(predictions, 'predictions_10days.pt')
```

---

## 벤치마크 및 최적화

### RTX 5090 성능 최적화

**optimize_inference.py:**

```python
import torch
import torch.backends.cudnn as cudnn

# cuDNN 벤치마크 활성화
cudnn.benchmark = True
cudnn.deterministic = False

# TF32 활성화 (Ampere 이상)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 메모리 최적화
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.95, 0)

print("Optimization settings applied")
print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
print(f"cuDNN benchmark: {cudnn.benchmark}")
```

### 성능 측정 스크립트

**benchmark.py:**

```python
import torch
import time
from earth2mip.networks import get_model

model = get_model('fcn', device='cuda:0')
model.eval()

# 테스트 데이터
input_data = torch.randn(1, 73, 721, 1440).cuda()

# 워밍업
for _ in range(5):
    with torch.no_grad():
        _ = model(input_data, num_steps=1)

# 벤치마크
num_runs = 20
torch.cuda.synchronize()
start_time = time.time()

for _ in range(num_runs):
    with torch.no_grad():
        predictions = model(input_data, num_steps=28)
    torch.cuda.synchronize()

end_time = time.time()
avg_time = (end_time - start_time) / num_runs

print(f"Average inference time: {avg_time:.3f} seconds")
print(f"Throughput: {1/avg_time:.2f} inferences/second")
```

### GPU 모니터링

```bash
# 실시간 GPU 모니터링
watch -n 1 nvidia-smi

# 상세 모니터링
nvidia-smi dmon -s pucvmet

# 특정 프로세스 모니터링
nvidia-smi pmon
```

---

## 시각화

### 예측 결과 시각화

**visualize.py:**

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch

def plot_prediction(data, variable='t2m', timestep=0, save_path='prediction.png'):
    """기후 예측 결과 시각화"""
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    
    # 경도/위도 그리드 생성
    lons = np.linspace(-180, 180, data.shape[1])
    lats = np.linspace(-90, 90, data.shape[0])
    
    # 데이터 플롯
    im = ax.contourf(lons, lats, data,
                     transform=ccrs.PlateCarree(),
                     cmap='RdBu_r',
                     levels=20)
    
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(im, label=f'{variable}', shrink=0.7)
    plt.title(f'Prediction at timestep {timestep}', fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_timeseries(predictions, lat_idx, lon_idx, variable='temperature'):
    """특정 지점의 시계열 플롯"""
    plt.figure(figsize=(12, 6))
    
    timesteps = range(predictions.shape[0])
    values = predictions[:, lat_idx, lon_idx]
    
    plt.plot(timesteps, values, marker='o')
    plt.xlabel('Timestep (6-hour intervals)')
    plt.ylabel(f'{variable}')
    plt.title(f'Time series at lat_idx={lat_idx}, lon_idx={lon_idx}')
    plt.grid(True)
    plt.savefig('timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: timeseries.png")

# 사용 예제
if __name__ == "__main__":
    # 예측 결과 로드
    predictions = torch.load('predictions_10days.pt')
    
    # GPU에서 CPU로 이동
    predictions_cpu = predictions.cpu().numpy()
    
    # 첫 번째 timestep 시각화
    plot_prediction(predictions_cpu[0, 0, :, :], 
                   variable='2m Temperature',
                   timestep=0)
    
    # 시계열 플롯
    plot_timeseries(predictions_cpu[:, 0, 360, 720])
```

### 애니메이션 생성

**create_animation.py:**

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import numpy as np

def create_animation(predictions, output_file='prediction_animation.mp4'):
    """예측 결과 애니메이션 생성"""
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    
    lons = np.linspace(-180, 180, predictions.shape[2])
    lats = np.linspace(-90, 90, predictions.shape[1])
    
    def update(frame):
        ax.clear()
        ax.coastlines()
        im = ax.contourf(lons, lats, predictions[frame, :, :],
                        transform=ccrs.PlateCarree(),
                        cmap='RdBu_r',
                        levels=20)
        ax.set_title(f'Forecast Hour: {frame * 6}')
        return [im]
    
    anim = animation.FuncAnimation(fig, update, 
                                  frames=predictions.shape[0],
                                  interval=200, blit=False)
    
    anim.save(output_file, writer='ffmpeg', fps=5, dpi=150)
    plt.close()
    print(f"Animation saved: {output_file}")

# 사용 예제
# predictions = torch.load('predictions_10days.pt').cpu().numpy()
# create_animation(predictions[0])  # 첫 번째 변수
```

---

## Docker 환경 (선택사항)

### Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Earth-2 설치
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/earth2mip.git
WORKDIR /workspace/earth2mip
RUN pip install -e .

# 추가 패키지
RUN pip install xarray netCDF4 cfgrib eccodes zarr dask cartopy cdsapi

# 작업 디렉토리 설정
WORKDIR /workspace

CMD ["/bin/bash"]
```

### Docker 빌드 및 실행

```bash
# 이미지 빌드
docker build -t earth2-rtx5090 .

# 컨테이너 실행
docker run --gpus all -it --rm \
    -v ~/earth2_data:/data \
    -v ~/earth2_models:/models \
    -v ~/earth2_output:/output \
    earth2-rtx5090

# 백그라운드 실행
docker run --gpus all -d \
    --name earth2-worker \
    -v ~/earth2_data:/data \
    -v ~/earth2_models:/models \
    earth2-rtx5090 \
    python inference_example.py
```

### Docker Compose (다중 컨테이너 환경)

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  earth2:
    image: earth2-rtx5090
    container_name: earth2-main
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    volumes:
      - ./data:/data
      - ./models:/models
      - ./output:/output
      - ./scripts:/workspace/scripts
    command: bash
    stdin_open: true
    tty: true

  jupyter:
    image: earth2-rtx5090
    container_name: earth2-jupyter
    runtime: nvidia
    ports:
      - "8888:8888"
    volumes:
      - ./data:/data
      - ./models:/models
      - ./notebooks:/workspace/notebooks
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

```bash
# Docker Compose 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

---

## 트러블슈팅

### 1. CUDA Out of Memory

**해결 방법:**

```python
# 메모리 부족 시 배치 크기 줄이기
batch_size = 1  # 또는 2

# Gradient checkpointing 사용 (학습 시)
model.gradient_checkpointing_enable()

# 혼합 정밀도 사용
with torch.cuda.amp.autocast():
    output = model(input)

# 메모리 정리
torch.cuda.empty_cache()

# 메모리 사용량 확인
print(torch.cuda.memory_summary())
```

### 2. cuDNN 오류

```bash
# cuDNN 라이브러리 경로 확인
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# 재설치
sudo apt-get install --reinstall libcudnn9-cuda-12

# 버전 확인
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### 3. 낮은 GPU 사용률

```python
# 데이터 로더 최적화
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=8,  # CPU 코어 수에 따라 조정
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

# 비동기 데이터 전송
for data in dataloader:
    data = data.cuda(non_blocking=True)
```

### 4. 드라이버 호환성 문제

```bash
# 드라이버 버전 확인
nvidia-smi

# CUDA 호환성 확인
nvidia-smi -q | grep "CUDA Version"

# 드라이버 재설치
sudo apt-get purge nvidia-*
sudo apt-get install nvidia-driver-560
sudo reboot
```

### 5. 모델 다운로드 실패

```python
# 수동 다운로드 및 캐시 설정
import os
os.environ['EARTH2MIP_MODEL_REGISTRY'] = '/path/to/models'

# 프록시 설정 (필요 시)
os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'
```

### 6. ERA5 데이터 다운로드 오류

```bash
# CDS API 키 확인
cat ~/.cdsapirc

# 재로그인 필요 시
# https://cds.climate.copernicus.eu/user/login

# 네트워크 타임아웃 증가
# ~/.cdsapirc에 추가:
# timeout: 300
# sleep_max: 120
```

### 7. Import Error

```bash
# 환경 확인
conda activate earth2
which python
python -c "import sys; print(sys.path)"

# 패키지 재설치
pip uninstall earth2mip
pip install earth2mip --no-cache-dir
```

---

## 고급 활용

### 1. 앙상블 예측

**ensemble_forecast.py:**

```python
import torch
from earth2mip.networks import get_model

model = get_model('fcn', device='cuda:0')

# 여러 초기 조건 생성 (perturbation)
num_ensemble = 10
initial_conditions = []

base_ic = torch.randn(1, 73, 721, 1440).cuda()

for i in range(num_ensemble):
    # 작은 perturbation 추가
    perturbed = base_ic + torch.randn_like(base_ic) * 0.01
    initial_conditions.append(perturbed)

# 앙상블 예측 실행
ensemble_predictions = []

for ic in initial_conditions:
    with torch.no_grad():
        pred = model(ic, num_steps=28)
        ensemble_predictions.append(pred)

# 앙상블 평균 및 불확실성
ensemble_mean = torch.stack(ensemble_predictions).mean(dim=0)
ensemble_std = torch.stack(ensemble_predictions).std(dim=0)

print(f"Ensemble mean shape: {ensemble_mean.shape}")
print(f"Ensemble std shape: {ensemble_std.shape}")
```

### 2. 커스텀 메트릭 계산

**metrics.py:**

```python
import torch
import numpy as np

def rmse(pred, target):
    """Root Mean Square Error"""
    return torch.sqrt(torch.mean((pred - target) ** 2))

def acc(pred, target):
    """Anomaly Correlation Coefficient"""
    pred_mean = pred.mean()
    target_mean = target.mean()
    
    numerator = ((pred - pred_mean) * (target - target_mean)).sum()
    denominator = torch.sqrt(
        ((pred - pred_mean) ** 2).sum() * 
        ((target - target_mean) ** 2).sum()
    )
    
    return numerator / denominator

def evaluate_forecast(predictions, ground_truth):
    """예측 성능 평가"""
    metrics = {
        'rmse': rmse(predictions, ground_truth).item(),
        'acc': acc(predictions, ground_truth).item(),
        'bias': (predictions - ground_truth).mean().item()
    }
    
    return metrics

# 사용 예제
# metrics = evaluate_forecast(predictions, ground_truth)
# print(metrics)
```

### 3. 멀티 GPU 활용

**multi_gpu_inference.py:**

```python
import torch
import torch.nn as nn
from earth2mip.networks import get_model

# DataParallel 사용
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = get_model('fcn', device='cuda:0')
    model = nn.DataParallel(model)
else:
    model = get_model('fcn', device='cuda:0')

# 대규모 배치 처리
batch_size = 8  # GPU 개수에 따라 조정
input_data = torch.randn(batch_size, 73, 721, 1440).cuda()

with torch.no_grad():
    predictions = model(input_data, num_steps=28)

print(f"Processed batch shape: {predictions.shape}")
```

---

## 프로젝트 구조 예시

```
earth2-project/
├── data/
│   ├── era5/              # ERA5 데이터
│   ├── processed/         # 전처리된 데이터
│   └── cache/             # 캐시 파일
├── models/
│   ├── checkpoints/       # 모델 체크포인트
│   └── configs/           # 설정 파일
├── scripts/
│   ├── download_era5.py
│   ├── inference_example.py
│   ├── batch_inference.py
│   ├── benchmark.py
│   ├── visualize.py
│   └── ensemble_forecast.py
├── notebooks/             # Jupyter 노트북
├── outputs/
│   ├── predictions/       # 예측 결과
│   ├── visualizations/    # 시각화 결과
│   └── metrics/           # 평가 지표
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 성능 벤치마크 (RTX 5090)

### 예상 성능 지표

| 모델 | 해상도 | 배치 크기 | 추론 시간 (10일) | VRAM 사용량 |
|------|--------|-----------|-----------------|-------------|
| FourCastNet | 0.25° | 1 | ~15초 | ~18GB |
| FourCastNet | 0.25° | 4 | ~45초 | ~22GB |
| GraphCast | 0.25° | 1 | ~20초 | ~20GB |
| Pangu-Weather | 0.25° | 1 | ~18초 | ~19GB |

*실제 성능은 시스템 구성에 따라 달라질 수 있습니다.*

---

## 참고 자료

### 공식 문서
- [NVIDIA Earth-2 공식 사이트](https://www.nvidia.com/en-us/high-performance-computing/earth-2/)
- [Earth-2 MIP Documentation](https://nvidia.github.io/earth2mip/)
- [Earth-2 MIP GitHub](https://github.com/NVIDIA/earth2mip)

### 논문
- [FourCastNet: A Global Data-driven High-resolution Weather Model](https://arxiv.org/abs/2202.11214)
- [GraphCast: Learning skillful medium-range global weather forecasting](https://arxiv.org/abs/2212.12794)
- [Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast](https://arxiv.org/abs/2211.02556)

### 데이터셋
- [ERA5 Reanalysis Data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)
- [CMIP6 Climate Models](https://esgf-node.llnl.gov/projects/cmip6/)

### 커뮤니티
- [NVIDIA Developer Forum](https://forums.developer.nvidia.com/)
- [Earth-2 GitHub Discussions](https://github.com/NVIDIA/earth2mip/discussions)

---

## FAQ

**Q: RTX 5090 없이도 실행 가능한가요?**
A: 네, 하지만 낮은 해상도나 작은 배치 크기를 사용해야 할 수 있습니다. 최소 16GB VRAM 권장.

**Q: Windows에서 실행 가능한가요?**
A: WSL2를 통해 가능하지만, 네이티브 Linux 환경 권장.

**Q: 상업적 사용이 가능한가요?**
A: 각 모델의 라이센스를 확인하세요. 일부는 연구용으로만 제한될 수 있습니다.

**Q: 실시간 예측이 가능한가요?**
A: RTX 5090에서 10일 예측이 약 15-20초 소요되므로 준실시간 예측이 가능합니다.

---

## 기여

이슈 리포트나 개선 제안은 GitHub Issues를 통해 제출해주세요.

---

## 라이센스

이 가이드는 MIT 라이센스를 따르며, NVIDIA Earth-2는 별도의 라이센스를 따릅니다.

---

## 작성자

- RTX 5090 최적화: 나무 (Embedded Systems Engineer)
- 작성일: 2026-02-18
- 버전: 1.0

---

## 업데이트 로그

- 2026-02-18: 초기 버전 작성
  - RTX 5090 환경 설정 가이드
  - CUDA 12.4 및 cuDNN 9.x 설정
  - 기본 추론 및 최적화 예제
  - Docker 환경 구성
