# Target Detection Integration 문서

## 1. 현재 구조 분석

### 1.1 `repo/Target_detection/` — Interactive Target Detection Webapp

Flask 기반 독립 웹앱 (`web_app.py`, ~3900줄)으로, 초분광 영상에서 interactive target detection을 수행함.

#### 핵심 기능

**Positive/Negative 클릭 인터랙션:**
- **좌클릭 (button=0)** → Positive point (초록색 마커) — "이것이 타겟이다"
- **우클릭 (button=2)** → Negative point (빨간색 마커) — "이것은 타겟이 아니다"
- 클릭 시 500ms 디바운스 후 자동으로 detection 재실행
- Undo 기능 (`history` 배열로 마지막 클릭 취소)

```javascript
// web_app.py 내 JS (line 2409-2423)
function handlePromptClick(ox, oy, btn){
  if(btn === 0){
    posPoints.push({x:ox, y:oy});      // 좌클릭 = positive
    history.push('pos');
  } else if(btn === 2){
    negPoints.push({x:ox, y:oy});      // 우클릭 = negative
    history.push('neg');
  }
  scheduleAutoRun();  // 500ms 후 자동 실행
}
```

**원본 이미지 + Mask 동시 클릭 가능:**
- 원본 이미지 캔버스: `canvasToOrig(e)` → 원본 좌표로 변환 → `handlePromptClick()`
- Score Map 패널: `viewportToOrig(e, wrap)` → 원본 좌표로 변환 → `handlePromptClick()`
- Seg Map 패널: 동일한 방식으로 원본 좌표로 변환 → `handlePromptClick()`
- **3개 패널 모두 동기화된 pan/zoom** (`panX`, `panY`, `zoomLevel` 공유)

```javascript
// Score Map, Seg Map 패널 클릭 핸들러 (line 3069-3087)
['zoom-wrap-score','zoom-wrap-seg'].forEach(id=>{
  wrap.addEventListener('mousedown', e=>{
    const {x, y} = viewportToOrig(e, wrap);  // 원본 좌표로 변환
    handlePromptClick(x, y, e.button);        // 동일한 핸들러 사용
  });
});
```

#### 모델 구조

| 모델 | 파일 | 설명 |
|------|------|------|
| SAM | `models/sam.py` | Segment Anything Model |
| SAM2 | `models/sam2.py` | SAM v2 |
| SAMitizer | `models/samitizer.py` | SAM + spectral fusion |
| **NewMethod MLP** | `models/new_method_mlp.py` | **MLP projector + AMF/ACE (핵심)** |
| NewMethod Bilinear | `models/new_method_bilinear.py` | Bilinear interaction variant |
| Classical | `models/classical_detectors.py` | SAM, ACE, MF, CEM |
| OSP+AMF | `models/osp_amf.py` | Core spectral detectors |

#### 실행 파이프라인

```
사용자 클릭 (pos/neg)
    ↓
POST /detect_one {model, pos:[[x,y],...], neg:[[x,y],...]}
    ↓
webapp_core/model_runner.py: run_single_model()
    ├─ PCA-RGB 변환 (SAM 모델용)
    ├─ models/webapp_registry.py: predict_webapp_model()
    │   └─ 모델별 predict() 호출
    ├─ 결과 upsampling
    └─ 시간 측정
    ↓
webapp_core/visualization.py: 히트맵 + 세그맵 생성
    ↓
JSON 응답 {heatmap: base64, seg: base64, ms, threshold, iou, ...}
```

---

### 1.2 MLP Projector 모델 상세 (`models/new_method_mlp.py`)

**핵심 아이디어**: D차원 스펙트럼을 10차원으로 학습 가능한 MLP로 투영하여, 타겟/배경 분리를 극대화한 후 AMF/ACE 수행.

#### LearnableProjector 아키텍처 (line 152-191)

```
Input (D-dim spectrum)
    ↓
LayerNorm(D)
    ↓
Linear(D → hidden_dim=32)
    ↓
ReLU
    ↓
Linear(32 → out_dim=10)
    ↓
Output (10-dim projected features)
```

#### 학습 과정 (line 616-958, `detect_step_new_method`)

1. **입력 정규화**: Raw hyperspectral bands를 band별 정규화
2. **Projector 학습** (100 iterations, Adam lr=0.001):
   - `L_det` (Pairwise Ranking Loss): positive 점수 > negative/background 점수
   - `L_decorr` (Decorrelation Loss): 투영된 차원 간 상관관계 최소화
   - `L_var` (Variance Floor Loss): 각 차원의 분산 유지
   - `L_reg` (L2 Regularization): 가중치 크기 제한
3. **투영 적용**: 모든 픽셀을 MLP에 통과 → 10차원 feature map
4. **공분산 추정**: Ledoit-Wolf shrinkage로 배경 공분산 행렬 추정
5. **Detection 수행**: 투영된 공간에서 AMF 또는 ACE 스코어 계산
6. **Threshold**: Otsu method로 이진화

#### 모델 변형

```python
class NewMethodMLP:       # AMF 기반 scoring
    detector = "amf"

class NewMethodMLPACE:    # ACE 기반 scoring
    detector = "ace"
```

---

### 1.3 현재 Platform의 Target Detection

#### Backend (`backend/services/target_detection.py`)

5개 classical detector 구현:
- **SAM**: Spectral Angle Mapper (각도 유사도)
- **ACE**: Adaptive Cosine Estimator (배경 정규화)
- **RXD**: Reed-Xiaoli Detector (이상치 탐지)
- **CEM**: Constrained Energy Minimization (최적 필터)
- **MF**: Matched Filter (centered 스펙트럼)

공통 파이프라인:
```python
def run_target_detection(raster_path, target_points_latlon, algorithm, ...):
    # 1. 래스터 로드
    # 2. lat/lon → pixel 좌표 변환
    # 3. 타겟 스펙트럼 추출 (복수 포인트 평균)
    # 4. 배경 통계 추정 (공분산 + 평균, 타겟 영역 제외)
    # 5. 알고리즘 실행 → score map
    # 6. Otsu/percentile threshold → binary mask
    # 7. 시각화 생성
```

#### API (`backend/api/routes_target_detection.py`)

| Endpoint | Method | 용도 |
|----------|--------|------|
| `/api/target-detection/algorithms` | GET | 사용 가능 알고리즘 목록 |
| `/api/target-detection/run` | POST | Detection 실행 |
| `/api/target-detection/apply-threshold` | POST | Threshold 재조정 |
| `/api/target-detection/get-spectrum` | POST | 특정 좌표 스펙트럼 조회 |

#### Frontend (`frontend/js/target-detection.js`)

- `TargetDetectionController` 클래스
- **Positive 클릭만 지원** (빨간 원 마커 + 번호 라벨)
- Negative 클릭 미구현
- Threshold 조정: colorbar min/max 드래그
- 결과 오버레이: Leaflet ImageOverlay

#### 현재 스키마 (`backend/api/schemas.py`)

```python
class TargetDetectionRequest(BaseModel):
    image_id: str
    bbox: List[float]
    target_points: List[TargetPoint]    # positive만
    algorithm: str = "SAM"
    auto_threshold: Optional[bool] = True
    selected_bands: Optional[List[int]] = None
    # negative_points 없음
```

---

## 2. 통합 계획

### 2.1 `repo/Target_detection/`를 모델 라이브러리로 재구성

**목표**: repo가 순수 모델 추론 기능만 제공하고, Platform이 I/O/UI/API를 담당.

**새 파일**: `repo/Target_detection/inference.py`

```python
"""Platform에서 호출하는 단일 진입점."""

def detect(
    cube: np.ndarray,                          # (H, W, C) 래스터 데이터
    pos_points: List[Tuple[int, int]],         # [(row, col), ...] positive 포인트
    neg_points: List[Tuple[int, int]],         # [(row, col), ...] negative 포인트
    model_name: str = "new_method_mlp",        # 모델 선택
    threshold: float | None = None,            # None이면 Otsu 자동
    device: str = "cuda",
    progress_callback: Callable | None = None,
) -> Dict[str, Any]:
    """
    Returns:
        {
            "mask": np.ndarray (H, W) bool,
            "score_map": np.ndarray (H, W) float32,
            "threshold": float,
            "train_info": dict  # MLP 학습 정보 (loss 등)
        }
    """
```

**내부에서 사용하는 파일들**:
- `models/new_method_mlp.py` — MLP projector 모델
- `models/osp_amf.py` — AMF/ACE/OSP 코어 알고리즘
- `models/config.py` — RuntimeConfig
- `webapp_core/model_runner.py` → 리팩토링하여 inference.py에 통합

### 2.2 Platform Backend 수정

**`backend/services/target_detection.py` 수정:**

```python
# 기존 classical detectors 유지
CLASSICAL_ALGORITHMS = ["SAM", "ACE", "RXD", "CEM", "MF"]

# MLP 모델 추가
MLP_ALGORITHMS = ["MLP_AMF", "MLP_ACE"]

def run_target_detection(raster_path, target_points_latlon, 
                         negative_points_latlon=None,  # 추가
                         algorithm="SAM", ...):
    if algorithm in MLP_ALGORITHMS:
        from repo.Target_detection.inference import detect
        result = detect(
            cube=raster_data,
            pos_points=pos_pixels,
            neg_points=neg_pixels,
            model_name="new_method_mlp" if algorithm == "MLP_AMF" else "new_method_mlp_ace",
        )
    else:
        # 기존 classical 로직
        ...
```

**`backend/api/schemas.py` 수정:**

```python
class TargetDetectionRequest(BaseModel):
    image_id: str
    bbox: List[float]
    target_points: List[TargetPoint]
    negative_points: Optional[List[TargetPoint]] = None  # 추가
    algorithm: str = "SAM"
    ...
```

**`backend/api/routes_target_detection.py` 수정:**
- `negative_points`를 `run_target_detection()`에 전달
- 알고리즘 목록에 MLP_AMF, MLP_ACE 추가

### 2.3 Frontend 수정

**`frontend/js/target-detection.js` 수정:**

#### Positive/Negative 클릭 구현

```javascript
// 상태 추가
negativePoints = [];
negativeMarkers = [];
clickMode = 'positive';  // 'positive' | 'negative'

// 클릭 핸들러 수정
handleMapClick(e) {
    if (this.clickMode === 'positive') {
        this.targetPoints.push({lat, lng});
        // 초록색 마커 + "+" 라벨
    } else {
        this.negativePoints.push({lat, lng});
        // 빨간색 마커 + "-" 라벨
    }
}
```

**UI 추가 요소:**
- Positive/Negative 모드 전환 토글 버튼 (또는 우클릭 = negative)
- Undo 버튼 (마지막 클릭 제거)
- 포인트 카운트 표시 (Positive: N, Negative: M)
- 알고리즘 드롭다운에 MLP_AMF, MLP_ACE 추가

#### Mask/Overlay 위에서 클릭 가능하도록

```javascript
// Leaflet ImageOverlay에 interactive 옵션 추가
const overlay = L.imageOverlay(url, bounds, {
    interactive: true,  // 클릭 이벤트 활성화
    zIndex: 400,
});

overlay.on('click', (e) => {
    // overlay 위 클릭도 target point로 등록
    this.handleMapClick(e);
});
```

### 2.4 수정 대상 파일 요약

| 파일 | 작업 | 우선순위 |
|------|------|----------|
| `repo/Target_detection/inference.py` | **신규 생성** — 모델 추론 API | P0 |
| `backend/services/target_detection.py` | MLP 모델 dispatch + neg_points 지원 | P0 |
| `backend/api/schemas.py` | `negative_points` 필드 추가 | P0 |
| `backend/api/routes_target_detection.py` | neg_points 전달 + 알고리즘 목록 확장 | P0 |
| `frontend/js/target-detection.js` | pos/neg 클릭, overlay 클릭, undo | P0 |
| `frontend/index.html` | UI 요소 추가 (토글 버튼 등) | P1 |
| `backend/requirements.txt` | torch 의존성 확인 | P1 |

---

## 3. 주요 기술 포인트

### 3.1 좌표계 변환

**repo 웹앱**: 클릭 좌표 → `canvasToOrig()` → 원본 이미지 pixel 좌표 (x, y)
**Platform**: 클릭 좌표 → Leaflet lat/lng → GeoTIFF affine transform → pixel 좌표 (row, col)

MLP 모델은 pixel 좌표 `(row, col)`을 받으므로, Platform에서 lat/lng → pixel 변환 후 전달.

### 3.2 데이터 포맷

**repo**: `cube` = `np.ndarray (H, W, C)` — 메모리에 로드된 hyperspectral cube
**Platform**: GeoTIFF 파일 → `rasterio`로 로드 → `(C, H, W)` → transpose 필요

### 3.3 Negative Points의 역할 (MLP 모델)

MLP 학습 시 negative points는:
- `L_det` loss에서 negative 샘플로 사용 (positive > negative 점수 강제)
- 배경 공분산 추정에서 negative 영역 포함/제외 결정
- Classical detector (SAM, ACE 등)는 negative points를 사용하지 않음 → 무시 처리

### 3.4 GPU 요구사항

- MLP projector 학습에 CUDA 권장 (CPU에서도 동작하지만 느림)
- 100 iterations × 작은 MLP → GPU 있으면 1-2초, CPU 5-10초
- SAM/SAM2 모델은 이번 통합 범위에서 제외

---

## 4. 검증 방법

1. `bash run.sh`로 Platform 실행
2. Sentinel-2 영상 검색 및 처리
3. Target Detection 패널 열기
4. 알고리즘에서 "MLP_AMF" 선택
5. 좌클릭으로 타겟 영역에 positive point 추가
6. 우클릭 (또는 negative 모드)으로 비타겟 영역에 negative point 추가
7. Detection 실행 → score map + mask 오버레이 확인
8. 오버레이 위에서 추가 클릭 → 새로운 포인트 등록 확인
9. Threshold colorbar로 조정 → mask 업데이트 확인
10. Classical 알고리즘 (SAM, ACE)과 결과 비교
