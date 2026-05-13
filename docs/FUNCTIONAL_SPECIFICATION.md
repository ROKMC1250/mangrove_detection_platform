# EarthScope - 기능명세서

## 문서 정보
| 항목 | 내용 |
|------|------|
| **문서명** | EarthScope 기능명세서 |
| **버전** | 2.0 |
| **작성일** | 2026-01-21 |
| **플랫폼 목적** | 위성 영상 기반 맹그로브 및 식생 분석 플랫폼 |

---

## 플랫폼 개요

EarthScope는 Google Earth Engine과 연동하여 Sentinel-2 (광학), Sentinel-1 (SAR) 위성 영상을 검색, 시각화, 분석할 수 있는 웹 기반 플랫폼입니다. 딥러닝 기반 맹그로브 분류, 분광지수 분석, 타겟 탐지, 시계열 변화 모니터링 기능을 제공합니다.

---

## 1. 지도 및 AOI 관리 기능

### 1.1 AOI (Area of Interest) 설정

| 기능명 | 설명 |
|--------|------|
| **사각형 그리기** | 지도에서 드래그하여 분석 영역(AOI)을 사각형으로 지정 |
| **좌표 이동** | 위도, 경도 좌표 입력으로 해당 위치로 지도 이동 (예: `26.761, 103.794`) |
| **SHP/KML 업로드** | Shapefile(.zip) 또는 KML 파일을 드래그앤드롭으로 업로드하여 AOI 설정 |
| **AOI 삭제** | Clear 버튼으로 현재 설정된 AOI 영역 초기화 |

**[스크린샷 1.1: AOI 설정 화면]**
- **캡처 범위**: 좌측 컨트롤 패널 상단부
- **포함 요소**:
  - 좌표 입력 필드 (`lat, lng` 형식)
  - "Go" 버튼
  - "Draw AOI" 버튼
  - 드래그앤드롭 영역 ("Drop SHP/KML here")
  - "Clear" 버튼
- **권장 상태**: AOI 미설정 상태

---

## 2. 위성 영상 검색 기능

### 2.1 Sentinel-2 (광학 영상) 검색

| 항목 | 상세 |
|------|------|
| **데이터셋** | COPERNICUS/S2_SR_HARMONIZED |
| **검색 파라미터** | 시작일, 종료일, 최대 구름량(%) |
| **반환 정보** | 영상 ID, 촬영일시, 구름량, AOI 커버리지 |
| **최대 결과** | 100개 |

**[스크린샷 2.1: Sentinel-2 검색 화면]**
- **캡처 범위**: 검색 탭 전체
- **포함 요소**:
  - 날짜 선택 (Start Date, End Date)
  - 구름량 슬라이더 (Max Cloud Cover: 0-100%)
  - 위성 선택 버튼 (S2 버튼 활성화 상태 - 파란색)
  - "Search Satellite Images" 버튼
- **권장 상태**: S2 버튼 활성화, 날짜 입력 완료

### 2.2 Sentinel-1 (SAR 영상) 검색

| 항목 | 상세 |
|------|------|
| **데이터셋** | COPERNICUS/S1_GRD |
| **밴드** | VV, VH |
| **검색 파라미터** | 시작일, 종료일 |
| **반환 정보** | 영상 ID, 촬영일시, 궤도정보, AOI 커버리지 |

**[스크린샷 2.2: Sentinel-1 검색 화면]**
- **캡처 범위**: S1 선택 시 패널 영역
- **포함 요소**:
  - S1 버튼 활성화 상태
  - VV/VH 밴드 드래그앤드롭 영역
  - RGB 슬롯 (R, G, B)
  - Min/Max dB 값 입력 (기본: -25 ~ 0)
- **권장 상태**: S1 선택, 밴드 설정 완료

### 2.3 검색 결과 표시

| 항목 | 상세 |
|------|------|
| **결과 위치** | 검색 탭 하단 인라인 표시 |
| **표시 정보** | 영상 ID, 촬영일시, 구름량 |
| **상호작용** | 클릭하여 지도에 미리보기 표시 |

**[스크린샷 2.3: 검색 결과 목록]**
- **캡처 범위**: 검색 결과 영역
- **포함 요소**:
  - 결과 개수 표시
  - 영상 카드 목록 (날짜, 구름량 표시)
  - 선택된 영상 하이라이트 표시
- **권장 상태**: 여러 개의 검색 결과가 표시된 상태

---

## 3. 밴드 시각화 기능

### 3.1 Sentinel-2 밴드 조합 (드래그앤드롭)

플랫폼은 직관적인 드래그앤드롭 인터페이스로 밴드를 RGB 채널에 할당합니다.

| 사용 가능 밴드 | 파장대 | 용도 |
|----------------|--------|------|
| **B2** | Blue (490nm) | 가시광선 |
| **B3** | Green (560nm) | 가시광선 |
| **B4** | Red (665nm) | 가시광선 |
| **B5** | Red Edge 1 (705nm) | 식생 경계 |
| **B6** | Red Edge 2 (740nm) | 식생 경계 |
| **B7** | Red Edge 3 (783nm) | 식생 경계 |
| **B8** | NIR (842nm) | 근적외선 |
| **B8A** | NIR Narrow (865nm) | 근적외선 |
| **B11** | SWIR1 (1610nm) | 단파적외선 |
| **B12** | SWIR2 (2190nm) | 단파적외선 |

**기본 프리셋**
| 프리셋 | RGB 밴드 조합 | 용도 |
|--------|---------------|------|
| **True Color** | B4, B3, B2 | 자연색 |
| **False Color** | B8, B4, B3 | 식생 강조 |
| **Agriculture** | B11, B8, B2 | 농업 분석 |
| **SWIR** | B12, B8A, B4 | 수분/지질 분석 |

**[스크린샷 3.1: S2 밴드 드래그앤드롭 UI]**
- **캡처 범위**: 밴드 선택 영역
- **포함 요소**:
  - 밴드 풀 (B2~B12 칩들)
  - RGB 슬롯 (R: 빨간색, G: 초록색, B: 파란색 배경)
  - 현재 할당된 밴드 표시
  - Min/Max 값 입력 (기본: 0 ~ 3000)
  - "Apply" 버튼
- **권장 상태**: 기본 True Color (B4-B3-B2) 설정

### 3.2 Sentinel-1 SAR 밴드 조합

| 사용 가능 밴드 | 설명 |
|----------------|------|
| **VV** | 수직-수직 편파 |
| **VH** | 수직-수평 편파 |

| 프리셋 | 밴드 조합 | 용도 |
|--------|----------|------|
| **VV only** | VV, VV, VV | VV 편파 분석 |
| **VH only** | VH, VH, VH | VH 편파 분석 |
| **VV/VH/VV** | VV, VH, VV | 복합 분석 |

**[스크린샷 3.2: S1 SAR 밴드 UI]**
- **캡처 범위**: S1 선택 시 밴드 영역
- **포함 요소**:
  - VV, VH 밴드 칩
  - RGB 슬롯
  - Min/Max dB 값 입력 (기본: -25 ~ 0)
  - "Apply" 버튼
- **권장 상태**: VV-VH-VV 설정

---

## 4. 영상 처리 및 분석 기능 (Process Image)

### 4.1 처리 개요

선택한 위성 영상을 다운로드하고 다중 분석 결과를 생성합니다.

| 분석 항목 | 설명 | 출력 |
|-----------|------|------|
| **Cloud Mask** | S2 Cloud Probability 기반 구름 마스크 (>30%) | 오렌지색 오버레이 |
| **Mangrove Segmentation** | 딥러닝 기반 맹그로브 분류 | 분류 마스크 |
| **NDVI** | 정규식생지수 | RdYlGn 컬러맵 |
| **NDMI** | 정규수분지수 | RdYlGn 컬러맵 |
| **MVI** | 맹그로브식생지수 | viridis 컬러맵 |
| **AlphaEarth Embedding** | Google 위성 임베딩 PCA 시각화 | RGB 오버레이 |

### 4.2 분광지수 계산식

#### NDVI (Normalized Difference Vegetation Index)
```
NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
범위: -1 ~ 1
컬러맵: RdYlGn (빨강-노랑-초록)
```

#### NDMI (Normalized Difference Moisture Index)
```
NDMI = (SWIR2 - Green) / (SWIR2 + Green) = (B12 - B3) / (B12 + B3)
범위: -1 ~ 1
컬러맵: RdYlGn
```

#### MVI (Mangrove Vegetation Index)
```
MVI = (NIR - Green) / (SWIR1 - Green) = (B8 - B3) / (B11 - B3)
범위: 자동 계산
컬러맵: viridis
```

**[스크린샷 4.1: 처리 진행 화면]**
- **캡처 범위**: 로딩 오버레이 전체
- **포함 요소**:
  - 로딩 스피너
  - 현재 단계 텍스트 (예: "Running segmentation model")
  - 진행률 바
  - "Cancel" 버튼
- **권장 상태**: 처리 중간 상태 (50% 정도)

### 4.3 맹그로브 분류 모델

| 항목 | 상세 |
|------|------|
| **모델 프레임워크** | PyTorch + Segmentation Models PyTorch (SMP) |
| **모델 아키텍처** | Segformer (기본) |
| **인코더** | MIT-B2 |
| **입력 채널** | 13채널 (Sentinel-2 전체 밴드) |
| **추론 방식** | Sliding Window (패치 기반) |
| **패치 크기** | 256 (설정 가능) |
| **오버랩** | 0.5 (50%, 설정 가능) |
| **출력** | 이진 분류 마스크 (맹그로브/비맹그로브) |

모델 설정은 `backend/model_config.yaml` 파일에서 변경할 수 있습니다:
```yaml
model_dir: "/path/to/your/model"
checkpoint: "last.pt"
gpus: "0"
patch_size: 256
overlap: 0.5
use_tta: false
```

**[스크린샷 4.2: 분석 결과 탭]**
- **캡처 범위**: Analysis Results 탭 전체
- **포함 요소**:
  - 분석 결과 카드들 (썸네일 + 이름)
  - Cloud Mask, Mangrove Segmentation, NDVI, NDMI, MVI, AlphaEarth
  - 각 카드 클릭 시 지도에 오버레이 표시
- **권장 상태**: 처리 완료 후 결과들이 표시된 상태

**[스크린샷 4.3: NDVI 오버레이 및 컬러바]**
- **캡처 범위**: 지도 + 분석 결과 패널
- **포함 요소**:
  - 지도 위 NDVI 컬러맵 오버레이
  - 컬러바 범례 (-1 ~ 1)
  - NDVI 카드 활성화 상태
- **권장 상태**: NDVI 오버레이 활성화

**[스크린샷 4.4: 맹그로브 분류 결과]**
- **캡처 범위**: 지도 + 분석 결과 패널
- **포함 요소**:
  - 지도 위 맹그로브 분류 오버레이 (빨간색)
  - Mangrove Segmentation 카드 활성화 상태
- **권장 상태**: 맹그로브 오버레이 활성화

---

## 5. 시계열 변화 모니터링 기능 (Change Monitoring)

### 5.1 이미지 검색

| 기능 | 설명 |
|------|------|
| **날짜 범위** | 분석할 기간의 시작일과 종료일 선택 |
| **구름량 필터** | 최대 구름량(%) 설정 |
| **AOI 커버리지** | 최소 AOI 커버리지(%) 설정 |
| **분석 방법** | NDVI, NDMI, MVI 중 선택 |

**[스크린샷 5.1: Change Monitoring 탭]**
- **캡처 범위**: Change Monitoring 탭 전체
- **포함 요소**:
  - Search Period (Start Date, End Date)
  - Max Cloud Cover 슬라이더
  - Min AOI Coverage 슬라이더
  - Spectral Analysis Method 드롭다운 (NDVI/NDMI/MVI)
  - "Search Available Images" 버튼
- **권장 상태**: 초기 상태

### 5.2 날짜 선택 및 분석

| 기능 | 설명 |
|------|------|
| **캘린더 뷰** | 검색된 이미지를 날짜별로 표시 |
| **다중 선택** | 체크박스로 여러 날짜 선택 |
| **전체 선택** | "Select All" 버튼으로 모든 날짜 선택 |
| **미리보기** | 각 날짜의 영상을 지도에서 미리보기 |

**[스크린샷 5.2: 캘린더 날짜 선택 UI]**
- **캡처 범위**: 캘린더 그리드 영역
- **포함 요소**:
  - 날짜별 이미지 리스트
  - 체크박스 (선택 상태)
  - 구름량 표시
  - 선택된 이미지 개수 표시
  - "Select All" 버튼
  - "Run Time Series Analysis" 버튼
- **권장 상태**: 여러 날짜가 선택된 상태

### 5.3 분석 결과

| 출력 항목 | 설명 |
|-----------|------|
| **시계열 차트** | Chart.js 기반 시간에 따른 지수 변화 그래프 |
| **분석 이미지 목록** | 날짜별 분석된 이미지 리스트 |
| **통계 요약** | 총 변화량, 최대/최소 면적, 평균 지수 |
| **CSV 다운로드** | 분석 결과 CSV 파일 다운로드 |

**[스크린샷 5.3: 시계열 분석 결과]**
- **캡처 범위**: 분석 결과 영역
- **포함 요소**:
  - 시계열 차트 (X축: 날짜, Y축: 지수 평균값)
  - "Download CSV" 버튼
  - Analyzed Images 목록
  - 각 이미지의 👁️ 보기 버튼
- **권장 상태**: 분석 완료 후 차트가 표시된 상태

---

## 6. 타겟 탐지 기능 (Target Detection)

### 6.1 지원 알고리즘

| 알고리즘 | 설명 |
|----------|------|
| **SAM** | Spectral Angle Mapper - 스펙트럼 각도 기반 유사도 |
| **ACE** | Adaptive Cosine Estimator - 적응형 코사인 추정 |
| **MF** | Matched Filter - 정합 필터 |
| **CEM** | Constrained Energy Minimization - 제약 에너지 최소화 |

### 6.2 타겟 탐지 워크플로우

```
1. Analysis Results 탭에서 Target Detection 클릭
   ↓
2. 알고리즘 선택 (SAM/ACE/MF/CEM 드롭다운)
   ↓
3. 📍 Select 버튼 클릭 → 지도에서 타겟 포인트 클릭 (1개 이상)
   ↓
4. 🔧 Bands 버튼으로 사용할 밴드 선택 (선택사항)
   ↓
5. Run 버튼 클릭 → 탐지 실행
   ↓
6. 스코어 맵 + 컬러바 + 임계값 슬라이더 표시
   ↓
7. 차트 박스 표시 (Target vs Background, Score Distribution)
   ↓
8. Apply → 이진 마스크 생성 / Cancel → 스코어 맵으로 복귀
   ↓
9. 🔄 Try another model → 설정 화면으로 복귀
```

### 6.3 밴드 선택

타겟 탐지 시 사용할 밴드를 선택할 수 있습니다:

| 밴드 | 이름 | 기본 선택 |
|------|------|----------|
| B2 | Blue | ✓ |
| B3 | Green | ✓ |
| B4 | Red | ✓ |
| B5 | RE1 | ✓ |
| B6 | RE2 | ✓ |
| B7 | RE3 | ✓ |
| B8 | NIR | ✓ |
| B8A | NIRn | ✓ |
| B11 | SWIR1 | ✓ |
| B12 | SWIR2 | ✓ |

**[스크린샷 6.1: 타겟 탐지 설정 UI]**
- **캡처 범위**: Target Detection 아이템 확장 상태
- **포함 요소**:
  - 알고리즘 선택 드롭다운 (SAM, ACE, MF, CEM)
  - 📍 Select 버튼 + 선택된 포인트 개수
  - 🔧 Bands 버튼
  - Run 버튼 (비활성화/활성화 상태)
- **권장 상태**: 알고리즘 선택 완료, 포인트 1개 이상 선택

**[스크린샷 6.2: 밴드 선택 패널]**
- **캡처 범위**: 밴드 선택 패널
- **포함 요소**:
  - 10개 밴드 체크박스 그리드
  - All / None 버튼
- **권장 상태**: 일부 밴드만 선택된 상태

**[스크린샷 6.3: 타겟 탐지 스코어 맵]**
- **캡처 범위**: 지도 + 스코어 UI
- **포함 요소**:
  - 지도 위 탐지 스코어 맵 (jet 컬러맵)
  - 컬러바 (min ~ max)
  - 양쪽 핸들이 있는 임계값 슬라이더
  - Min/Max 값 입력 필드
  - Apply / Cancel 버튼
- **권장 상태**: 스코어 맵이 표시된 지도

**[스크린샷 6.4: 분석 차트]**
- **캡처 범위**: 차트 박스들
- **포함 요소**:
  - Target vs Background Spectrum 차트
  - Score Distribution 히스토그램
  - 임계값 라인 표시
- **권장 상태**: 탐지 완료 후 차트 표시

**[스크린샷 6.5: 이진 마스크 결과]**
- **캡처 범위**: 지도 + 마스크 UI
- **포함 요소**:
  - 지도 위 탐지 마스크 (빨간색)
  - 탐지 픽셀 수, 비율 표시
  - 임계값 조정 슬라이더
  - Apply / Cancel / 🔄 Try another model 버튼
- **권장 상태**: 마스크가 표시된 지도

---

## 7. 임계값 적용 기능

### 7.1 범위 기반 임계값

분광지수(NDVI, NDMI, MVI) 및 타겟 탐지 결과에 대해 임계값 범위를 적용하여 이진 마스크를 생성할 수 있습니다.

| 기능 | 설명 |
|------|------|
| **범위 선택** | Min/Max 핸들을 드래그하여 범위 설정 |
| **직접 입력** | 숫자 입력 필드에 직접 값 입력 |
| **마스크 생성** | 범위 내 픽셀만 표시되는 이진 마스크 |
| **적용 대상** | NDVI, NDMI, MVI, Custom Index, 타겟 탐지 스코어 |

**[스크린샷 7.1: 임계값 컨트롤러]**
- **캡처 범위**: 컬러바 + 임계값 UI
- **포함 요소**:
  - 컬러 그라데이션 트랙
  - 선택 영역 표시 (반투명 하이라이트)
  - Min/Max 핸들 (드래그 가능)
  - Min/Max 값 입력 필드
  - Apply 버튼
- **권장 상태**: 범위가 조정된 상태

---

## 8. 사용자 정의 시각화 기능
ㄴ
### 8.1 RGB Composite

| 기능 | 설명 |
|------|------|
| **밴드 선택** | 드래그앤드롭으로 R, G, B 채널에 밴드 할당 |
| **Min/Max 조정** | 영상 밝기/대비 조정 |
| **실시간 적용** | Apply 버튼으로 즉시 지도에 반영 |

### 8.2 Custom Index

| 기능 | 설명 |
|------|------|
| **밴드 선택** | Band A, Band B 선택 |
| **계산식** | (A - B) / (A + B) 정규화 지수 |
| **컬러맵** | viridis, RdYlGn 등 선택 가능 |

---

## 9. 이미지 다운로드 기능

### 9.1 다운로드 옵션

| 위성 | 형식 | 설명 |
|------|------|------|
| **Sentinel-2** | Raw GeoTIFF | 13밴드 원본 데이터, 10m 리샘플링 |
| **Sentinel-2** | Visualization PNG | 현재 시각화 설정 기반 PNG |
| **Sentinel-1** | Raw GeoTIFF | VV, VH 밴드 Float32 |
| **Sentinel-1** | Visualization PNG | 현재 시각화 설정 기반 PNG |

---

## 10. 픽셀 값 조회 기능

| 기능 | 설명 |
|------|------|
| **지도 클릭** | 클릭한 위치의 분석 값 조회 |
| **표시 정보** | 좌표 (lat, lng), 해당 레이어 값 |
| **지원 레이어** | NDVI, NDMI, MVI, 타겟 탐지 스코어 |

---

## 11. 진행률 표시 기능

### 11.1 처리 단계

| 단계 | 설명 |
|------|------|
| **Initialization** | AOI 설정, 이미지 참조 |
| **Download** | GEE에서 이미지 다운로드 |
| **Model Inference** | 구름 마스크 + 딥러닝 모델 추론 |
| **Index Calculation** | NDVI, NDMI, MVI 계산 |
| **Visualization** | 시각화 생성 및 AOI 정렬 |
| **Finalization** | 결과 정리 |

**[스크린샷 11.1: 진행률 오버레이]**
- **캡처 범위**: 로딩 오버레이 전체
- **포함 요소**:
  - 로딩 스피너
  - 현재 단계 텍스트
  - 진행률 바 (%)
  - Cancel 버튼
- **권장 상태**: 진행률 50% 정도

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vanilla JS)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Leaflet  │ │ Chart.js │ │ GeoRaster│ │ Platform Modules │ │
│  │   Map    │ │  Charts  │ │  Layer   │ │                  │ │
│  └──────────┘ └──────────┘ └──────────┘ │ - map-core.js    │ │
│                                          │ - map-drawing.js │ │
│  ┌────────────────────────────────────┐ │ - map-layers.js  │ │
│  │ Controller Modules                  │ │ - image-search   │ │
│  │ - image-processor.js               │ │ - analysis-ctrl  │ │
│  │ - change-monitoring.js             │ │ - threshold-ctrl │ │
│  │ - target-detection.js              │ │ - target-detect  │ │
│  └────────────────────────────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend (FastAPI + Python)                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ routes_search│ │routes_process│ │routes_target_detect  │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │routes_downld │ │routes_analyss│ │    services/         │ │
│  └──────────────┘ └──────────────┘ │ - earth_engine.py    │ │
│                                     │ - model_inference.py │ │
│  ┌──────────────────────────────┐  │ - target_detection.py│ │
│  │ model_config.yaml            │  │ - visualization.py   │ │
│  │ (세그멘테이션 모델 설정)      │  │ - spectral_analysis  │ │
│  └──────────────────────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Services                          │
│  ┌──────────────────┐  ┌───────────────────────────────────┐│
│  │ Google Earth     │  │ Deep Learning Model               ││
│  │ Engine API       │  │ (PyTorch + SMP)                   ││
│  │ - Sentinel-2     │  │ - Segformer/UNet++/MAnet/PAN/FPN  ││
│  │ - Sentinel-1     │  │ - Mangrove Segmentation           ││
│  │ - S2 Cloud Prob  │  └───────────────────────────────────┘│
│  │ - AlphaEarth     │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 스크린샷 체크리스트

### 필요 스크린샷 목록

| 번호 | 이미지 ID | 파일명 제안 | 캡처 범위 | 필요 상태/조건 |
|------|-----------|-------------|-----------|----------------|
| 1 | 1.1 | aoi_setup.png | 좌측 패널 상단 | AOI 미설정 상태 |
| 2 | 2.1 | s2_search.png | 검색 탭 전체 | S2 선택, 날짜 입력 완료 |
| 3 | 2.2 | s1_search.png | S1 패널 영역 | S1 선택, 밴드 설정 |
| 4 | 2.3 | search_results.png | 검색 결과 영역 | 여러 결과 표시 |
| 5 | 3.1 | s2_band_dnd.png | 밴드 선택 영역 | 드래그앤드롭 UI |
| 6 | 3.2 | s1_band_ui.png | S1 밴드 영역 | VV-VH 설정 |
| 7 | 4.1 | processing_loading.png | 로딩 오버레이 | 처리 중 50% |
| 8 | 4.2 | analysis_results.png | Analysis Results 탭 | 결과 표시 상태 |
| 9 | 4.3 | ndvi_overlay.png | 지도 + 패널 | NDVI 활성화 |
| 10 | 4.4 | mangrove_result.png | 지도 + 패널 | 맹그로브 활성화 |
| 11 | 5.1 | change_monitoring_tab.png | Change Monitoring 탭 | 초기 상태 |
| 12 | 5.2 | calendar_selection.png | 캘린더 영역 | 날짜 선택 상태 |
| 13 | 5.3 | timeseries_chart.png | 분석 결과 영역 | 차트 표시 |
| 14 | 6.1 | td_setup.png | Target Detection UI | 설정 모드 |
| 15 | 6.2 | td_bands.png | 밴드 선택 패널 | 일부 선택 |
| 16 | 6.3 | td_score_map.png | 지도 + 스코어 UI | 탐지 완료 |
| 17 | 6.4 | td_charts.png | 차트 박스들 | 2개 차트 표시 |
| 18 | 6.5 | td_mask.png | 지도 + 마스크 UI | 임계값 적용 후 |
| 19 | 7.1 | threshold_controller.png | 임계값 컨트롤러 | 범위 조정 상태 |
| 20 | 11.1 | progress_overlay.png | 로딩 오버레이 | 진행률 50% |

### 스크린샷 캡처 권장사항

1. **해상도**: 1920x1080 이상
2. **포맷**: PNG (고품질)
3. **강조 표시**: 중요 UI 요소에 빨간색 박스 또는 화살표 추가 권장
4. **데이터**: 실제 위성 영상 데이터로 캡처 (더미 데이터 X)
5. **일관성**: 동일한 AOI 영역 사용하여 일관된 스크린샷

### 이미지 폴더 구조 제안

```
docs/
├── FUNCTIONAL_SPECIFICATION.md
└── images/
    ├── 01_aoi/
    │   └── aoi_setup.png
    ├── 02_search/
    │   ├── s2_search.png
    │   ├── s1_search.png
    │   └── search_results.png
    ├── 03_visualization/
    │   ├── s2_band_dnd.png
    │   └── s1_band_ui.png
    ├── 04_processing/
    │   ├── processing_loading.png
    │   ├── analysis_results.png
    │   ├── ndvi_overlay.png
    │   └── mangrove_result.png
    ├── 05_change_monitoring/
    │   ├── change_monitoring_tab.png
    │   ├── calendar_selection.png
    │   └── timeseries_chart.png
    ├── 06_target_detection/
    │   ├── td_setup.png
    │   ├── td_bands.png
    │   ├── td_score_map.png
    │   ├── td_charts.png
    │   └── td_mask.png
    ├── 07_threshold/
    │   └── threshold_controller.png
    └── 11_progress/
        └── progress_overlay.png
```

---

## API 엔드포인트 요약

| 엔드포인트 | 메서드 | 기능 |
|------------|--------|------|
| `/api/search-images` | POST | Sentinel-2 이미지 검색 |
| `/api/search-s1-images` | POST | Sentinel-1 이미지 검색 |
| `/api/get-gee-tile` | POST | GEE 타일 URL 획득 |
| `/api/get-s1-tile` | POST | S1 타일 URL 획득 |
| `/api/get-s2-tile-custom` | POST | S2 커스텀 시각화 타일 |
| `/api/process-image` | POST | 이미지 처리 (다중 분석) |
| `/api/change-monitoring` | POST | 시계열 변화 분석 |
| `/api/process-spectral-image` | POST | 분광 이미지 생성 |
| `/api/download-s1-image` | POST | S1 이미지 다운로드 |
| `/api/download-s2-image` | POST | S2 이미지 다운로드 |
| `/api/custom-visualization` | POST | 사용자 정의 시각화 |
| `/api/get-pixel-value` | POST | 픽셀 값 조회 |
| `/api/apply-threshold-range` | POST | 임계값 범위 적용 |
| `/api/target-detection/algorithms` | GET | 타겟 탐지 알고리즘 목록 |
| `/api/target-detection/run` | POST | 타겟 탐지 실행 |
| `/api/target-detection/apply-threshold` | POST | 탐지 임계값 적용 |
| `/api/target-detection/get-spectrum` | POST | 픽셀 스펙트럼 조회 |
| `/api/progress/{job_id}` | GET | 작업 진행률 조회 |

---

## 환경 설정

### 필수 환경변수

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `GOOGLE_APPLICATION_CREDENTIALS` | GEE 서비스 계정 키 경로 | `/path/to/key.json` |

### model_config.yaml 설정

```yaml
# 모델 디렉토리 경로
model_dir: "/path/to/your/model/logs"

# 체크포인트 파일명
checkpoint: "last.pt"

# GPU 설정 (비워두면 CPU)
gpus: "0"

# 예측 설정
patch_size: 256
overlap: 0.5
use_tta: false

# 기본 모델 파라미터
default_model:
  name: "Segformer"
  encoder_name: "mit_b2"
  in_channels: 13
  classes: 1
```

---

*문서 끝*
