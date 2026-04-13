/**
 * Target Detection Controller
 *
 * Flow:
 * 1. Click Target Detection → Show: [Algorithm] [📍 Select] [🔧 Bands] [Run]
 * 2. Select target points on map (shared across all detections)
 * 3. Pick algorithm, click Run → result added to results list
 * 4. Each result has score map + threshold controls → apply for binary mask
 * 5. Multiple detections can be stacked and toggled independently
 */

class TargetDetectionController {
    constructor(platformController) {
        this.platform = platformController;

        // Shared target point workspace
        this.targetSelectMode = false;
        this.clickMode = 'positive';  // 'positive' | 'negative'
        this.targetPoints = [];       // positive points
        this.negativePoints = [];     // negative points
        this.targetMarkers = [];
        this.clickHistory = [];       // ['pos', 'neg', ...] for undo

        // Results storage (multiple stacked)
        this.results = [];  // Array of detection result objects
        this.overlayLayers = {};  // resultId → true (managed via mapManager)

        // Live detection state (auto-run)
        this._currentLiveResult = null;   // Current live detection result entry
        this._currentLiveOverlayId = null;
        this._detectPending = false;
        this._detectAbort = false;

        this.selectedAlgorithm = 'SAM';
        this.selectedBands = null;

        this.algorithms = [
            { id: 'sam', name: 'SAM' },
            { id: 'ace', name: 'ACE' },
            { id: 'mf', name: 'MF' },
            { id: 'cem', name: 'CEM' },
            { id: 'mlp_amf', name: 'MLP_AMF' },
            { id: 'mlp_ace', name: 'MLP_ACE' }
        ];

        this.allBands = [
            { idx: 0, name: 'B2', label: 'Blue' },
            { idx: 1, name: 'B3', label: 'Green' },
            { idx: 2, name: 'B4', label: 'Red' },
            { idx: 3, name: 'B5', label: 'RE1' },
            { idx: 4, name: 'B6', label: 'RE2' },
            { idx: 5, name: 'B7', label: 'RE3' },
            { idx: 6, name: 'B8', label: 'NIR' },
            { idx: 7, name: 'B8A', label: 'NIRn' },
            { idx: 8, name: 'B11', label: 'SWIR1' },
            { idx: 9, name: 'B12', label: 'SWIR2' }
        ];

        this.secondaryTargetMarkers = [];  // mirrors targetMarkers on secondary map (compare mode)

        this.handleMapClick = this.handleMapClick.bind(this);
        this._handleContextMenu = this._handleContextMenu.bind(this);

        // Sync markers when compare mode is toggled
        window.addEventListener('comparemap:activated', () => this._syncMarkersToSecondary());
        window.addEventListener('comparemap:deactivated', () => { this.secondaryTargetMarkers = []; });
    }

    isLocalMode() {
        const li = this.platform.localImage;
        return !!li?.localMapActive || !!li?.isUploadedImageActive;
    }

    isUploadedMode() {
        const li = this.platform.localImage;
        return li?.currentMode === 'load-image' && !!li?.isUploadedImageActive;
    }

    getMap() {
        if (this.isUploadedMode()) return window.mapManager?.map;
        if (this.isLocalMode()) return this.platform.localImage?.localMap;
        return window.mapManager?.map;
    }

    getImageInfo() {
        if (this.isUploadedMode()) {
            const li = this.platform.localImage;
            return {
                local: true,
                uploaded: true,
                upload_id: li.uploadId,
            };
        }
        if (this.isLocalMode()) {
            const li = this.platform.localImage;
            if (!li?.currentImageDir || !li?.currentAlgoDir) return null;
            return { local: true, image_dir: li.currentImageDir, algorithm_dir: li.currentAlgoDir };
        }
        const imageId = this.platform.selectedImageId;
        if (!imageId) return null;
        return {
            id: imageId,
            bbox: window.mapManager?.getCurrentBounds(),
            geometry: window.mapManager?.getCurrentGeoJSON()?.geometry
        };
    }

    // Called when clicking Target Detection item
    handleItemClick() {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification(this.isLocalMode() ? 'Load a local image first' : 'Process an image first', 'warning');
            return;
        }

        const item = document.querySelector('.analysis-item.target-detection-option');
        const hasUI = item?.querySelector('.td-ui');

        if (hasUI) {
            this.hideUI();
        } else {
            this.showSetupUI();
        }
    }

    // ========== SETUP UI ==========
    showSetupUI() {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;

        item.querySelector('.td-ui')?.remove();

        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const bandsForUI = this.isLocalMode()
            ? (this.platform.localImage?.availableBands || []).map((b, i) => ({ idx: i, name: b.replace('after_', '').replace('.tif', ''), label: b.replace('after_', '').replace('.tif', '') }))
            : this.allBands;

        const ui = document.createElement('div');
        ui.className = 'td-ui td-setup';
        ui.innerHTML = `
            <div class="td-row">
                <select id="td-algo">
                    ${this.algorithms.map(a => `<option value="${a.id}">${a.name}</option>`).join('')}
                </select>
                <button id="td-pick" class="td-btn ${this.targetSelectMode ? 'active' : ''}" title="Left-click: positive, Right-click: negative">📍 <span id="td-count">${this.targetPoints.length}</span></button>
                <button id="td-mode-toggle" class="td-btn td-mode-positive" title="Current: Positive (click to switch)">+</button>
                <button id="td-undo" class="td-btn" title="Undo last point" ${this.clickHistory.length === 0 ? 'disabled' : ''}>↩</button>
                <button id="td-clear" class="td-btn" title="Clear all points" ${this.clickHistory.length === 0 ? 'disabled' : ''}>🗑</button>
                <button id="td-bands-toggle" class="td-btn">🔧 Bands</button>
                <button id="td-save" class="td-btn save" ${!this._currentLiveResult ? 'disabled' : ''} title="Save current detection result">💾 Save</button>
            </div>
            <div class="td-point-info" style="font-size:11px; color:#888; padding:2px 4px;">
                <span id="td-pos-count" style="color:#22cc44;">+${this.targetPoints.length}</span>
                <span id="td-neg-count" style="color:#ee3344; margin-left:6px;">-${this.negativePoints.length}</span>
                <span id="td-live-status" style="margin-left:8px; color:#ff9800;"></span>
            </div>
            <div id="td-bands-panel" class="td-bands-panel" style="display:none;">
                <div class="td-bands-grid">
                    ${bandsForUI.map(b => `
                        <label class="td-band-item">
                            <input type="checkbox" value="${b.idx}" checked>
                            <span>${b.name}</span>
                        </label>
                    `).join('')}
                </div>
                <div class="td-bands-actions">
                    <button id="td-bands-all" class="td-btn-sm">All</button>
                    <button id="td-bands-none" class="td-btn-sm">None</button>
                </div>
            </div>
            <div class="td-results-list"></div>
        `;

        item.appendChild(ui);
        item.classList.add('expanded');

        // Re-render existing results
        this.results.forEach(r => this._renderResultItem(r));

        // Initialize algorithm from dropdown
        const algoSelect = document.getElementById('td-algo');
        this.selectedAlgorithm = algoSelect.value.toUpperCase();

        // Events
        algoSelect.onchange = (e) => {
            this.selectedAlgorithm = e.target.value.toUpperCase();
            // Auto-run detection with new algorithm if points exist
            this._autoDetect();
        };

        document.getElementById('td-pick').onclick = (e) => {
            e.stopPropagation();
            this.toggleTargetSelection();
        };

        document.getElementById('td-mode-toggle').onclick = (e) => {
            e.stopPropagation();
            this.clickMode = this.clickMode === 'positive' ? 'negative' : 'positive';
            const btn = document.getElementById('td-mode-toggle');
            if (this.clickMode === 'positive') {
                btn.textContent = '+';
                btn.className = 'td-btn td-mode-positive';
                btn.title = 'Current: Positive (click to switch)';
            } else {
                btn.textContent = '−';
                btn.className = 'td-btn td-mode-negative';
                btn.title = 'Current: Negative (click to switch)';
            }
        };

        document.getElementById('td-undo').onclick = (e) => {
            e.stopPropagation();
            this.undoLastPoint();
        };

        document.getElementById('td-clear').onclick = (e) => {
            e.stopPropagation();
            this.clearAllPoints();
        };

        document.getElementById('td-bands-toggle').onclick = (e) => {
            e.stopPropagation();
            const panel = document.getElementById('td-bands-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        };

        document.getElementById('td-bands-all').onclick = (e) => {
            e.stopPropagation();
            document.querySelectorAll('#td-bands-panel input[type="checkbox"]').forEach(cb => cb.checked = true);
        };

        document.getElementById('td-bands-none').onclick = (e) => {
            e.stopPropagation();
            document.querySelectorAll('#td-bands-panel input[type="checkbox"]').forEach(cb => cb.checked = false);
        };

        document.getElementById('td-save').onclick = (e) => {
            e.stopPropagation();
            this.saveCurrentResult();
        };
    }

    hideUI() {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;
        item.querySelector('.td-ui')?.remove();
        item.classList.remove('expanded');
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    getSelectedBands() {
        const checkboxes = document.querySelectorAll('#td-bands-panel input[type="checkbox"]:checked');
        if (checkboxes.length === 0 || checkboxes.length === this.allBands.length) {
            return null;
        }
        return Array.from(checkboxes).map(cb => parseInt(cb.value));
    }

    // ========== TARGET POINT SELECTION ==========
    toggleTargetSelection() {
        if (this.targetSelectMode) {
            this.stopTargetSelection();
        } else {
            this.startTargetSelection();
        }
    }

    startTargetSelection() {
        // Stop SAM2 selection if active (prevent both listening at once)
        const sam2 = this.platform.sam2Controller;
        if (sam2?.selectMode) {
            sam2.stopSelection();
        }

        this.targetSelectMode = true;
        document.getElementById('td-pick')?.classList.add('active');
        const map = this.getMap();
        if (map) {
            map.getContainer().classList.add('prompt-cursor');
            map.on('click', this.handleMapClick);
            map.getContainer().addEventListener('contextmenu', this._handleContextMenu);

            // Attach click/contextmenu handlers to all existing interactive overlays
            // so clicks on analysis layers pass through to target detection
            this._attachOverlayClickHandlers(map);
        }
        this.platform.showNotification('Left-click: positive target, Right-click: negative', 'info');
    }

    stopTargetSelection() {
        this.targetSelectMode = false;
        document.getElementById('td-pick')?.classList.remove('active');
        const map = this.getMap();
        if (map) {
            map.getContainer().classList.remove('prompt-cursor');
            map.off('click', this.handleMapClick);
            map.getContainer().removeEventListener('contextmenu', this._handleContextMenu);

            // Clean up overlay click handlers
            this._detachOverlayClickHandlers();
        }
    }

    /**
     * Attach click handlers to all interactive overlay layers on the map
     * so target detection clicks work through analysis overlays
     */
    _attachOverlayClickHandlers(map) {
        this._overlayClickHandlers = [];

        const attachToLayer = (layer) => {
            if (!layer || !layer.on) return;
            const clickHandler = (e) => {
                if (this.targetSelectMode) {
                    L.DomEvent.stopPropagation(e);
                    this._addPoint(e.latlng, this.clickMode);
                }
            };
            const ctxHandler = (e) => {
                if (this.targetSelectMode) {
                    L.DomEvent.stopPropagation(e);
                    L.DomEvent.preventDefault(e);
                    this._addPoint(e.latlng, 'negative');
                }
            };
            layer.on('click', clickHandler);
            layer.on('contextmenu', ctxHandler);
            this._overlayClickHandlers.push({ layer, clickHandler, ctxHandler });
        };

        // Attach to all analysis layers
        const analysisLayers = window.mapManager?.analysisLayers || {};
        Object.values(analysisLayers).forEach(attachToLayer);

        // Attach to all tile layers (preview images)
        const tileLayers = window.mapManager?.tileLayers || {};
        Object.values(tileLayers).forEach(attachToLayer);

        // Attach to all image layers
        const imageLayers = window.mapManager?.imageLayers || {};
        Object.values(imageLayers).forEach(attachToLayer);
    }

    _detachOverlayClickHandlers() {
        if (!this._overlayClickHandlers) return;
        this._overlayClickHandlers.forEach(({ layer, clickHandler, ctxHandler }) => {
            try {
                layer.off('click', clickHandler);
                layer.off('contextmenu', ctxHandler);
            } catch (e) {}
        });
        this._overlayClickHandlers = [];
    }

    _handleContextMenu(e) {
        if (!this.targetSelectMode) return;
        e.preventDefault();
        e.stopPropagation();
        // Convert DOM event to latlng
        const map = this.getMap();
        if (!map) return;
        const latlng = map.containerPointToLatLng(L.point(
            e.clientX - map.getContainer().getBoundingClientRect().left,
            e.clientY - map.getContainer().getBoundingClientRect().top
        ));
        this._addPoint(latlng, 'negative');
    }

    handleMapClick(e) {
        if (!this.targetSelectMode) return;
        this._addPoint(e.latlng, this.clickMode);
    }

    _addPoint(latlng, mode) {
        const isPositive = mode === 'positive';
        let pointData;
        if (this.isUploadedMode()) {
            // Convert lat/lng to pixel row/col for uploaded geo-referenced image
            const px = this.platform.localImage.latlngToPixel(latlng);
            pointData = px ? { row: px.row, col: px.col } : { lat: latlng.lat, lng: latlng.lng };
        } else if (this.isLocalMode()) {
            pointData = { row: Math.round(latlng.lat), col: Math.round(latlng.lng) };
        } else {
            pointData = { lat: latlng.lat, lng: latlng.lng };
        }

        if (isPositive) {
            this.targetPoints.push(pointData);
            this.clickHistory.push('pos');
        } else {
            this.negativePoints.push(pointData);
            this.clickHistory.push('neg');
        }

        const map = this.getMap();
        if (map && window.L) {
            const fillColor = isPositive ? '#22cc44' : '#ee3344';
            const symbol = isPositive ? '+' : '−';

            const marker = L.circleMarker([latlng.lat, latlng.lng], {
                radius: 10,
                fillColor: fillColor,
                color: '#ffffff',
                weight: 3,
                opacity: 1,
                fillOpacity: 1
            }).addTo(map);

            const label = L.marker([latlng.lat, latlng.lng], {
                icon: L.divIcon({
                    className: 'td-marker-number',
                    html: `<span>${symbol}</span>`,
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                })
            }).addTo(map);

            this.targetMarkers.push({ marker, label, mode: isPositive ? 'pos' : 'neg' });

            // Duplicate marker on secondary map if compare mode is active
            const dmc = this.platform.dualMapController;
            if (dmc?.isActive && dmc.secondaryMap) {
                const marker2 = L.circleMarker([latlng.lat, latlng.lng], {
                    radius: 10, fillColor: fillColor, color: '#ffffff',
                    weight: 3, opacity: 1, fillOpacity: 1
                }).addTo(dmc.secondaryMap);
                const label2 = L.marker([latlng.lat, latlng.lng], {
                    icon: L.divIcon({
                        className: 'td-marker-number',
                        html: `<span>${symbol}</span>`,
                        iconSize: [20, 20], iconAnchor: [10, 10]
                    })
                }).addTo(dmc.secondaryMap);
                this.secondaryTargetMarkers.push({ marker: marker2, label: label2 });
            }
        }

        this._updatePointCounts();
        this._autoDetect();
    }

    undoLastPoint() {
        if (this.clickHistory.length === 0) return;
        const last = this.clickHistory.pop();
        if (last === 'pos') {
            this.targetPoints.pop();
        } else {
            this.negativePoints.pop();
        }
        // Remove last marker from primary map
        const lastMarker = this.targetMarkers.pop();
        const map = this.getMap();
        if (lastMarker) {
            if (lastMarker.marker) map?.removeLayer(lastMarker.marker);
            if (lastMarker.label) map?.removeLayer(lastMarker.label);
        }
        // Remove from secondary map
        const lastSecondary = this.secondaryTargetMarkers.pop();
        const dmc = this.platform.dualMapController;
        if (lastSecondary && dmc?.secondaryMap) {
            if (lastSecondary.marker) dmc.secondaryMap.removeLayer(lastSecondary.marker);
            if (lastSecondary.label) dmc.secondaryMap.removeLayer(lastSecondary.label);
        }
        this._updatePointCounts();

        // Auto-detect after undo
        if (this.targetPoints.length > 0) {
            this._autoDetect();
        } else {
            this._hideLiveOverlay();
        }
    }

    _updatePointCounts() {
        const countEl = document.getElementById('td-count');
        if (countEl) countEl.textContent = this.targetPoints.length + this.negativePoints.length;
        const posEl = document.getElementById('td-pos-count');
        if (posEl) posEl.textContent = `+${this.targetPoints.length}`;
        const negEl = document.getElementById('td-neg-count');
        if (negEl) negEl.textContent = `-${this.negativePoints.length}`;
        const undoBtn = document.getElementById('td-undo');
        if (undoBtn) undoBtn.disabled = this.clickHistory.length === 0;
        const clearBtn = document.getElementById('td-clear');
        if (clearBtn) clearBtn.disabled = this.clickHistory.length === 0;
        const saveBtn = document.getElementById('td-save');
        if (saveBtn) saveBtn.disabled = !this._currentLiveResult;
    }

    _setLiveStatus(text) {
        const el = document.getElementById('td-live-status');
        if (el) el.textContent = text;
    }

    clearAllPoints() {
        this.clearMarkers();
        this.targetPoints = [];
        this.negativePoints = [];
        this.clickHistory = [];
        this._hideLiveOverlay();
        // Reset detection state so new points can trigger auto-detect
        this._detectPending = false;
        this._detectAbort = false;
        clearTimeout(this._autoDetectTimer);
        this._updatePointCounts();
    }

    clearMarkers() {
        const map = this.getMap();
        this.targetMarkers.forEach(item => {
            if (item.marker) map?.removeLayer(item.marker);
            if (item.label) map?.removeLayer(item.label);
        });
        this.targetMarkers = [];
        // Clear secondary map markers
        const dmc = this.platform.dualMapController;
        this.secondaryTargetMarkers.forEach(item => {
            if (item.marker) dmc?.secondaryMap?.removeLayer(item.marker);
            if (item.label) dmc?.secondaryMap?.removeLayer(item.label);
        });
        this.secondaryTargetMarkers = [];
    }

    // ========== AUTO DETECTION ==========
    _autoDetect() {
        if (this.targetPoints.length === 0) return;
        if (this._detectPending) {
            this._detectAbort = true;
        }
        this._detectPending = true;
        this._setLiveStatus('Detecting...');

        clearTimeout(this._autoDetectTimer);
        this._autoDetectTimer = setTimeout(() => {
            this.runDetection();
        }, 100);
    }

    _hideLiveOverlay() {
        if (this._currentLiveOverlayId) {
            if (this.isLocalMode()) {
                this.platform.localImage?.hideLocalAnalysisLayer(this._currentLiveOverlayId);
            } else if (window.mapManager) {
                window.mapManager.hideAnalysisLayer(this._currentLiveOverlayId);
            }
        }
        this._currentLiveResult = null;
        this._currentLiveOverlayId = null;
        this._setLiveStatus('');
        this._updatePointCounts();
    }

    saveCurrentResult() {
        if (!this._currentLiveResult) return;

        // Move live result into saved results list
        const resultEntry = this._currentLiveResult;
        this.results.push(resultEntry);
        this._renderResultItem(resultEntry);

        // Clear live state (overlay stays as it becomes a saved result)
        this._currentLiveResult = null;
        this._currentLiveOverlayId = null;

        // Reset detection state but keep points and markers for continued use
        this._detectPending = false;
        this._detectAbort = false;
        clearTimeout(this._autoDetectTimer);

        this.platform.showNotification(`${resultEntry.algorithm} detection saved`, 'success');
    }

    // ========== RUN DETECTION (auto-triggered) ==========
    async runDetection() {
        if (this.targetPoints.length === 0) {
            this._detectPending = false;
            return;
        }

        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this._detectPending = false;
            return;
        }

        this._detectAbort = false;

        try {
            const selectedBands = this.getSelectedBands();
            let endpoint, requestBody;

            if (imageInfo.uploaded) {
                endpoint = '/api/local/uploaded/target-detection/run';
                requestBody = {
                    upload_id: imageInfo.upload_id,
                    target_points: this.targetPoints,
                    negative_points: this.negativePoints.length > 0 ? this.negativePoints : null,
                    algorithm: this.selectedAlgorithm,
                    auto_threshold: true,
                    selected_bands: selectedBands
                };
            } else if (imageInfo.local) {
                endpoint = '/api/local/target-detection/run';
                requestBody = {
                    image_dir: imageInfo.image_dir,
                    algorithm_dir: imageInfo.algorithm_dir,
                    target_points: this.targetPoints,
                    negative_points: this.negativePoints.length > 0 ? this.negativePoints : null,
                    algorithm: this.selectedAlgorithm,
                    auto_threshold: true,
                    selected_bands: selectedBands
                };
            } else {
                endpoint = '/api/target-detection/run';
                requestBody = {
                    image_id: imageInfo.id,
                    bbox: imageInfo.bbox,
                    geometry: imageInfo.geometry,
                    target_points: this.targetPoints,
                    negative_points: this.negativePoints.length > 0 ? this.negativePoints : null,
                    algorithm: this.selectedAlgorithm,
                    auto_threshold: true,
                    selected_bands: selectedBands
                };
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            // If user clicked again while waiting, discard and re-run
            if (this._detectAbort) {
                this._detectPending = false;
                this._autoDetect();
                return;
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Detection failed' }));
                throw new Error(errorData.detail || `Detection failed`);
            }

            const result = await response.json();

            if (!result || !result.detection_id) {
                throw new Error('Invalid response from server');
            }

            // Build result entry (live preview, not yet saved)
            const resultEntry = {
                id: result.detection_id,
                algorithm: result.algorithm,
                threshold: result.threshold,
                min_val: result.min_val,
                max_val: result.max_val,
                detected_pixels: result.detected_pixels,
                detection_percentage: result.detection_percentage,
                detection_result: result.detection_result,
                mask_result: result.mask_result,
                charts: result.charts,
                training_chart: result.training_chart || null,
                showingMask: false,
                visible: true
            };

            // Remove previous live overlay
            this._hideLiveOverlay();

            // Hide SAM2 preview overlay so it doesn't cover target detection result
            const sam2 = this.platform.sam2Controller;
            if (sam2?.currentOverlayId) {
                sam2._hideCurrentPreview();
            }

            // Show as live overlay
            this._currentLiveResult = resultEntry;
            this._showOverlay(resultEntry.id, resultEntry.detection_result);
            this._currentLiveOverlayId = `td-${resultEntry.id}`;

            this._updatePointCounts();
            this._setLiveStatus(`${result.algorithm} | ${result.detection_percentage}%`);

            // Show training loss chart modal for MLP algorithms
            if (result.training_chart) {
                this._showTrainingChartModal(result.training_chart, result.algorithm);
            }

        } catch (error) {
            console.error('Target detection error:', error);
            this._setLiveStatus('Error');
        } finally {
            this._detectPending = false;
        }
    }

    // ========== RESULT ITEM RENDERING ==========
    _renderResultItem(resultEntry) {
        const item = document.querySelector('.analysis-item.target-detection-option');
        const resultsList = item?.querySelector('.td-results-list');
        if (!resultsList) return;

        // Don't duplicate
        if (resultsList.querySelector(`[data-result-id="${resultEntry.id}"]`)) return;

        const r = resultEntry;
        const minVal = r.min_val;
        const maxVal = r.max_val;
        const thresholdVal = r.threshold ?? minVal;

        const resultItem = document.createElement('div');
        resultItem.className = 'td-result-item';
        resultItem.dataset.resultId = r.id;
        resultItem.dataset.visible = 'false';

        resultItem.innerHTML = `
            <div class="td-result-header">
                <img class="td-result-thumb" src="${r.detection_result?.preview_url || ''}" alt="${r.algorithm}" />
                ${r.training_chart ? `<img class="td-training-thumb" src="${r.training_chart}" alt="Loss" title="Click to view training loss" />` : ''}
                <div class="td-result-info">
                    <span class="td-result-name">${r.algorithm}</span>
                    <span class="td-result-status">Click to toggle overlay</span>
                </div>
                <div class="td-result-actions">
                    <button class="td-result-remove-btn" title="Remove">✕</button>
                </div>
            </div>
            <div class="td-result-colorbar">
                <div class="colorbar-with-threshold">
                    <div class="colorbar-track td-track">
                        <div class="colorbar-gradient td-gradient"></div>
                        <div class="colorbar-selection"></div>
                        <div class="colorbar-handle min-handle"></div>
                        <div class="colorbar-handle max-handle"></div>
                    </div>
                </div>
                <div class="colorbar-values">
                    <input type="number" class="colorbar-min-input td-result-min" value="${minVal.toFixed(3)}" step="0.001">
                    <div class="colorbar-buttons">
                        <button class="td-result-apply-btn colorbar-apply-btn">Apply</button>
                        <button class="td-result-cancel-btn colorbar-cancel-btn">Cancel</button>
                    </div>
                    <input type="number" class="colorbar-max-input td-result-max" value="${maxVal.toFixed(3)}" step="0.001">
                </div>
            </div>
            <div class="td-result-stats" style="display:none;">
                <span class="td-result-pixels"></span>
                <span class="td-result-pct"></span>
            </div>
        `;

        resultsList.appendChild(resultItem);

        // Toggle overlay on header click
        resultItem.querySelector('.td-result-header').onclick = (e) => {
            if (e.target.closest('.td-result-actions')) return;
            if (e.target.closest('.td-training-thumb')) return;
            e.stopPropagation();
            this._toggleOverlay(r.id);
        };

        // Training chart thumbnail click → show modal
        const trainingThumb = resultItem.querySelector('.td-training-thumb');
        if (trainingThumb) {
            trainingThumb.onclick = (e) => {
                e.stopPropagation();
                this._showTrainingChartModal(r.training_chart, r.algorithm);
            };
        }

        // Remove button
        resultItem.querySelector('.td-result-remove-btn').onclick = (e) => {
            e.stopPropagation();
            this._removeResult(r.id);
        };

        // Setup threshold drag/apply
        this._setupResultThreshold(resultItem, r);
    }

    _setupResultThreshold(resultItem, resultEntry) {
        const minVal = resultEntry.min_val;
        const maxVal = resultEntry.max_val;
        let currentMin = minVal;
        let currentMax = maxVal;

        const track = resultItem.querySelector('.colorbar-track');
        const selection = resultItem.querySelector('.colorbar-selection');
        const minHandle = resultItem.querySelector('.min-handle');
        const maxHandle = resultItem.querySelector('.max-handle');
        const minInput = resultItem.querySelector('.td-result-min');
        const maxInput = resultItem.querySelector('.td-result-max');
        const applyBtn = resultItem.querySelector('.td-result-apply-btn');
        const cancelBtn = resultItem.querySelector('.td-result-cancel-btn');

        const updateUI = () => {
            const trackWidth = track.offsetWidth;
            const range = maxVal - minVal;
            if (range === 0 || trackWidth === 0) return;
            const minPos = ((currentMin - minVal) / range) * trackWidth;
            const maxPos = ((currentMax - minVal) / range) * trackWidth;

            minHandle.style.left = `${minPos}px`;
            maxHandle.style.left = `${maxPos}px`;
            selection.style.left = `${minPos}px`;
            selection.style.width = `${maxPos - minPos}px`;
        };

        setTimeout(updateUI, 50);

        const startDrag = (handle, isMin) => {
            const onMove = (e) => {
                e.preventDefault();
                e.stopPropagation();
                const rect = track.getBoundingClientRect();
                const x = (e.clientX || e.touches?.[0]?.clientX) - rect.left;
                const ratio = Math.max(0, Math.min(1, x / rect.width));
                const value = minVal + ratio * (maxVal - minVal);
                if (isMin) {
                    currentMin = Math.min(value, currentMax - 0.001);
                    minInput.value = currentMin.toFixed(3);
                } else {
                    currentMax = Math.max(value, currentMin + 0.001);
                    maxInput.value = currentMax.toFixed(3);
                }
                updateUI();
            };
            const onUp = () => {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
                document.removeEventListener('touchmove', onMove);
                document.removeEventListener('touchend', onUp);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
            document.addEventListener('touchmove', onMove);
            document.addEventListener('touchend', onUp);
        };

        minHandle.addEventListener('mousedown', (e) => { e.preventDefault(); e.stopPropagation(); startDrag(minHandle, true); });
        maxHandle.addEventListener('mousedown', (e) => { e.preventDefault(); e.stopPropagation(); startDrag(maxHandle, false); });
        minHandle.addEventListener('touchstart', (e) => { e.preventDefault(); e.stopPropagation(); startDrag(minHandle, true); });
        maxHandle.addEventListener('touchstart', (e) => { e.preventDefault(); e.stopPropagation(); startDrag(maxHandle, false); });

        minInput.onchange = () => {
            currentMin = Math.max(minVal, Math.min(parseFloat(minInput.value), currentMax - 0.001));
            minInput.value = currentMin.toFixed(3);
            updateUI();
        };
        minInput.onclick = (e) => e.stopPropagation();

        maxInput.onchange = () => {
            currentMax = Math.min(maxVal, Math.max(parseFloat(maxInput.value), currentMin + 0.001));
            maxInput.value = currentMax.toFixed(3);
            updateUI();
        };
        maxInput.onclick = (e) => e.stopPropagation();

        // Apply threshold → show binary mask
        applyBtn.onclick = async (e) => {
            e.stopPropagation();
            await this._applyThreshold(resultEntry, currentMin, currentMax, resultItem);
        };

        // Cancel → restore score map
        cancelBtn.onclick = (e) => {
            e.stopPropagation();
            currentMin = minVal;
            currentMax = maxVal;
            minInput.value = minVal.toFixed(3);
            maxInput.value = maxVal.toFixed(3);
            updateUI();
            // Restore to score overlay
            resultEntry.showingMask = false;
            this._showOverlay(resultEntry.id, resultEntry.detection_result);
            resultItem.querySelector('.td-result-stats').style.display = 'none';
            resultItem.querySelector('.td-result-thumb').src = resultEntry.detection_result?.preview_url || '';
            resultItem.querySelector('.td-result-status').textContent = `${resultEntry.algorithm} — Score map`;
        };

        // Prevent colorbar clicks from toggling
        resultItem.querySelector('.td-result-colorbar').addEventListener('click', (e) => e.stopPropagation());
    }

    async _applyThreshold(resultEntry, min, max, resultItem) {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) return;

        try {
            let endpoint, reqBody;
            if (imageInfo.local) {
                endpoint = '/api/local/target-detection/apply-threshold';
                reqBody = {
                    detection_id: resultEntry.id,
                    min_threshold: min,
                    max_threshold: max,
                };
            } else {
                endpoint = '/api/target-detection/apply-threshold';
                reqBody = {
                    detection_id: resultEntry.id,
                    min_threshold: min,
                    max_threshold: max,
                    bbox: imageInfo.bbox
                };
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });

            if (!response.ok) throw new Error('Failed to apply threshold');

            const result = await response.json();
            resultEntry.mask_result = result.mask_result;
            resultEntry.detected_pixels = result.detected_pixels;
            resultEntry.detection_percentage = result.detection_percentage;
            resultEntry.showingMask = true;

            // Show mask overlay
            this._showOverlay(resultEntry.id, result.mask_result);

            // Update UI
            if (resultItem) {
                const stats = resultItem.querySelector('.td-result-stats');
                stats.style.display = 'flex';
                stats.querySelector('.td-result-pixels').textContent = `🎯 ${result.detected_pixels?.toLocaleString() || 0} px`;
                stats.querySelector('.td-result-pct').textContent = `${result.detection_percentage || 0}%`;

                if (result.mask_result?.preview_url) {
                    resultItem.querySelector('.td-result-thumb').src = result.mask_result.preview_url;
                }
                resultItem.querySelector('.td-result-status').textContent = `Mask (${min.toFixed(3)} - ${max.toFixed(3)})`;
            }

            this.platform.showNotification(`${resultEntry.algorithm}: ${result.detection_percentage}% detected`, 'success');
        } catch (error) {
            console.error('Threshold error:', error);
            this.platform.showNotification('Failed to apply threshold', 'error');
        }
    }

    // ========== OVERLAY MANAGEMENT ==========
    _isLiveOverlay(resultId) {
        return this._currentLiveResult && this._currentLiveResult.id === resultId;
    }

    _showOverlay(resultId, overlayData) {
        if (!overlayData?.overlay_url) return;

        // Remove existing for this result
        this._hideOverlay(resultId);

        const layerId = `td-${resultId}`;

        const displayName = this._isLiveOverlay(resultId) ? 'Target Detection Preview' : 'Target Detection';

        if (this.isLocalMode()) {
            const li = this.platform.localImage;
            li?.showLocalAnalysisLayer(layerId, overlayData.overlay_url, li.preloadedSize.width, li.preloadedSize.height, displayName);
        } else if (window.mapManager) {
            window.mapManager.showAnalysisLayer(layerId, overlayData.overlay_url, displayName);
        }

        // Make overlay clickable for adding target points (both modes use mapManager in load-image)
        const layer = window.mapManager?.analysisLayers?.[layerId];
        if (layer) {
            layer.on('click', (e) => {
                if (this.targetSelectMode) {
                    L.DomEvent.stopPropagation(e);
                    this._addPoint(e.latlng, this.clickMode);
                }
            });
            layer.on('contextmenu', (e) => {
                if (this.targetSelectMode) {
                    L.DomEvent.stopPropagation(e);
                    L.DomEvent.preventDefault(e);
                    this._addPoint(e.latlng, 'negative');
                }
            });
        }

        this.overlayLayers[resultId] = layerId;

        const resultItem = document.querySelector(`.td-result-item[data-result-id="${resultId}"]`);
        if (resultItem) {
            resultItem.dataset.visible = 'true';
            const entry = this.results.find(r => r.id === resultId);
            if (entry) entry.visible = true;
        }
    }

    _hideOverlay(resultId) {
        const layerId = this.overlayLayers[resultId];
        if (!layerId) return;

        if (this.isLocalMode()) {
            this.platform.localImage?.hideLocalAnalysisLayer(layerId);
        } else if (window.mapManager) {
            window.mapManager.hideAnalysisLayer(layerId);
        }

        delete this.overlayLayers[resultId];

        const resultItem = document.querySelector(`.td-result-item[data-result-id="${resultId}"]`);
        if (resultItem) {
            resultItem.dataset.visible = 'false';
            const entry = this.results.find(r => r.id === resultId);
            if (entry) entry.visible = false;
        }
    }

    _toggleOverlay(resultId) {
        const entry = this.results.find(r => r.id === resultId);
        if (!entry) return;

        const resultItem = document.querySelector(`.td-result-item[data-result-id="${resultId}"]`);
        const isVisible = resultItem?.dataset.visible === 'true';

        if (isVisible) {
            this._hideOverlay(resultId);
            if (resultItem) {
                resultItem.querySelector('.td-result-status').textContent = 'Click to toggle overlay';
            }
        } else {
            const overlayData = entry.showingMask ? entry.mask_result : entry.detection_result;
            this._showOverlay(resultId, overlayData);
            if (resultItem) {
                resultItem.querySelector('.td-result-status').textContent = `${entry.algorithm} — Click to hide`;
            }
        }
    }

    _removeResult(resultId) {
        this._hideOverlay(resultId);
        this.results = this.results.filter(r => r.id !== resultId);
        const el = document.querySelector(`.td-result-item[data-result-id="${resultId}"]`);
        if (el) el.remove();
    }

    // ========== HELPERS ==========
    updateStatus(text) {
        const item = document.querySelector('.analysis-item.target-detection-option');
        const statusEl = item?.querySelector('.analysis-status');
        if (statusEl) {
            statusEl.textContent = text;
        }
    }

    updateItemActive(active) {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (item) {
            item.classList.toggle('active', active);
        }
    }

    reset() {
        this.stopTargetSelection();
        this.clearMarkers();

        // Remove live overlay
        this._hideLiveOverlay();

        // Remove all overlay layers
        Object.keys(this.overlayLayers).forEach(id => this._hideOverlay(id));

        this.results = [];
        this.targetPoints = [];
        this.negativePoints = [];
        this.clickHistory = [];
        this.clickMode = 'positive';
        this.overlayLayers = {};
        this._currentLiveResult = null;
        this._currentLiveOverlayId = null;

        const item = document.querySelector('.analysis-item.target-detection-option');
        item?.querySelector('.td-ui')?.remove();
        item?.classList.remove('expanded', 'active');

        const infoEl = item?.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    /**
     * Sync existing markers to the secondary map when compare mode activates
     */
    _syncMarkersToSecondary() {
        const dmc = this.platform.dualMapController;
        if (!dmc?.isActive || !dmc.secondaryMap) return;

        // Clear any stale secondary markers
        this.secondaryTargetMarkers.forEach(item => {
            if (item.marker) dmc.secondaryMap.removeLayer(item.marker);
            if (item.label) dmc.secondaryMap.removeLayer(item.label);
        });
        this.secondaryTargetMarkers = [];

        // Duplicate all current markers
        this.targetMarkers.forEach(item => {
            const latlng = item.marker.getLatLng();
            const isPositive = item.mode === 'pos';
            const fillColor = isPositive ? '#22cc44' : '#ee3344';
            const symbol = isPositive ? '+' : '−';

            const marker2 = L.circleMarker([latlng.lat, latlng.lng], {
                radius: 10, fillColor: fillColor, color: '#ffffff',
                weight: 3, opacity: 1, fillOpacity: 1
            }).addTo(dmc.secondaryMap);
            const label2 = L.marker([latlng.lat, latlng.lng], {
                icon: L.divIcon({
                    className: 'td-marker-number',
                    html: `<span>${symbol}</span>`,
                    iconSize: [20, 20], iconAnchor: [10, 10]
                })
            }).addTo(dmc.secondaryMap);
            this.secondaryTargetMarkers.push({ marker: marker2, label: label2 });
        });
    }

    cleanup() {
        this.reset();
    }

    // ========== TRAINING CHART MODAL ==========
    _showTrainingChartModal(chartUrl, algorithm) {
        // Remove existing modal
        document.querySelector('.td-training-modal')?.remove();

        const modal = document.createElement('div');
        modal.className = 'td-training-modal';
        modal.innerHTML = `
            <div class="td-training-modal-backdrop"></div>
            <div class="td-training-modal-content">
                <div class="td-training-modal-header">
                    <span>${algorithm} Training Loss</span>
                    <button class="td-training-modal-close">✕</button>
                </div>
                <img src="${chartUrl}" alt="Training Loss Curve" />
            </div>
        `;

        document.body.appendChild(modal);

        const close = () => modal.remove();
        modal.querySelector('.td-training-modal-close').onclick = close;
        modal.querySelector('.td-training-modal-backdrop').onclick = close;
    }
}
