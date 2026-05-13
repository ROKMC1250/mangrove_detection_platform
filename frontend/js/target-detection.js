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

        // Per-slot stash: see /home/hjh1037/.claude/plans/2-quiet-metcalfe.md §1.1
        this._slotStash = { A: null, B: null };
        // Slot the currently-pending detection was kicked off on — used to
        // drop completions that arrive after the user has switched away.
        this._detectOriginSlot = null;

        this.handleMapClick = this.handleMapClick.bind(this);
        this._handleContextMenu = this._handleContextMenu.bind(this);

        // Sync markers and click handlers when compare mode is toggled
        window.addEventListener('comparemap:activated', () => {
            this._syncMarkersToSecondary();
            // If target selection is active, also attach to the new secondary map
            if (this.targetSelectMode) {
                const dmc = this.platform.dualMapController;
                if (dmc?.secondaryMap) {
                    dmc.secondaryMap.getContainer().classList.add('prompt-cursor');
                    dmc.secondaryMap.on('click', this.handleMapClick);
                    dmc.secondaryMap.getContainer().addEventListener('contextmenu', this._handleContextMenu);
                }
            }
        });
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
                <button id="td-pick" class="td-btn ${this.targetSelectMode ? 'active' : ''}" title="Left-click: positive, Right-click: negative">📍 <span id="td-count">${this.targetPoints.length + this.negativePoints.length}</span></button>
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
        this.stopTargetSelection();
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
        // Stop SAM3 selection if active (prevent both listening at once)
        const sam3 = this.platform.sam3Controller;
        if (sam3?.selectMode) {
            sam3.stopSelection();
        }

        this.targetSelectMode = true;
        document.getElementById('td-pick')?.classList.add('active');

        // Attach to primary map
        const map = this.getMap();
        if (map) {
            map.getContainer().classList.add('prompt-cursor');
            map.on('click', this.handleMapClick);
            map.getContainer().addEventListener('contextmenu', this._handleContextMenu);
            this._attachOverlayClickHandlers(map);
        }

        // Also attach to secondary map if compare mode is active
        const dmc = this.platform.dualMapController;
        if (dmc?.isActive && dmc.secondaryMap) {
            dmc.secondaryMap.getContainer().classList.add('prompt-cursor');
            dmc.secondaryMap.on('click', this.handleMapClick);
            dmc.secondaryMap.getContainer().addEventListener('contextmenu', this._handleContextMenu);
        }

        this.platform.showNotification('Left-click: positive target, Right-click: negative', 'info');
    }

    stopTargetSelection() {
        this.targetSelectMode = false;
        document.getElementById('td-pick')?.classList.remove('active');

        // Detach from primary map
        const map = this.getMap();
        if (map) {
            map.getContainer().classList.remove('prompt-cursor');
            map.off('click', this.handleMapClick);
            map.getContainer().removeEventListener('contextmenu', this._handleContextMenu);
            this._detachOverlayClickHandlers();
        }

        // Detach from secondary map
        const dmc = this.platform.dualMapController;
        if (dmc?.isActive && dmc.secondaryMap) {
            dmc.secondaryMap.getContainer().classList.remove('prompt-cursor');
            dmc.secondaryMap.off('click', this.handleMapClick);
            dmc.secondaryMap.getContainer().removeEventListener('contextmenu', this._handleContextMenu);
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
        // Resolve the originating map from the event target so right-clicks
        // on the secondary (right) map in compare mode don't get projected
        // against the primary map's container.
        const map = this._resolveMapFromEvent(e);
        if (!map) return;
        const rect = map.getContainer().getBoundingClientRect();
        const latlng = map.containerPointToLatLng(L.point(
            e.clientX - rect.left,
            e.clientY - rect.top
        ));
        this._addPoint(latlng, 'negative');
    }

    _resolveMapFromEvent(e) {
        const dmc = this.platform.dualMapController;
        const container = e.currentTarget;
        if (dmc?.isActive && dmc.secondaryMap && container === dmc.secondaryMap.getContainer()) {
            return dmc.secondaryMap;
        }
        return this.getMap();
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
            const { marker, label } = this._createMarkerPair(latlng, isPositive, map);
            this.targetMarkers.push({ marker, label, mode: isPositive ? 'pos' : 'neg' });

            // Duplicate marker on secondary map if compare mode is active
            const dmc = this.platform.dualMapController;
            if (dmc?.isActive && dmc.secondaryMap) {
                const { marker: m2, label: l2 } = this._createMarkerPair(latlng, isPositive, dmc.secondaryMap);
                this.secondaryTargetMarkers.push({ marker: m2, label: l2 });
            }
        }

        this._updatePointCounts();
        this._autoDetect();
    }

    /**
     * Create a marker + label pair for a target point on the given map.
     */
    _createMarkerPair(latlng, isPositive, targetMap) {
        const fillColor = isPositive ? '#00bfff' : '#ff4444';
        const borderColor = isPositive ? '#0088cc' : '#cc0000';
        const symbol = isPositive ? '+' : '−';

        const marker = L.circleMarker([latlng.lat, latlng.lng], {
            radius: 6,
            fillColor,
            color: borderColor,
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9
        }).addTo(targetMap);

        const label = L.marker([latlng.lat, latlng.lng], {
            icon: L.divIcon({
                className: `td-marker-number ${isPositive ? 'td-positive' : 'td-negative'}`,
                html: `<span>${symbol}</span>`,
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            })
        }).addTo(targetMap);

        return { marker, label };
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
        // Stamp the slot this run belongs to. If the user switches slots
        // before completion, _handleDetectionResult drops the result.
        this._detectOriginSlot = this.platform.currentSlot;

        try {
            const selectedBands = this.getSelectedBands();
            let endpoint, requestBody;
            const isMLP = this.selectedAlgorithm === 'MLP_AMF' || this.selectedAlgorithm === 'MLP_ACE';

            if (imageInfo.uploaded) {
                // MLP uses the streaming variant so the loss curve animates
                // live; non-MLP algorithms run to completion without
                // intermediate events.
                endpoint = isMLP
                    ? '/api/local/uploaded/target-detection/run-stream'
                    : '/api/local/uploaded/target-detection/run';
                requestBody = {
                    upload_id: imageInfo.upload_id,
                    target_points: this.targetPoints,
                    negative_points: this.negativePoints.length > 0 ? this.negativePoints : null,
                    algorithm: this.selectedAlgorithm,
                    auto_threshold: true,
                    selected_bands: selectedBands
                };
            } else if (imageInfo.local) {
                endpoint = isMLP
                    ? '/api/local/target-detection/run-stream'
                    : '/api/local/target-detection/run';
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
                // Satellite source — use SSE streaming for MLP algorithms.
                endpoint = isMLP ? '/api/target-detection/run-stream' : '/api/target-detection/run';
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

            // Any MLP run is now streaming regardless of image source.
            if (isMLP) {
                await this._runDetectionSSE(endpoint, requestBody);
            } else {
                await this._runDetectionFetch(endpoint, requestBody);
            }

        } catch (error) {
            console.error('Target detection error:', error);
            this._setLiveStatus('Error');
        } finally {
            this._detectPending = false;
        }
    }

    async _runDetectionFetch(endpoint, requestBody) {
        const isMLP = requestBody.algorithm === 'MLP_AMF' || requestBody.algorithm === 'MLP_ACE';
        if (isMLP) {
            this._setLiveStatus('Training...');
            this._showLiveTrainingChart(requestBody.algorithm);
        }

        // Non-MLP algorithms (SAM/ACE/RXD/CEM/MF) have no training UI of their
        // own — show the global loading overlay so the user has explicit feedback
        // until the result image is on screen. MLP path keeps its bespoke chart.
        const useGlobalLoader = !isMLP && this.platform && typeof this.platform.withLoading === 'function';

        const runFetch = async () => {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (this._detectAbort) { this._detectPending = false; this._autoDetect(); return; }

            if (!response.ok) {
                if (isMLP) this._closeLiveTrainingChart();
                const errorData = await response.json().catch(() => ({ detail: 'Detection failed' }));
                throw new Error(errorData.detail || 'Detection failed');
            }

            const result = await response.json();

            if (isMLP && result.training_chart) {
                this._showStaticTrainingChart(result.training_chart, requestBody.algorithm);
            } else if (isMLP) {
                this._closeLiveTrainingChart();
            }

            this._handleDetectionResult(result);
            // Hold the loading overlay until the analysis image actually paints.
            if (result && result.detection_id) {
                await this._awaitOverlayPaint(`td-${result.detection_id}`);
            }
        };

        if (useGlobalLoader) {
            const algo = requestBody.algorithm || 'detection';
            await this.platform.withLoading(`Running target detection (${algo})...`, runFetch);
        } else {
            await runFetch();
        }
    }

    /**
     * Wait until the just-added analysis overlay's underlying <img> finishes
     * decoding, so the loading UI doesn't disappear before the result is visible.
     * Resolves either way; never rejects. 15s defensive timeout.
     */
    _awaitOverlayPaint(layerId) {
        return new Promise(resolve => {
            const layer = window.mapManager?.analysisLayers?.[layerId];
            if (!layer || typeof layer.once !== 'function') return resolve();
            const el = (typeof layer.getElement === 'function') ? layer.getElement() : null;
            if (el && el.complete && el.naturalHeight > 0) return resolve();
            let settled = false;
            const finish = () => { if (settled) return; settled = true; resolve(); };
            const t = setTimeout(finish, 15000);
            layer.once('load',  () => { clearTimeout(t); finish(); });
            layer.once('error', () => { clearTimeout(t); finish(); });
        });
    }

    async _runDetectionSSE(endpoint, requestBody) {
        this._setLiveStatus('Training...');
        this._showLiveTrainingChart(requestBody.algorithm);

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Detection failed' }));
            throw new Error(errorData.detail || 'Detection failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        // `eventType` persists across read() chunks. The final "done"
        // event carries base64 overlays that easily exceed TCP chunk
        // size, so "event: done" and its "data: ..." line often arrive
        // in separate reads. A fresh chunk-local declaration would lose
        // the event type before the data line is processed.
        let eventType = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (this._detectAbort) { reader.cancel(); this._detectPending = false; this._autoDetect(); return; }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ') && eventType) {
                    const data = JSON.parse(line.slice(6));
                    if (eventType === 'training') {
                        this._updateLiveTrainingChart(data);
                        this._setLiveStatus(`Training ${data.step}/${data.n_steps}`);
                    } else if (eventType === 'done') {
                        this._closeLiveTrainingChart();
                        this._handleDetectionResult(data);
                    } else if (eventType === 'error') {
                        this._closeLiveTrainingChart();
                        throw new Error(data.detail || 'Detection failed');
                    }
                    eventType = null;
                }
            }
        }
    }

    _handleDetectionResult(result) {
        if (!result || !result.detection_id) {
            throw new Error('Invalid response from server');
        }

        // Drop completions whose origin slot is no longer active. Without
        // this, an MLP_AMF training that finishes after the user clicked
        // Time B would write into slot B's results.
        const origin = this._detectOriginSlot;
        this._detectOriginSlot = null;
        if (origin && origin !== this.platform.currentSlot) {
            this.platform.showNotification(
                `Detection on Time ${origin} dropped — you switched slots mid-run.`,
                'warning'
            );
            return;
        }

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
            showingMask: false,
            visible: true
        };

        this._hideLiveOverlay();

        const sam3 = this.platform.sam3Controller;
        if (sam3?.currentOverlayId) { sam3._hideCurrentPreview(); }

        resultEntry.slotId = this.platform.currentSlot;
        this._currentLiveResult = resultEntry;
        this._showOverlay(resultEntry.id, resultEntry.detection_result);
        this._currentLiveOverlayId = `td-${resultEntry.id}`;

        // Register so the layer is tracked, but mark hasMask=false until
        // the user explicitly applies a threshold below — change-detection
        // requires that explicit step before the result is selectable.
        if (typeof this.platform.registerSlotAnalysis === 'function') {
            this.platform.registerSlotAnalysis({
                id: resultEntry.id,
                type: 'detection',
                name: `${result.algorithm || 'Target'} · ${String(resultEntry.id).slice(-6)}`,
                hasMask: false,
            });
        }

        this._updatePointCounts();
        this._setLiveStatus(`${result.algorithm} | ${result.detection_percentage}%`);
    }

    // ========== RESULT ITEM RENDERING ==========
    _renderResultItem(resultEntry) {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;
        const resultsList = item.querySelector('.td-results-list');
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
            e.stopPropagation();
            this._toggleOverlay(r.id);
        };

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

            // Re-register so change-detection sees the newest threshold's
            // binary mask. hasMask stays true.
            if (typeof this.platform.registerSlotAnalysis === 'function') {
                this.platform.registerSlotAnalysis({
                    id: resultEntry.id,
                    type: 'detection',
                    name: `${resultEntry.algorithm || 'Target'} · ${String(resultEntry.id).slice(-6)}`,
                    hasMask: true,
                });
            }

            // Show mask overlay
            this._showOverlay(resultEntry.id, result.mask_result, true);

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

    _showOverlay(resultId, overlayData, isBinary = false) {
        if (!overlayData?.overlay_url) return;

        // Remove existing for this result
        this._hideOverlay(resultId);

        const layerId = `td-${resultId}`;

        const displayName = this._isLiveOverlay(resultId) ? 'Target Detection Preview' : 'Target Detection';

        if (this.isLocalMode()) {
            const li = this.platform.localImage;
            li?.showLocalAnalysisLayer(layerId, overlayData.overlay_url, li.preloadedSize.width, li.preloadedSize.height, displayName, isBinary);
        } else if (window.mapManager) {
            window.mapManager.showAnalysisLayer(layerId, overlayData.overlay_url, displayName, null, isBinary);
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
            this._showOverlay(resultId, overlayData, !!entry.showingMask);
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

    // ========== Per-slot stash (Time A / Time B) ==========

    _freezeToSlot(slotId) {
        if (slotId !== 'A' && slotId !== 'B') return;
        this._slotStash[slotId] = {
            results: this.results.slice(),
            overlayLayers: { ...this.overlayLayers },
            targetPoints: this.targetPoints.slice(),
            negativePoints: this.negativePoints.slice(),
            clickHistory: this.clickHistory.slice(),
            targetMarkers: this.targetMarkers.slice(),
            secondaryTargetMarkers: this.secondaryTargetMarkers.slice(),
            _currentLiveResult: this._currentLiveResult,
            _currentLiveOverlayId: this._currentLiveOverlayId,
            _detectPending: this._detectPending,
            selectedAlgorithm: this.selectedAlgorithm,
            selectedBands: this.selectedBands,
            clickMode: this.clickMode,
        };

        // Detach markers from both maps. Use the snapshot — don't call
        // clearMarkers() which nukes the live arrays.
        const primaryMap = this.getMap();
        this._slotStash[slotId].targetMarkers.forEach(({ marker, label }) => {
            if (marker && primaryMap?.hasLayer?.(marker)) primaryMap.removeLayer(marker);
            if (label && primaryMap?.hasLayer?.(label)) primaryMap.removeLayer(label);
        });
        const dmc = this.platform.dualMapController;
        if (dmc?.secondaryMap) {
            this._slotStash[slotId].secondaryTargetMarkers.forEach(({ marker, label }) => {
                if (marker && dmc.secondaryMap.hasLayer?.(marker)) dmc.secondaryMap.removeLayer(marker);
                if (label && dmc.secondaryMap.hasLayer?.(label)) dmc.secondaryMap.removeLayer(label);
            });
        }

        // The live-training modal lives under <body>, not inside .td-ui, so
        // showAnalysisResults doesn't tear it down. Close it on freeze.
        if (typeof this._closeLiveTrainingChart === 'function') {
            try { this._closeLiveTrainingChart(); } catch (e) { /* no-op */ }
        }
    }

    _thawFromSlot(slotId) {
        const stash = (slotId === 'A' || slotId === 'B') ? this._slotStash[slotId] : null;
        if (stash) {
            this.results = Array.isArray(stash.results) ? stash.results.slice() : [];
            this.overlayLayers = { ...(stash.overlayLayers || {}) };
            this.targetPoints = Array.isArray(stash.targetPoints) ? stash.targetPoints.slice() : [];
            this.negativePoints = Array.isArray(stash.negativePoints) ? stash.negativePoints.slice() : [];
            this.clickHistory = Array.isArray(stash.clickHistory) ? stash.clickHistory.slice() : [];
            this.targetMarkers = Array.isArray(stash.targetMarkers) ? stash.targetMarkers.slice() : [];
            this.secondaryTargetMarkers = Array.isArray(stash.secondaryTargetMarkers)
                ? stash.secondaryTargetMarkers.slice() : [];
            this._currentLiveResult = stash._currentLiveResult || null;
            this._currentLiveOverlayId = stash._currentLiveOverlayId || null;
            this._detectPending = !!stash._detectPending;
            this.selectedAlgorithm = stash.selectedAlgorithm || this.selectedAlgorithm;
            this.selectedBands = stash.selectedBands;
            this.clickMode = stash.clickMode || 'positive';

            // Re-attach markers to the (new) current map.
            const primaryMap = this.getMap();
            if (primaryMap) {
                this.targetMarkers.forEach(({ marker, label }) => {
                    if (marker && !primaryMap.hasLayer?.(marker)) marker.addTo(primaryMap);
                    if (label && !primaryMap.hasLayer?.(label)) label.addTo(primaryMap);
                });
            }
            const dmc = this.platform.dualMapController;
            if (dmc?.isActive && dmc.secondaryMap) {
                this.secondaryTargetMarkers.forEach(({ marker, label }) => {
                    if (marker && !dmc.secondaryMap.hasLayer?.(marker)) marker.addTo(dmc.secondaryMap);
                    if (label && !dmc.secondaryMap.hasLayer?.(label)) label.addTo(dmc.secondaryMap);
                });
            } else {
                // Compare mode was turned off while away — drop any stale refs
                // so they don't reappear the next time compare is toggled.
                this.secondaryTargetMarkers = [];
            }

            this._slotStash[slotId] = null;
        } else {
            // First visit to this slot — fresh state.
            this.results = [];
            this.overlayLayers = {};
            this.targetPoints = [];
            this.negativePoints = [];
            this.clickHistory = [];
            this.targetMarkers = [];
            this.secondaryTargetMarkers = [];
            this._currentLiveResult = null;
            this._currentLiveOverlayId = null;
            this._detectPending = false;
            this.clickMode = 'positive';
        }
    }

    handleSlotChange(oldSlot, newSlot) {
        if (oldSlot === newSlot) return;
        this.stopTargetSelection();
        this._freezeToSlot(oldSlot);
        this._thawFromSlot(newSlot);
        const item = document.querySelector('.analysis-item.target-detection-option');
        const list = item?.querySelector('.td-results-list');
        if (list) {
            list.innerHTML = '';
            this.results.forEach(r => this._renderResultItem(r));
            if (typeof this._updatePointCounts === 'function') this._updatePointCounts();
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
            const { marker: m2, label: l2 } = this._createMarkerPair(latlng, isPositive, dmc.secondaryMap);
            this.secondaryTargetMarkers.push({ marker: m2, label: l2 });
        });
    }

    cleanup() {
        this.reset();
    }

    // ========== LIVE TRAINING CHART ==========

    _showLiveTrainingChart(algorithm) {
        this._closeLiveTrainingChart();

        const modal = document.createElement('div');
        modal.className = 'td-training-modal';
        modal.id = 'td-live-training-modal';
        modal.innerHTML = `
            <div class="td-training-modal-content td-training-live">
                <div class="td-training-modal-header">
                    <span>${algorithm} Training</span>
                    <span class="td-training-step" id="td-training-step">0 / ?</span>
                </div>
                <div class="td-training-chart-wrap">
                    <canvas id="td-live-loss-canvas"></canvas>
                </div>
                <div class="td-training-loss-value" id="td-training-loss-value">Loss: —</div>
            </div>
        `;
        document.body.appendChild(modal);

        const ctx = document.getElementById('td-live-loss-canvas').getContext('2d');
        this._liveChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: '#7c4dff',
                    backgroundColor: 'rgba(124, 77, 255, 0.08)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 },
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Step', font: { size: 11 } }, ticks: { font: { size: 10 } } },
                    y: { title: { display: true, text: 'Loss', font: { size: 11 } }, ticks: { font: { size: 10 } }, beginAtZero: false }
                }
            }
        });
    }

    _updateLiveTrainingChart(data) {
        if (!this._liveChart) return;

        const history = data.loss_history || [];
        this._liveChart.data.labels = history.map((_, i) => i + 1);
        this._liveChart.data.datasets[0].data = history;
        this._liveChart.update('none'); // no animation for speed

        const stepEl = document.getElementById('td-training-step');
        if (stepEl) stepEl.textContent = `${data.step} / ${data.n_steps}`;

        const lossEl = document.getElementById('td-training-loss-value');
        if (lossEl && data.loss != null) lossEl.textContent = `Loss: ${data.loss.toFixed(6)}`;
    }

    _showStaticTrainingChart(chartDataUri, algorithm) {
        // Replace the live chart modal with the static image from backend
        this._closeLiveTrainingChart();

        const modal = document.createElement('div');
        modal.className = 'td-training-modal';
        modal.id = 'td-live-training-modal';
        modal.innerHTML = `
            <div class="td-training-modal-content td-training-live">
                <div class="td-training-modal-header">
                    <span>${algorithm} Training Complete</span>
                    <button class="td-training-close-btn" title="Close">✕</button>
                </div>
                <div class="td-training-chart-wrap">
                    <img src="${chartDataUri}" style="width:100%;height:100%;object-fit:contain;" />
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        modal.querySelector('.td-training-close-btn')?.addEventListener('click', () => modal.remove());
        // Auto-close after 5 seconds
        setTimeout(() => modal.remove(), 5000);
    }

    _closeLiveTrainingChart() {
        if (this._liveChart) {
            this._liveChart.destroy();
            this._liveChart = null;
        }
        document.getElementById('td-live-training-modal')?.remove();
    }
}
