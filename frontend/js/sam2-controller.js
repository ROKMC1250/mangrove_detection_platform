/**
 * SAM2 Segmentation Controller
 *
 * Flow:
 * 1. Click SAM2 item -> Show: [Select] [+/-] [Undo] [Clear] [Save]
 * 2. Place positive/negative point prompts on map
 * 3. Auto-segment on every point change (real-time preview)
 * 4. Click Save -> mask saved, points cleared, ready for next object
 * 5. Repeat for multiple objects; saved masks listed below with toggle/remove
 */

class SAM2Controller {
    constructor(platformController) {
        this.platform = platformController;

        // Point prompt state
        this.selectMode = false;
        this.clickMode = 'positive';
        this.positivePoints = [];
        this.negativePoints = [];
        this.markers = [];
        this.clickHistory = [];

        // Mask state
        this.savedMasks = [];
        this.currentMaskId = null;
        this.currentOverlayId = null;

        // Auto-segment state
        this._segmentPending = false;
        this._segmentAbort = null;

        this.handleMapClick = this.handleMapClick.bind(this);
        this._handleContextMenu = this._handleContextMenu.bind(this);
    }

    isLocalMode() {
        const li = this.platform.localImage;
        return !!li?.isUploadedImageActive;
    }

    isUploadedMode() {
        const li = this.platform.localImage;
        return li?.currentMode === 'load-image' && !!li?.isUploadedImageActive;
    }

    getMap() {
        if (this.isUploadedMode()) return window.mapManager?.map;
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
        const imageId = this.platform.selectedImageId;
        if (!imageId) return null;
        return {
            id: imageId,
            bbox: this.platform.processedBbox || window.mapManager?.getCurrentBounds(),
            geometry: this.platform.processedGeometry || window.mapManager?.getCurrentGeoJSON()?.geometry
        };
    }

    handleItemClick() {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification(this.isLocalMode() ? 'Load an image first' : 'Process an image first', 'warning');
            return;
        }

        const item = document.querySelector('.analysis-item.sam2-option');
        const hasUI = item?.querySelector('.sam2-ui');

        if (hasUI) {
            this.hideUI();
        } else {
            this.showSetupUI();
        }
    }

    // ========== SETUP UI ==========
    showSetupUI() {
        const item = document.querySelector('.analysis-item.sam2-option');
        if (!item) return;

        item.querySelector('.sam2-ui')?.remove();
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const ui = document.createElement('div');
        ui.className = 'sam2-ui sam2-setup';
        ui.innerHTML = `
            <div class="sam2-row">
                <button id="sam2-pick" class="sam2-btn ${this.selectMode ? 'active' : ''}" title="Left-click: positive, Right-click: negative">📍 <span id="sam2-count">${this.positivePoints.length + this.negativePoints.length}</span></button>
                <button id="sam2-mode-toggle" class="sam2-btn sam2-mode-positive" title="Current: Positive (click to switch)">+</button>
                <button id="sam2-undo" class="sam2-btn" title="Undo last point" ${this.clickHistory.length === 0 ? 'disabled' : ''}>↩</button>
                <button id="sam2-clear" class="sam2-btn" title="Clear all points" ${this.clickHistory.length === 0 ? 'disabled' : ''}>🗑</button>
                <button id="sam2-save" class="sam2-btn save" ${!this.currentMaskId ? 'disabled' : ''} title="Save current mask and start next object">💾 Save</button>
            </div>
            <div class="sam2-point-info" style="font-size:11px; color:#888; padding:2px 4px;">
                <span id="sam2-pos-count" style="color:#22cc44;">+${this.positivePoints.length}</span>
                <span id="sam2-neg-count" style="color:#ee3344; margin-left:6px;">-${this.negativePoints.length}</span>
                <span id="sam2-status" style="margin-left:8px; color:#7b1fa2;"></span>
            </div>
            <div class="sam2-saved-list"></div>
        `;

        item.appendChild(ui);
        item.classList.add('expanded');

        this.savedMasks.forEach(m => this._renderSavedMask(m));

        document.getElementById('sam2-pick').onclick = (e) => {
            e.stopPropagation();
            this.toggleSelection();
        };

        document.getElementById('sam2-mode-toggle').onclick = (e) => {
            e.stopPropagation();
            this.clickMode = this.clickMode === 'positive' ? 'negative' : 'positive';
            const btn = document.getElementById('sam2-mode-toggle');
            if (this.clickMode === 'positive') {
                btn.textContent = '+';
                btn.className = 'sam2-btn sam2-mode-positive';
                btn.title = 'Current: Positive (click to switch)';
            } else {
                btn.textContent = '\u2212';
                btn.className = 'sam2-btn sam2-mode-negative';
                btn.title = 'Current: Negative (click to switch)';
            }
        };

        document.getElementById('sam2-undo').onclick = (e) => {
            e.stopPropagation();
            this.undoLastPoint();
        };

        document.getElementById('sam2-clear').onclick = (e) => {
            e.stopPropagation();
            this.clearCurrentPoints();
        };

        document.getElementById('sam2-save').onclick = (e) => {
            e.stopPropagation();
            this.saveAndNext();
        };
    }

    hideUI() {
        const item = document.querySelector('.analysis-item.sam2-option');
        if (!item) return;
        item.querySelector('.sam2-ui')?.remove();
        item.classList.remove('expanded');
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    // ========== POINT SELECTION ==========
    toggleSelection() {
        if (this.selectMode) {
            this.stopSelection();
        } else {
            this.startSelection();
        }
    }

    startSelection() {
        // Stop target detection selection if active (prevent both listening at once)
        const td = this.platform.targetDetection;
        if (td?.targetSelectMode) {
            td.stopTargetSelection();
        }

        this.selectMode = true;
        document.getElementById('sam2-pick')?.classList.add('active');
        const map = this.getMap();
        if (map) {
            map.getContainer().classList.add('prompt-cursor');
            map.on('click', this.handleMapClick);
            map.getContainer().addEventListener('contextmenu', this._handleContextMenu);
        }
        this.platform.showNotification('Left-click: positive target, Right-click: negative', 'info');
    }

    stopSelection() {
        this.selectMode = false;
        document.getElementById('sam2-pick')?.classList.remove('active');
        const map = this.getMap();
        if (map) {
            map.getContainer().classList.remove('prompt-cursor');
            map.off('click', this.handleMapClick);
            map.getContainer().removeEventListener('contextmenu', this._handleContextMenu);
        }
    }

    _handleContextMenu(e) {
        if (!this.selectMode) return;
        e.preventDefault();
        e.stopPropagation();
        const map = this.getMap();
        if (!map) return;
        const latlng = map.containerPointToLatLng(L.point(
            e.clientX - map.getContainer().getBoundingClientRect().left,
            e.clientY - map.getContainer().getBoundingClientRect().top
        ));
        this._addPoint(latlng, 'negative');
    }

    handleMapClick(e) {
        if (!this.selectMode) return;
        this._addPoint(e.latlng, this.clickMode);
    }

    _addPoint(latlng, mode) {
        const isPositive = mode === 'positive';
        let pointData;
        if (this.isUploadedMode()) {
            const px = this.platform.localImage.latlngToPixel(latlng);
            pointData = px ? { row: px.row, col: px.col } : { lat: latlng.lat, lng: latlng.lng };
        } else {
            pointData = { lat: latlng.lat, lng: latlng.lng };
        }

        if (isPositive) {
            this.positivePoints.push(pointData);
            this.clickHistory.push('pos');
        } else {
            this.negativePoints.push(pointData);
            this.clickHistory.push('neg');
        }

        const map = this.getMap();
        if (map && window.L) {
            const fillColor = isPositive ? '#22cc44' : '#ee3344';
            const symbol = isPositive ? '+' : '\u2212';

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
                    className: 'sam2-marker-label',
                    html: `<span>${symbol}</span>`,
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                })
            }).addTo(map);

            this.markers.push({ marker, label, mode: isPositive ? 'pos' : 'neg' });
        }

        this._updatePointCounts();

        // Auto-segment on every point change
        this._autoSegment();
    }

    undoLastPoint() {
        if (this.clickHistory.length === 0) return;
        const last = this.clickHistory.pop();
        if (last === 'pos') {
            this.positivePoints.pop();
        } else {
            this.negativePoints.pop();
        }
        const lastMarker = this.markers.pop();
        const map = this.getMap();
        if (lastMarker) {
            if (lastMarker.marker) map?.removeLayer(lastMarker.marker);
            if (lastMarker.label) map?.removeLayer(lastMarker.label);
        }
        this._updatePointCounts();

        // Auto-segment after undo
        if (this.positivePoints.length > 0) {
            this._autoSegment();
        } else {
            this._hideCurrentPreview();
        }
    }

    clearCurrentPoints() {
        const map = this.getMap();
        this.markers.forEach(item => {
            if (item.marker) map?.removeLayer(item.marker);
            if (item.label) map?.removeLayer(item.label);
        });
        this.markers = [];
        this.positivePoints = [];
        this.negativePoints = [];
        this.clickHistory = [];
        this._updatePointCounts();
        this._hideCurrentPreview();
    }

    _updatePointCounts() {
        const countEl = document.getElementById('sam2-count');
        if (countEl) countEl.textContent = this.positivePoints.length + this.negativePoints.length;
        const posEl = document.getElementById('sam2-pos-count');
        if (posEl) posEl.textContent = `+${this.positivePoints.length}`;
        const negEl = document.getElementById('sam2-neg-count');
        if (negEl) negEl.textContent = `-${this.negativePoints.length}`;
        const undoBtn = document.getElementById('sam2-undo');
        if (undoBtn) undoBtn.disabled = this.clickHistory.length === 0;
        const clearBtn = document.getElementById('sam2-clear');
        if (clearBtn) clearBtn.disabled = this.clickHistory.length === 0;
        const saveBtn = document.getElementById('sam2-save');
        if (saveBtn) saveBtn.disabled = !this.currentMaskId;
    }

    _setStatus(text) {
        const el = document.getElementById('sam2-status');
        if (el) el.textContent = text;
    }

    // ========== AUTO SEGMENTATION ==========
    _autoSegment() {
        if (this.positivePoints.length === 0) return;
        if (this._segmentPending) {
            this._segmentAbort = true;
        }
        this._segmentPending = true;
        this._setStatus('Segmenting...');

        // Small delay to batch rapid clicks
        clearTimeout(this._autoSegmentTimer);
        this._autoSegmentTimer = setTimeout(() => {
            this._runSegmentation();
        }, 100);
    }

    async _runSegmentation() {
        if (this.positivePoints.length === 0) {
            this._segmentPending = false;
            return;
        }

        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this._segmentPending = false;
            return;
        }

        this._segmentAbort = false;

        try {
            let endpoint, requestBody;

            if (imageInfo.uploaded) {
                endpoint = '/api/local/uploaded/sam2/predict';
                requestBody = {
                    upload_id: imageInfo.upload_id,
                    positive_points: this.positivePoints,
                    negative_points: this.negativePoints.length > 0 ? this.negativePoints : null,
                };
            } else {
                endpoint = '/api/sam2/predict';
                requestBody = {
                    image_id: imageInfo.id,
                    bbox: imageInfo.bbox,
                    geometry: imageInfo.geometry,
                    positive_points: this.positivePoints,
                    negative_points: this.negativePoints.length > 0 ? this.negativePoints : null,
                };
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            // If user clicked again while waiting, discard this result and re-run
            if (this._segmentAbort) {
                this._segmentPending = false;
                this._autoSegment();
                return;
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Segmentation failed' }));
                throw new Error(errorData.detail || `Segmentation failed`);
            }

            const result = await response.json();

            if (!result || !result.mask_id) {
                throw new Error('Invalid response from server');
            }

            // Show mask overlay as live preview
            this._hideCurrentPreview();

            // Hide target detection live overlay so it doesn't cover SAM2 result
            const td = this.platform.targetDetection;
            if (td?._currentLiveOverlayId) {
                td._hideLiveOverlay();
            }

            this.currentMaskId = result.mask_id;
            this.currentOverlayId = `sam2-preview-${result.mask_id}`;
            this._lastOverlayUrl = result.overlay_url;
            this._lastOverlayMeta = result.overlay_meta;

            const sam2Name = `SAM2 Preview`;

            if (this.isLocalMode()) {
                const li = this.platform.localImage;
                li?.showLocalAnalysisLayer(
                    this.currentOverlayId,
                    result.overlay_url,
                    result.overlay_meta?.width,
                    result.overlay_meta?.height,
                    sam2Name
                );
            } else if (window.mapManager) {
                window.mapManager.showAnalysisLayer(
                    this.currentOverlayId,
                    result.overlay_url,
                    sam2Name
                );
            }

            // Make overlay clickable for adding more points
            this._makeOverlayInteractive(this.currentOverlayId);

            this._updatePointCounts();
            this._setStatus(`Score: ${result.score.toFixed(3)} | ${result.pixel_count.toLocaleString()} px`);

        } catch (error) {
            console.error('SAM2 segmentation error:', error);
            this._setStatus('Error');
            this.platform.showNotification(`SAM2 failed: ${error.message}`, 'error');
        } finally {
            this._segmentPending = false;
        }
    }

    // ========== SAVE & NEXT ==========
    async saveAndNext() {
        if (!this.currentMaskId) return;

        const saveBtn = document.getElementById('sam2-save');
        if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = '...'; }

        try {
            const response = await fetch('/api/sam2/save-mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mask_id: this.currentMaskId })
            });

            if (!response.ok) throw new Error('Failed to save mask');

            const maskEntry = {
                mask_id: this.currentMaskId,
                visible: true,
                layerId: this.currentOverlayId,
                overlayUrl: this._lastOverlayUrl,
                overlayMeta: this._lastOverlayMeta
            };

            this.savedMasks.push(maskEntry);
            this._renderSavedMask(maskEntry);

            this.currentMaskId = null;
            this.currentOverlayId = null;
            this.clearCurrentPoints();
            this._setStatus('');

            this.platform.showNotification('Mask saved. Place new points for the next object.', 'success');

        } catch (error) {
            console.error('SAM2 save error:', error);
            this.platform.showNotification(`Failed to save mask: ${error.message}`, 'error');
        } finally {
            if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = '\uD83D\uDCBE Save'; }
        }
    }

    // ========== SAVED MASKS LIST ==========
    _renderSavedMask(maskEntry) {
        const item = document.querySelector('.analysis-item.sam2-option');
        const savedList = item?.querySelector('.sam2-saved-list');
        if (!savedList) return;

        if (savedList.querySelector(`[data-mask-id="${maskEntry.mask_id}"]`)) return;

        const idx = this.savedMasks.indexOf(maskEntry) + 1;
        const maskItem = document.createElement('div');
        maskItem.className = 'sam2-mask-item';
        maskItem.dataset.maskId = maskEntry.mask_id;
        maskItem.dataset.visible = maskEntry.visible ? 'true' : 'false';

        maskItem.innerHTML = `
            <div class="sam2-mask-header">
                <div class="sam2-mask-color" style="background: linear-gradient(135deg, #7b1fa2, #e91e63);"></div>
                <div class="sam2-mask-info">
                    <span class="sam2-mask-name">SAM2 Mask ${idx}</span>
                    <span class="sam2-mask-status">${maskEntry.visible ? 'Visible' : 'Hidden'}</span>
                </div>
                <div class="sam2-mask-actions">
                    <button class="sam2-mask-remove-btn" title="Remove">✕</button>
                </div>
            </div>
        `;

        savedList.appendChild(maskItem);

        // Click anywhere on the mask item to toggle visibility (like other layers)
        maskItem.addEventListener('click', (e) => {
            if (e.target.closest('.sam2-mask-remove-btn')) return;
            this._toggleMaskOverlay(maskEntry.mask_id);
        });

        maskItem.querySelector('.sam2-mask-remove-btn').onclick = (e) => {
            e.stopPropagation();
            this._removeMask(maskEntry.mask_id);
        };
    }

    _toggleMaskOverlay(maskId) {
        const entry = this.savedMasks.find(m => m.mask_id === maskId);
        if (!entry) return;
        const maskItem = document.querySelector(`.sam2-mask-item[data-mask-id="${maskId}"]`);

        if (entry.visible) {
            this._hideOverlayById(entry.layerId);
            entry.visible = false;
            if (maskItem) {
                maskItem.dataset.visible = 'false';
                maskItem.querySelector('.sam2-mask-status').textContent = 'Hidden';
            }
        } else {
            // Re-show via the proper show function so layer events are dispatched
            const idx = this.savedMasks.indexOf(entry) + 1;
            this._showOverlayById(entry.layerId, entry.overlayUrl, entry.overlayMeta);
            entry.visible = true;
            if (maskItem) {
                maskItem.dataset.visible = 'true';
                maskItem.querySelector('.sam2-mask-status').textContent = 'Visible';
            }
        }
    }

    async _removeMask(maskId) {
        const entry = this.savedMasks.find(m => m.mask_id === maskId);
        if (!entry) return;
        this._hideOverlayById(entry.layerId);
        try {
            await fetch(`/api/sam2/masks/${maskId}`, { method: 'DELETE' });
        } catch (e) {
            console.warn('Failed to delete mask from server:', e);
        }
        this.savedMasks = this.savedMasks.filter(m => m.mask_id !== maskId);
        const el = document.querySelector(`.sam2-mask-item[data-mask-id="${maskId}"]`);
        if (el) el.remove();
    }

    // ========== OVERLAY HELPERS ==========
    _hideOverlayById(layerId) {
        if (!layerId) return;
        if (this.isLocalMode()) {
            this.platform.localImage?.hideLocalAnalysisLayer(layerId);
        } else if (window.mapManager) {
            window.mapManager.hideAnalysisLayer(layerId);
        }
    }

    _showOverlayById(layerId, overlayUrl, meta) {
        if (this.isLocalMode()) {
            const li = this.platform.localImage;
            li?.showLocalAnalysisLayer(layerId, overlayUrl, meta?.width, meta?.height, 'SAM2');
        } else if (window.mapManager) {
            window.mapManager.showAnalysisLayer(layerId, overlayUrl, 'SAM2');
        }
    }

    _makeOverlayInteractive(layerId) {
        // Try to find the layer and add click handlers
        // In load-image mode, layers are now managed by mapManager
        let layer = window.mapManager?.analysisLayers?.[layerId];
        if (!layer && this.isLocalMode()) {
            const li = this.platform.localImage;
            layer = li?.localAnalysisOverlays?.[layerId];
        }
        if (layer) {
            layer.options.interactive = true;
            layer.on('click', (e) => {
                if (this.selectMode) {
                    L.DomEvent.stopPropagation(e);
                    this._addPoint(e.latlng, this.clickMode);
                }
            });
            layer.on('contextmenu', (e) => {
                if (this.selectMode) {
                    L.DomEvent.stopPropagation(e);
                    L.DomEvent.preventDefault(e);
                    this._addPoint(e.latlng, 'negative');
                }
            });
        }
    }

    _hideCurrentPreview() {
        this._hideOverlayById(this.currentOverlayId);
        this.currentMaskId = null;
        this.currentOverlayId = null;
        this._updatePointCounts();
    }

    // ========== CLEANUP ==========
    reset() {
        this.stopSelection();
        this.clearCurrentPoints();
        this.savedMasks.forEach(m => {
            this._hideOverlayById(m.layerId);
        });
        this.savedMasks = [];
        this.currentMaskId = null;
        this.currentOverlayId = null;

        const item = document.querySelector('.analysis-item.sam2-option');
        item?.querySelector('.sam2-ui')?.remove();
        item?.classList.remove('expanded', 'active');
        const infoEl = item?.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    cleanup() {
        this.reset();
    }
}
