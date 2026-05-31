/**
 * Flood Segmentation Controller
 *
 * Sentinel-1 VV + UNet++ flood/water mask. Mirrors the Mangrove
 * Segmentation surface (run -> probability map -> threshold -> binary mask).
 * Reuses the .ms-* DOM classes for styling but scopes all DOM queries to
 * this.itemEl so it never collides with a concurrently-open mangrove panel.
 */

class FloodSegmentationController {
    constructor(platformController) {
        this.platform = platformController;
        this.results = [];
        this.overlayLayers = {};
        this._slotStash = { A: null, B: null };
        this._originSlot = null;
        // At most one result can have eraser mode active across the
        // controller — flipping it on for B turns it off for A.
        this._activeEraserResultId = null;
        this._eraserHandlers = null;
        this._dragWasEnabled = undefined;
    }

    getImageInfo() {
        const imageId = this.platform.selectedImageId;
        if (!imageId) return null;
        return {
            id: imageId,
            bbox: this.platform.processedBbox || window.mapManager?.getCurrentBounds(),
            geometry: this.platform.processedGeometry || window.mapManager?.getCurrentGeoJSON()?.geometry,
        };
    }

    handleItemClick(item) {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification('Select a Sentinel-1 image first', 'warning');
            return;
        }

        this.itemEl = item || document.querySelector('.analysis-item.flood-segmentation-option');
        if (!this.itemEl) return;

        const hasUI = this.itemEl.querySelector('.ms-ui');
        if (hasUI) {
            this.hideUI();
        } else {
            this.showSetupUI();
        }
    }

    showSetupUI() {
        if (!this.itemEl) return;
        this.itemEl.querySelector('.ms-ui')?.remove();

        const infoEl = this.itemEl.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const ui = document.createElement('div');
        ui.className = 'ms-ui ms-setup';
        ui.innerHTML = `
            <div class="ms-row">
                <button data-fs-run class="ms-btn primary">Run Flood Segmentation</button>
            </div>
            <div class="ms-status" style="display:none;"></div>
            <div class="ms-results-list"></div>
        `;
        this.itemEl.appendChild(ui);
        this.itemEl.classList.add('expanded');

        this.results.forEach(r => this._renderResultItem(r));

        this.itemEl.querySelector('[data-fs-run]').onclick = (e) => {
            e.stopPropagation();
            this.runSegmentation();
        };
    }

    hideUI() {
        if (!this.itemEl) return;
        this.itemEl.querySelector('.ms-ui')?.remove();
        this.itemEl.classList.remove('expanded');
        const infoEl = this.itemEl.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    async runSegmentation() {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification('No image available', 'warning');
            return;
        }

        this._originSlot = this.platform.currentSlot;
        const runBtn = this.itemEl?.querySelector('[data-fs-run]');
        const statusEl = this.itemEl?.querySelector('.ms-status');

        if (runBtn) {
            runBtn.disabled = true;
            runBtn.textContent = 'Running...';
        }
        if (statusEl) {
            statusEl.style.display = 'block';
            statusEl.textContent = 'Running flood segmentation on GPU...';
            statusEl.className = 'ms-status running';
        }

        const run = async () => {
            const response = await fetch('/api/flood-segmentation/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                body: JSON.stringify({
                    image_id: imageInfo.id,
                    bbox: imageInfo.bbox,
                    geometry: imageInfo.geometry,
                }),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${response.status}`);
            }
            const data = await response.json();
            this._finishJob({ status: 'completed', result: data });
            if (data && data.segmentation_id) {
                await this._awaitOverlayPaint(`fs-${data.segmentation_id}`);
            }
        };

        try {
            if (this.platform && typeof this.platform.withLoading === 'function') {
                await this.platform.withLoading('Running flood segmentation...', run);
            } else {
                await run();
            }
        } catch (err) {
            console.error('Flood segmentation failed:', err);
            this._finishJob({ status: 'error', error: err.message });
        }
    }

    _awaitOverlayPaint(layerId) {
        return new Promise(resolve => {
            const layer = window.mapManager?.analysisLayers?.[layerId];
            if (!layer || typeof layer.once !== 'function') return resolve();
            const el = (typeof layer.getElement === 'function') ? layer.getElement() : null;
            if (el && el.complete && el.naturalHeight > 0) return resolve();
            let settled = false;
            const finish = () => { if (settled) return; settled = true; resolve(); };
            const t = setTimeout(finish, 60000);
            layer.once('load',  () => { clearTimeout(t); finish(); });
            layer.once('error', () => { clearTimeout(t); finish(); });
        });
    }

    _finishJob(status) {
        const runBtn = this.itemEl?.querySelector('[data-fs-run]');
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.textContent = 'Run Flood Segmentation';
        }
        const statusEl = this.itemEl?.querySelector('.ms-status');

        if (status.status === 'error') {
            if (statusEl) {
                statusEl.textContent = `Error: ${status.error || 'segmentation failed'}`;
                statusEl.className = 'ms-status error';
            }
            this.platform.showNotification(`Flood segmentation failed: ${status.error || 'unknown'}`, 'error');
            return;
        }
        const data = status.result;
        if (!data) {
            if (statusEl) {
                statusEl.textContent = 'Completed but result missing.';
                statusEl.className = 'ms-status error';
            }
            return;
        }

        const resultEntry = {
            id: data.segmentation_id,
            min_val: data.min_val,
            max_val: data.max_val,
            detection_result: {
                preview_url: data.preview_url,
                overlay_url: data.overlay_url,
                colormap: data.colormap,
            },
            mask_result: null,
            showingMask: false,
            visible: false,
        };

        const origin = this._originSlot;
        this._originSlot = null;
        if (origin && origin !== this.platform.currentSlot) {
            this.platform.showNotification(
                `Flood segmentation on Time ${origin} dropped — you switched slots mid-run.`,
                'warning'
            );
            return;
        }

        resultEntry.slotId = this.platform.currentSlot;
        this.results.push(resultEntry);
        this._renderResultItem(resultEntry);
        this._showOverlay(resultEntry.id, resultEntry.detection_result);

        if (typeof this.platform.registerSlotAnalysis === 'function') {
            this.platform.registerSlotAnalysis({
                id: resultEntry.id,
                type: 'segmentation',
                name: `Flood Seg · ${String(resultEntry.id).slice(-6)}`,
                hasMask: false,
            });
        }

        if (statusEl) {
            statusEl.textContent = 'Flood segmentation complete!';
            statusEl.className = 'ms-status success';
        }
        this.platform.showNotification('Flood segmentation complete', 'success');
    }

    _renderResultItem(resultEntry) {
        const resultsList = this.itemEl?.querySelector('.ms-results-list');
        if (!resultsList) return;
        if (resultsList.querySelector(`[data-result-id="${resultEntry.id}"]`)) return;

        const r = resultEntry;
        const minVal = r.min_val;
        const maxVal = r.max_val;

        const resultItem = document.createElement('div');
        resultItem.className = 'ms-result-item';
        resultItem.dataset.resultId = r.id;
        resultItem.dataset.visible = 'false';

        resultItem.innerHTML = `
            <div class="ms-result-header">
                <img class="ms-result-thumb" src="${r.detection_result?.preview_url || ''}" alt="Flood Segmentation" />
                <div class="ms-result-info">
                    <span class="ms-result-name">Water Probability</span>
                    <span class="ms-result-status">Click to toggle overlay</span>
                </div>
                <div class="ms-result-actions">
                    <button class="ms-result-remove-btn" title="Remove">&times;</button>
                </div>
            </div>
            <div class="ms-result-colorbar">
                <div class="colorbar-with-threshold">
                    <div class="colorbar-track ms-track">
                        <div class="colorbar-gradient ms-gradient"></div>
                        <div class="colorbar-selection"></div>
                        <div class="colorbar-handle min-handle"></div>
                        <div class="colorbar-handle max-handle"></div>
                    </div>
                </div>
                <div class="colorbar-values">
                    <input type="number" class="colorbar-min-input ms-result-min" value="${minVal.toFixed(3)}" step="0.001">
                    <div class="colorbar-buttons">
                        <button class="ms-result-apply-btn colorbar-apply-btn">Apply</button>
                        <button class="ms-result-cancel-btn colorbar-cancel-btn">Cancel</button>
                    </div>
                    <input type="number" class="colorbar-max-input ms-result-max" value="${maxVal.toFixed(3)}" step="0.001">
                </div>
            </div>
            <div class="ms-result-stats" style="display:none;">
                <span class="ms-result-pixels"></span>
                <span class="ms-result-pct"></span>
            </div>
            <div class="ms-result-eraser-row" style="display:none; align-items:center; gap:6px; padding:6px 8px; border-top:1px solid #eee; background:#fafafa;">
                <button class="ms-result-eraser-btn colorbar-apply-btn" type="button"
                        style="background:#1976d2; color:white;">🧽 Eraser</button>
                <button class="ms-result-reset-btn colorbar-cancel-btn" type="button"
                        title="Bring back all erased blobs">↺ Reset</button>
                <span class="ms-result-eraser-stats" style="margin-left:auto; font-size:11px; color:#666;"></span>
            </div>
        `;

        resultsList.appendChild(resultItem);

        resultItem.querySelector('.ms-result-header').onclick = (e) => {
            if (e.target.closest('.ms-result-actions')) return;
            e.stopPropagation();
            this._toggleOverlay(r.id);
        };
        resultItem.querySelector('.ms-result-remove-btn').onclick = (e) => {
            e.stopPropagation();
            this._removeResult(r.id);
        };

        this._setupResultThreshold(resultItem, r);
        this._setupEraserRow(resultItem, r);
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
        const minInput = resultItem.querySelector('.ms-result-min');
        const maxInput = resultItem.querySelector('.ms-result-max');
        const applyBtn = resultItem.querySelector('.ms-result-apply-btn');
        const cancelBtn = resultItem.querySelector('.ms-result-cancel-btn');

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

        applyBtn.onclick = async (e) => {
            e.stopPropagation();
            await this._applyThreshold(resultEntry, currentMin, currentMax, resultItem);
        };
        cancelBtn.onclick = (e) => {
            e.stopPropagation();
            currentMin = minVal;
            currentMax = maxVal;
            minInput.value = minVal.toFixed(3);
            maxInput.value = maxVal.toFixed(3);
            updateUI();
            resultEntry.showingMask = false;
            this._showOverlay(resultEntry.id, resultEntry.detection_result);
            resultItem.querySelector('.ms-result-stats').style.display = 'none';
            resultItem.querySelector('.ms-result-thumb').src = resultEntry.detection_result?.preview_url || '';
            resultItem.querySelector('.ms-result-status').textContent = 'Probability map';
            // Going back to the probability map invalidates the binary mask
            // on the server (apply-threshold no longer reflected here);
            // disable eraser controls and hide the row.
            if (this._activeEraserResultId === resultEntry.id) {
                this._toggleEraser(resultEntry, resultItem, /*force*/ false);
            }
            const eraserRow = resultItem.querySelector('.ms-result-eraser-row');
            if (eraserRow) eraserRow.style.display = 'none';
        };

        resultItem.querySelector('.ms-result-colorbar').addEventListener('click', (e) => e.stopPropagation());
    }

    _setupEraserRow(resultItem, resultEntry) {
        const eraserBtn = resultItem.querySelector('.ms-result-eraser-btn');
        const resetBtn = resultItem.querySelector('.ms-result-reset-btn');
        const row = resultItem.querySelector('.ms-result-eraser-row');
        if (!eraserBtn || !resetBtn || !row) return;

        row.addEventListener('click', (e) => e.stopPropagation());

        eraserBtn.onclick = (e) => {
            e.stopPropagation();
            this._toggleEraser(resultEntry, resultItem);
        };
        resetBtn.onclick = async (e) => {
            e.stopPropagation();
            await this._resetExclusions(resultEntry, resultItem);
        };
    }

    /** Show the eraser row once a binary mask is on screen. */
    _showEraserRow(resultItem) {
        const row = resultItem?.querySelector('.ms-result-eraser-row');
        if (row) row.style.display = 'flex';
    }

    /**
     * Toggle eraser mode for a result. When ON: map panning is disabled and
     * mousedown/mousemove/mouseup draw a temporary L.Rectangle. On mouseup
     * the rectangle's lat/lng bounds are POSTed to /erase-region, and every
     * connected component touching that box is removed. Only one result can
     * be in eraser mode at a time across the controller.
     */
    _toggleEraser(resultEntry, resultItem, forceState) {
        const wasActive = this._activeEraserResultId === resultEntry.id;
        const willBeActive = (typeof forceState === 'boolean')
            ? forceState
            : !wasActive;

        // Always tear down any existing handlers first.
        this._teardownEraserHandlers();

        // De-style any previously active button.
        if (this._activeEraserResultId && this._activeEraserResultId !== resultEntry.id) {
            const prev = this.itemEl?.querySelector(
                `.ms-result-item[data-result-id="${this._activeEraserResultId}"] .ms-result-eraser-btn`
            );
            if (prev) {
                prev.classList.remove('active');
                prev.textContent = '🧽 Eraser';
            }
        }

        const btn = resultItem?.querySelector('.ms-result-eraser-btn');

        if (!willBeActive) {
            this._activeEraserResultId = null;
            if (btn) {
                btn.classList.remove('active');
                btn.textContent = '🧽 Eraser';
            }
            return;
        }

        // Enable eraser for this result.
        this._activeEraserResultId = resultEntry.id;
        if (btn) {
            btn.classList.add('active');
            btn.textContent = '🧽 Eraser (ON)';
        }
        this._installEraserHandlers(resultEntry, resultItem);
        this.platform.showNotification(
            'Eraser ON — drag a box on the map to remove water blobs inside it.',
            'info'
        );
    }

    _installEraserHandlers(resultEntry, resultItem) {
        const map = window.mapManager?.map;
        if (!map) return;

        // Disable map panning while erasing — left-mousedrag should draw a
        // rectangle, not pan. Restored when teardown runs.
        if (map.dragging?.enable) {
            this._dragWasEnabled = map.dragging.enabled();
            map.dragging.disable();
        }
        if (map.boxZoom?.disable) map.boxZoom.disable();
        document.body.style.cursor = 'crosshair';

        let startLatLng = null;
        let dragRect = null;

        const onDown = (e) => {
            if (!e || !e.latlng) return;
            startLatLng = e.latlng;
            dragRect = L.rectangle(
                [[startLatLng.lat, startLatLng.lng], [startLatLng.lat, startLatLng.lng]],
                {
                    color: '#1976d2',
                    weight: 2,
                    fillColor: '#1976d2',
                    fillOpacity: 0.18,
                    interactive: false,
                }
            ).addTo(map);
        };
        const onMove = (e) => {
            if (!startLatLng || !dragRect || !e || !e.latlng) return;
            dragRect.setBounds([
                [startLatLng.lat, startLatLng.lng],
                [e.latlng.lat, e.latlng.lng],
            ]);
        };
        const onUp = async (e) => {
            if (!startLatLng || !dragRect) return;
            const bounds = dragRect.getBounds();
            map.removeLayer(dragRect);
            dragRect = null;
            const start = startLatLng;
            startLatLng = null;

            const latDelta = Math.abs(bounds.getNorth() - bounds.getSouth());
            const lngDelta = Math.abs(bounds.getEast() - bounds.getWest());
            // Ignore micro-drags (treat as accidental click).
            const minDeg = 1e-5;
            if (latDelta < minDeg || lngDelta < minDeg) {
                return;
            }
            await this._postEraseRegion(resultEntry, resultItem, bounds);
        };

        map.on('mousedown', onDown);
        map.on('mousemove', onMove);
        map.on('mouseup', onUp);

        // Stash so we can detach exactly these handlers later.
        this._eraserHandlers = { map, onDown, onMove, onUp };
    }

    _teardownEraserHandlers() {
        const h = this._eraserHandlers;
        if (h && h.map) {
            h.map.off('mousedown', h.onDown);
            h.map.off('mousemove', h.onMove);
            h.map.off('mouseup', h.onUp);
            if (this._dragWasEnabled !== false && h.map.dragging?.enable) {
                h.map.dragging.enable();
            }
            if (h.map.boxZoom?.enable) h.map.boxZoom.enable();
        }
        this._eraserHandlers = null;
        this._dragWasEnabled = undefined;
        document.body.style.cursor = '';
    }

    async _postEraseRegion(resultEntry, resultItem, bounds) {
        const body = {
            segmentation_id: resultEntry.id,
            lat_min: bounds.getSouth(),
            lng_min: bounds.getWest(),
            lat_max: bounds.getNorth(),
            lng_max: bounds.getEast(),
        };
        try {
            const response = await fetch('/api/flood-segmentation/erase-region', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                this.platform.showNotification(err.detail || 'Erase failed', 'warning');
                return;
            }
            const result = await response.json();
            resultEntry.mask_result = result.mask_result;
            resultEntry.showingMask = true;

            this._showOverlay(resultEntry.id, result.mask_result, true);
            this._refreshResultStats(resultItem, result);
            this._updateEraserStats(resultItem, result);

            const blobs = result.removed_blobs || 0;
            const px = result.removed_pixels || 0;
            this.platform.showNotification(
                `Erased ${blobs} blob${blobs === 1 ? '' : 's'} ` +
                `(${px.toLocaleString()} px). Drag another box or toggle Eraser off.`,
                'success'
            );
        } catch (err) {
            console.error('Erase-region error:', err);
            this.platform.showNotification('Failed to erase region', 'error');
        }
    }

    async _resetExclusions(resultEntry, resultItem) {
        try {
            const response = await fetch('/api/flood-segmentation/reset-exclusions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ segmentation_id: resultEntry.id }),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                this.platform.showNotification(err.detail || 'Reset failed', 'warning');
                return;
            }
            const result = await response.json();
            resultEntry.mask_result = result.mask_result;
            resultEntry.showingMask = true;

            this._showOverlay(resultEntry.id, result.mask_result, true);
            this._refreshResultStats(resultItem, result);
            this._updateEraserStats(resultItem, result);

            this.platform.showNotification('All erased blobs restored.', 'success');
        } catch (err) {
            console.error('Reset-exclusions error:', err);
            this.platform.showNotification('Failed to reset exclusions', 'error');
        }
    }

    /** Update the colorbar stats row (detected_pixels / detection_percentage). */
    _refreshResultStats(resultItem, result) {
        if (!resultItem) return;
        const stats = resultItem.querySelector('.ms-result-stats');
        if (stats) {
            stats.style.display = 'flex';
            stats.querySelector('.ms-result-pixels').textContent =
                `${(result.detected_pixels || 0).toLocaleString()} px`;
            stats.querySelector('.ms-result-pct').textContent =
                `${result.detection_percentage || 0}%`;
        }
        if (result.mask_result?.preview_url) {
            resultItem.querySelector('.ms-result-thumb').src = result.mask_result.preview_url;
        }
    }

    /** Update the eraser-row counter ("X blobs / Y px excluded"). */
    _updateEraserStats(resultItem, result) {
        const label = resultItem?.querySelector('.ms-result-eraser-stats');
        if (!label) return;
        const blobs = result.excluded_blobs || 0;
        const px = result.total_excluded_pixels || 0;
        if (blobs === 0) {
            label.textContent = '';
        } else {
            label.textContent = `${blobs} blob${blobs === 1 ? '' : 's'} / ${px.toLocaleString()} px excluded`;
        }
    }

    async _applyThreshold(resultEntry, min, max, resultItem) {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) return;
        try {
            const response = await fetch('/api/flood-segmentation/apply-threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    segmentation_id: resultEntry.id,
                    min_threshold: min,
                    max_threshold: max,
                    bbox: imageInfo.bbox,
                }),
            });
            if (!response.ok) throw new Error('Failed to apply threshold');
            const result = await response.json();
            resultEntry.mask_result = result.mask_result;
            resultEntry.showingMask = true;

            if (typeof this.platform.registerSlotAnalysis === 'function') {
                this.platform.registerSlotAnalysis({
                    id: resultEntry.id,
                    type: 'segmentation',
                    name: `Flood Seg · ${String(resultEntry.id).slice(-6)}`,
                    hasMask: true,
                });
            }

            this._showOverlay(resultEntry.id, result.mask_result, true);

            if (resultItem) {
                this._refreshResultStats(resultItem, result);
                resultItem.querySelector('.ms-result-status').textContent =
                    `Mask (${min.toFixed(3)} - ${max.toFixed(3)})`;
                // Binary mask is now on screen — eraser controls are meaningful.
                this._showEraserRow(resultItem);
                this._updateEraserStats(resultItem, result);
            }

            this.platform.showNotification(`Flood: ${result.detection_percentage}% detected`, 'success');
        } catch (error) {
            console.error('Threshold error:', error);
            this.platform.showNotification('Failed to apply threshold', 'error');
        }
    }

    _showOverlay(resultId, overlayData, isBinary = false) {
        if (!overlayData?.overlay_url) return;
        this._hideOverlay(resultId);
        const layerId = `fs-${resultId}`;
        if (window.mapManager) {
            window.mapManager.showAnalysisLayer(layerId, overlayData.overlay_url, 'Flood Segmentation', null, isBinary);
            window.mapManager.outlineAOI();
        }
        this.overlayLayers[resultId] = layerId;

        const resultItem = this.itemEl?.querySelector(`.ms-result-item[data-result-id="${resultId}"]`);
        if (resultItem) {
            resultItem.dataset.visible = 'true';
            const entry = this.results.find(r => r.id === resultId);
            if (entry) entry.visible = true;
        }
    }

    _hideOverlay(resultId) {
        const layerId = this.overlayLayers[resultId];
        if (!layerId) return;
        if (window.mapManager) {
            window.mapManager.hideAnalysisLayer(layerId);
        }
        delete this.overlayLayers[resultId];
        const resultItem = this.itemEl?.querySelector(`.ms-result-item[data-result-id="${resultId}"]`);
        if (resultItem) {
            resultItem.dataset.visible = 'false';
            const entry = this.results.find(r => r.id === resultId);
            if (entry) entry.visible = false;
        }
    }

    _toggleOverlay(resultId) {
        const entry = this.results.find(r => r.id === resultId);
        if (!entry) return;
        const resultItem = this.itemEl?.querySelector(`.ms-result-item[data-result-id="${resultId}"]`);
        const isVisible = resultItem?.dataset.visible === 'true';
        if (isVisible) {
            this._hideOverlay(resultId);
            if (resultItem) {
                resultItem.querySelector('.ms-result-status').textContent = 'Click to toggle overlay';
            }
        } else {
            const overlayData = entry.showingMask ? entry.mask_result : entry.detection_result;
            this._showOverlay(resultId, overlayData, !!entry.showingMask);
            if (resultItem) {
                resultItem.querySelector('.ms-result-status').textContent = 'Overlay active — click to hide';
            }
        }
    }

    _removeResult(resultId) {
        // If this result owns the active eraser, tear down the map listeners
        // before deleting the row.
        if (this._activeEraserResultId === resultId) {
            this._teardownEraserHandlers();
            this._activeEraserResultId = null;
        }
        this._hideOverlay(resultId);
        this.results = this.results.filter(r => r.id !== resultId);
        const resultItem = this.itemEl?.querySelector(`.ms-result-item[data-result-id="${resultId}"]`);
        if (resultItem) resultItem.remove();
    }

    _freezeToSlot(slotId) {
        if (slotId !== 'A' && slotId !== 'B') return;
        this._slotStash[slotId] = {
            results: this.results.slice(),
            overlayLayers: { ...this.overlayLayers },
        };
    }

    _thawFromSlot(slotId) {
        const stash = (slotId === 'A' || slotId === 'B') ? this._slotStash[slotId] : null;
        if (stash) {
            this.results = Array.isArray(stash.results) ? stash.results.slice() : [];
            this.overlayLayers = { ...(stash.overlayLayers || {}) };
            this._slotStash[slotId] = null;
        } else {
            this.results = [];
            this.overlayLayers = {};
        }
    }

    handleSlotChange(oldSlot, newSlot) {
        if (oldSlot === newSlot) return;
        this._freezeToSlot(oldSlot);
        this._thawFromSlot(newSlot);
        const list = document.querySelector(
            '.analysis-item.flood-segmentation-option .ms-results-list');
        if (list) {
            list.innerHTML = '';
            this.results.forEach(r => this._renderResultItem(r));
        }
    }
}

if (typeof window !== 'undefined') {
    window.FloodSegmentationController = FloodSegmentationController;
}
