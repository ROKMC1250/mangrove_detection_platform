/**
 * SAM3 Segmentation Controller
 *
 * Two modes share the same panel:
 *  - Point: positive/negative click prompts (single best mask per run, SAM2-style)
 *  - Text:  open-vocabulary noun phrase via Promptable Concept Segmentation
 *           (one prompt → every matching instance, each as its own coloured layer)
 *
 * Endpoints
 *   Cloud:  /api/sam3/{predict,text-predict,save-mask,masks/{id}}
 *   Local:  /api/local/uploaded/sam3/{predict,text-predict}
 */

class SAM3Controller {
    constructor(platformController) {
        this.platform = platformController;

        // Mode toggle
        this.mode = 'point';  // 'point' | 'text'

        // ===== Point mode state =====
        this.selectMode = false;
        this.clickMode = 'positive';
        this.positivePoints = [];
        this.negativePoints = [];
        this.markers = [];
        this.clickHistory = [];
        this.currentMaskId = null;
        this.currentOverlayId = null;
        this._lastOverlayUrl = null;
        this._lastOverlayMeta = null;

        // ===== Text mode state =====
        this.textPrompt = '';
        this.textThreshold = 0.5;
        this.textInstances = [];   // [{mask_id, layerId, overlayUrl, overlayMeta, score, color_hex, name}]
        this.textHighlightId = null;

        // Saved masks (both modes) — toggled visible in the right list
        this.savedMasks = [];

        // Auto-segment state (point mode)
        this._segmentPending = false;
        this._segmentAbort = null;
        this._autoSegmentTimer = null;

        // Per-slot stash (Time A / Time B)
        this._slotStash = { A: null, B: null };

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
        return window.mapManager?.map;
    }

    getImageInfo() {
        if (this.isUploadedMode()) {
            const li = this.platform.localImage;
            return { local: true, uploaded: true, upload_id: li.uploadId };
        }
        const imageId = this.platform.selectedImageId;
        if (!imageId) return null;
        return {
            id: imageId,
            bbox: this.platform.processedBbox || window.mapManager?.getCurrentBounds(),
            geometry: this.platform.processedGeometry || window.mapManager?.getCurrentGeoJSON()?.geometry,
        };
    }

    handleItemClick() {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification(
                this.isLocalMode() ? 'Load an image first' : 'Process an image first',
                'warning'
            );
            return;
        }

        const item = document.querySelector('.analysis-item.sam3-option');
        const hasUI = item?.querySelector('.sam3-ui');
        if (hasUI) this.hideUI(); else this.showSetupUI();
    }

    // ===========================================================================
    // SETUP UI
    // ===========================================================================
    showSetupUI() {
        const item = document.querySelector('.analysis-item.sam3-option');
        if (!item) return;

        item.querySelector('.sam3-ui')?.remove();
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const ui = document.createElement('div');
        ui.className = 'sam3-ui sam3-setup';
        ui.innerHTML = `
            <div class="sam3-mode-toggle" style="display:flex; gap:4px; margin-bottom:6px;">
                <button id="sam3-mode-point" class="sam3-btn ${this.mode === 'point' ? 'active' : ''}" title="Click prompts">📍 Point</button>
                <button id="sam3-mode-text"  class="sam3-btn ${this.mode === 'text'  ? 'active' : ''}" title="Open-vocabulary text prompt (PCS)">🅰 Text</button>
            </div>
            <div id="sam3-point-pane"  class="sam3-pane" style="${this.mode === 'point' ? '' : 'display:none;'}"></div>
            <div id="sam3-text-pane"   class="sam3-pane" style="${this.mode === 'text'  ? '' : 'display:none;'}"></div>
            <div class="sam3-saved-list"></div>
        `;
        item.appendChild(ui);
        item.classList.add('expanded');

        this._renderPointPane();
        this._renderTextPane();
        this.savedMasks.forEach((m) => this._renderSavedMask(m));

        document.getElementById('sam3-mode-point').onclick = (e) => {
            e.stopPropagation();
            this._switchMode('point');
        };
        document.getElementById('sam3-mode-text').onclick = (e) => {
            e.stopPropagation();
            this._switchMode('text');
        };
    }

    hideUI() {
        this.stopSelection();
        const item = document.querySelector('.analysis-item.sam3-option');
        if (!item) return;
        item.querySelector('.sam3-ui')?.remove();
        item.classList.remove('expanded');
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    _switchMode(mode) {
        if (mode === this.mode) return;
        this.mode = mode;

        document.getElementById('sam3-mode-point')?.classList.toggle('active', mode === 'point');
        document.getElementById('sam3-mode-text') ?.classList.toggle('active', mode === 'text');

        const pp = document.getElementById('sam3-point-pane');
        const tp = document.getElementById('sam3-text-pane');
        if (pp) pp.style.display = (mode === 'point') ? '' : 'none';
        if (tp) tp.style.display = (mode === 'text')  ? '' : 'none';

        if (mode !== 'point' && this.selectMode) this.stopSelection();
    }

    // ===========================================================================
    // POINT-MODE PANE
    // ===========================================================================
    _renderPointPane() {
        const pane = document.getElementById('sam3-point-pane');
        if (!pane) return;
        pane.innerHTML = `
            <div class="sam3-row">
                <button id="sam3-pick" class="sam3-btn ${this.selectMode ? 'active' : ''}" title="Left-click: positive, Right-click: negative">📍 <span id="sam3-count">${this.positivePoints.length + this.negativePoints.length}</span></button>
                <button id="sam3-mode-toggle" class="sam3-btn sam3-mode-positive" title="Current: Positive (click to switch)">+</button>
                <button id="sam3-undo" class="sam3-btn" title="Undo last point" ${this.clickHistory.length === 0 ? 'disabled' : ''}>↩</button>
                <button id="sam3-clear" class="sam3-btn" title="Clear all points" ${this.clickHistory.length === 0 ? 'disabled' : ''}>🗑</button>
                <button id="sam3-save" class="sam3-btn save" ${!this.currentMaskId ? 'disabled' : ''} title="Save current mask">💾 Save</button>
            </div>
            <div class="sam3-point-info" style="font-size:11px; color:#888; padding:2px 4px;">
                <span id="sam3-pos-count" style="color:#22cc44;">+${this.positivePoints.length}</span>
                <span id="sam3-neg-count" style="color:#ee3344; margin-left:6px;">-${this.negativePoints.length}</span>
                <span id="sam3-status" style="margin-left:8px; color:#E7344C;"></span>
            </div>
        `;

        document.getElementById('sam3-pick').onclick = (e) => {
            e.stopPropagation();
            this.toggleSelection();
        };
        document.getElementById('sam3-mode-toggle').onclick = (e) => {
            e.stopPropagation();
            this.clickMode = this.clickMode === 'positive' ? 'negative' : 'positive';
            const btn = document.getElementById('sam3-mode-toggle');
            if (this.clickMode === 'positive') {
                btn.textContent = '+';
                btn.className = 'sam3-btn sam3-mode-positive';
            } else {
                btn.textContent = '−';
                btn.className = 'sam3-btn sam3-mode-negative';
            }
        };
        document.getElementById('sam3-undo').onclick  = (e) => { e.stopPropagation(); this.undoLastPoint(); };
        document.getElementById('sam3-clear').onclick = (e) => { e.stopPropagation(); this.clearCurrentPoints(); };
        document.getElementById('sam3-save').onclick  = (e) => { e.stopPropagation(); this.saveAndNext(); };
    }

    // ===========================================================================
    // TEXT-MODE PANE
    // ===========================================================================
    _renderTextPane() {
        const pane = document.getElementById('sam3-text-pane');
        if (!pane) return;
        const tpl = (this.textPrompt || '').replace(/"/g, '&quot;');
        pane.innerHTML = `
            <div class="sam3-text-row" style="display:flex; gap:4px; align-items:center;">
                <input id="sam3-text-input" class="sam3-text-input" type="text"
                       placeholder="e.g. mangrove, boat, building"
                       value="${tpl}"
                       style="flex:1; padding:4px 6px; font-size:12px;" />
                <button id="sam3-text-run" class="sam3-btn save" title="Run PCS">🔍 Run</button>
            </div>
            <div class="sam3-thresh-row" style="display:flex; align-items:center; gap:6px; margin-top:4px; font-size:11px; color:#888;">
                <label for="sam3-thresh">conf ≥</label>
                <input id="sam3-thresh" type="range" min="0.3" max="0.95" step="0.05"
                       value="${this.textThreshold}" style="flex:1;" />
                <span id="sam3-thresh-val">${this.textThreshold.toFixed(2)}</span>
                <span id="sam3-text-status" style="margin-left:6px; color:#E7344C;"></span>
            </div>
            <div id="sam3-text-instances" class="sam3-text-instances" style="margin-top:6px;"></div>
        `;

        const threshEl = document.getElementById('sam3-thresh');
        const threshVal = document.getElementById('sam3-thresh-val');
        threshEl.addEventListener('input', () => {
            this.textThreshold = parseFloat(threshEl.value);
            threshVal.textContent = this.textThreshold.toFixed(2);
        });

        document.getElementById('sam3-text-run').onclick = (e) => {
            e.stopPropagation();
            const input = document.getElementById('sam3-text-input');
            this.textPrompt = (input?.value || '').trim();
            if (!this.textPrompt) {
                this.platform.showNotification('Type a noun phrase first', 'warning');
                return;
            }
            this._runTextSegmentation();
        };

        document.getElementById('sam3-text-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                document.getElementById('sam3-text-run').click();
            }
        });

        this._renderTextInstances();
    }

    _setTextStatus(text) {
        const el = document.getElementById('sam3-text-status');
        if (el) el.textContent = text;
    }

    // ===========================================================================
    // POINT SELECTION (mostly mirrors SAM2's flow)
    // ===========================================================================
    toggleSelection() {
        if (this.selectMode) this.stopSelection();
        else this.startSelection();
    }

    startSelection() {
        const td = this.platform.targetDetection;
        if (td?.targetSelectMode) td.stopTargetSelection();

        this.selectMode = true;
        document.getElementById('sam3-pick')?.classList.add('active');
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
        document.getElementById('sam3-pick')?.classList.remove('active');
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
        const rect = map.getContainer().getBoundingClientRect();
        const latlng = map.containerPointToLatLng(L.point(e.clientX - rect.left, e.clientY - rect.top));
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

        if (isPositive) { this.positivePoints.push(pointData); this.clickHistory.push('pos'); }
        else            { this.negativePoints.push(pointData); this.clickHistory.push('neg'); }

        const map = this.getMap();
        if (map && window.L) {
            const fillColor = isPositive ? '#22cc44' : '#ee3344';
            const symbol    = isPositive ? '+' : '−';

            const marker = L.circleMarker([latlng.lat, latlng.lng], {
                radius: 10, fillColor, color: '#ffffff',
                weight: 3, opacity: 1, fillOpacity: 1,
            }).addTo(map);

            const label = L.marker([latlng.lat, latlng.lng], {
                icon: L.divIcon({
                    className: 'sam3-marker-label',
                    html: `<span>${symbol}</span>`,
                    iconSize: [20, 20], iconAnchor: [10, 10],
                }),
            }).addTo(map);

            this.markers.push({ marker, label, mode: isPositive ? 'pos' : 'neg' });
        }

        this._updatePointCounts();
        this._autoSegment();
    }

    undoLastPoint() {
        if (this.clickHistory.length === 0) return;
        const last = this.clickHistory.pop();
        if (last === 'pos') this.positivePoints.pop();
        else                this.negativePoints.pop();
        const lastMarker = this.markers.pop();
        const map = this.getMap();
        if (lastMarker) {
            if (lastMarker.marker) map?.removeLayer(lastMarker.marker);
            if (lastMarker.label)  map?.removeLayer(lastMarker.label);
        }
        this._updatePointCounts();
        if (this.positivePoints.length > 0) this._autoSegment();
        else this._hideCurrentPreview();
    }

    clearCurrentPoints() {
        const map = this.getMap();
        this.markers.forEach((item) => {
            if (item.marker) map?.removeLayer(item.marker);
            if (item.label)  map?.removeLayer(item.label);
        });
        this.markers = [];
        this.positivePoints = [];
        this.negativePoints = [];
        this.clickHistory = [];
        this._updatePointCounts();
        this._hideCurrentPreview();
    }

    _updatePointCounts() {
        const countEl = document.getElementById('sam3-count');
        if (countEl) countEl.textContent = this.positivePoints.length + this.negativePoints.length;
        const posEl = document.getElementById('sam3-pos-count');
        if (posEl) posEl.textContent = `+${this.positivePoints.length}`;
        const negEl = document.getElementById('sam3-neg-count');
        if (negEl) negEl.textContent = `-${this.negativePoints.length}`;
        const undoBtn = document.getElementById('sam3-undo');
        if (undoBtn) undoBtn.disabled = this.clickHistory.length === 0;
        const clearBtn = document.getElementById('sam3-clear');
        if (clearBtn) clearBtn.disabled = this.clickHistory.length === 0;
        const saveBtn = document.getElementById('sam3-save');
        if (saveBtn) saveBtn.disabled = !this.currentMaskId;
    }

    _setStatus(text) {
        const el = document.getElementById('sam3-status');
        if (el) el.textContent = text;
    }

    // ===========================================================================
    // POINT-MODE SEGMENTATION (single best mask per run)
    // ===========================================================================
    _autoSegment() {
        if (this.positivePoints.length === 0) return;
        if (this._segmentPending) this._segmentAbort = true;
        this._segmentPending = true;
        this._setStatus('Segmenting...');
        clearTimeout(this._autoSegmentTimer);
        this._autoSegmentTimer = setTimeout(() => this._runSegmentation(), 100);
    }

    async _runSegmentation() {
        if (this.positivePoints.length === 0) { this._segmentPending = false; return; }
        const imageInfo = this.getImageInfo();
        if (!imageInfo) { this._segmentPending = false; return; }

        this._segmentAbort = false;

        const run = async () => {
            let endpoint, requestBody;
            if (imageInfo.uploaded) {
                endpoint = '/api/local/uploaded/sam3/predict';
                requestBody = {
                    upload_id: imageInfo.upload_id,
                    positive_points: this.positivePoints,
                    negative_points: this.negativePoints.length > 0 ? this.negativePoints : null,
                };
            } else {
                endpoint = '/api/sam3/predict';
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
                body: JSON.stringify(requestBody),
            });

            if (this._segmentAbort) {
                this._segmentPending = false;
                this._autoSegment();
                return;
            }
            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: 'Segmentation failed' }));
                throw new Error(err.detail || 'Segmentation failed');
            }
            const result = await response.json();
            if (!result || !result.mask_id) throw new Error('Invalid response from server');

            this._hideCurrentPreview();

            const td = this.platform.targetDetection;
            if (td?._currentLiveOverlayId) td._hideLiveOverlay();

            this.currentMaskId = result.mask_id;
            this.currentOverlayId = `sam3-preview-${result.mask_id}`;
            this._lastOverlayUrl = result.overlay_url;
            this._lastOverlayMeta = result.overlay_meta;

            const name = 'SAM3 Preview';
            if (this.isLocalMode()) {
                this.platform.localImage?.showLocalAnalysisLayer(
                    this.currentOverlayId, result.overlay_url,
                    result.overlay_meta?.width, result.overlay_meta?.height, name, true
                );
            } else if (window.mapManager) {
                window.mapManager.showAnalysisLayer(this.currentOverlayId, result.overlay_url, name, null, true);
            }

            this._makeOverlayInteractive(this.currentOverlayId);
            this._updatePointCounts();
            this._setStatus(`Score: ${result.score.toFixed(3)} | ${result.pixel_count.toLocaleString()} px`);
            await this._awaitOverlayPaint(this.currentOverlayId);
        };

        try {
            // First call per AOI triggers backend image-encoding (slow, ~5–15s);
            // subsequent calls reuse the embedding. We can't easily distinguish here,
            // so use a generic message that covers both.
            if (this.platform && typeof this.platform.withLoading === 'function') {
                await this.platform.withLoading('Running SAM3 segmentation...', run);
            } else {
                await run();
            }
        } catch (error) {
            console.error('SAM3 segmentation error:', error);
            this._setStatus('Error');
            this.platform.showNotification(`SAM3 failed: ${error.message}`, 'error');
        } finally {
            this._segmentPending = false;
        }
    }

    /** Resolve once the analysis overlay's <img> finishes decoding. 15s safety timeout. */
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

    // ===========================================================================
    // TEXT-MODE SEGMENTATION (multi-instance PCS)
    // ===========================================================================
    async _runTextSegmentation() {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification('Process or load an image first', 'warning');
            return;
        }
        const prompt = this.textPrompt;
        const threshold = this.textThreshold;

        this._setTextStatus('Running...');
        this._clearTextInstances();

        const run = async () => {
            let endpoint, body;
            if (imageInfo.uploaded) {
                endpoint = '/api/local/uploaded/sam3/text-predict';
                body = { upload_id: imageInfo.upload_id, prompt, score_threshold: threshold };
            } else {
                endpoint = '/api/sam3/text-predict';
                body = {
                    image_id: imageInfo.id,
                    bbox: imageInfo.bbox,
                    geometry: imageInfo.geometry,
                    prompt, score_threshold: threshold,
                };
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: 'Text prediction failed' }));
                throw new Error(err.detail || 'Text prediction failed');
            }
            const result = await response.json();

            this.textInstances = (result.instances || []).map((inst, idx) => {
                const layerId = `sam3-text-${inst.mask_id}`;
                const name = `${prompt} #${idx + 1}`;
                if (this.isLocalMode()) {
                    this.platform.localImage?.showLocalAnalysisLayer(
                        layerId, inst.overlay_url,
                        inst.overlay_meta?.width, inst.overlay_meta?.height, name, true
                    );
                } else if (window.mapManager) {
                    window.mapManager.showAnalysisLayer(layerId, inst.overlay_url, name, null, true);
                }
                return {
                    mask_id: inst.mask_id,
                    layerId,
                    overlayUrl: inst.overlay_url,
                    overlayMeta: inst.overlay_meta,
                    score: inst.score,
                    color_hex: inst.color_hex,
                    pixel_count: inst.pixel_count,
                    name,
                    visible: true,
                };
            });

            this._setTextStatus(`${this.textInstances.length} instance(s)`);
            this._renderTextInstances();
            // Wait for at least the first instance overlay to paint so the loading
            // overlay doesn't drop before the user sees anything.
            const first = this.textInstances[0];
            if (first) await this._awaitOverlayPaint(first.layerId);
        };

        try {
            if (this.platform && typeof this.platform.withLoading === 'function') {
                await this.platform.withLoading(`Running SAM3 text segmentation (${prompt})...`, run);
            } else {
                await run();
            }
        } catch (error) {
            console.error('SAM3 text predict error:', error);
            this._setTextStatus('Error');
            this.platform.showNotification(`SAM3 text failed: ${error.message}`, 'error');
        }
    }

    _clearTextInstances() {
        this.textInstances.forEach((inst) => this._hideOverlayById(inst.layerId));
        this.textInstances = [];
        this.textHighlightId = null;
        const list = document.getElementById('sam3-text-instances');
        if (list) list.innerHTML = '';
    }

    _renderTextInstances() {
        const list = document.getElementById('sam3-text-instances');
        if (!list) return;
        list.innerHTML = '';
        this.textInstances.forEach((inst, i) => {
            const row = document.createElement('div');
            row.className = 'sam3-instance-row';
            row.dataset.maskId = inst.mask_id;
            row.style.cssText = 'display:flex; align-items:center; gap:6px; padding:3px 4px; font-size:11px; cursor:pointer; border-radius:3px;';
            row.innerHTML = `
                <span class="sam3-instance-chip" style="display:inline-block; width:12px; height:12px; border-radius:2px; background:${inst.color_hex}; flex:0 0 12px;"></span>
                <span style="flex:1; overflow:hidden; text-overflow:ellipsis;">#${i + 1} ${inst.score.toFixed(2)} · ${inst.pixel_count.toLocaleString()}px</span>
                <button class="sam3-mini-btn" data-act="zoom"  title="Zoom to instance">🔍</button>
                <button class="sam3-mini-btn" data-act="hide"  title="Toggle visibility">${inst.visible ? '👁' : '🚫'}</button>
                <button class="sam3-mini-btn" data-act="save"  title="Save mask">💾</button>
                <button class="sam3-mini-btn" data-act="trash" title="Remove">✕</button>
            `;
            list.appendChild(row);

            row.addEventListener('click', (e) => {
                if (e.target.closest('button')) return;
                this._highlightTextInstance(inst.mask_id);
            });
            row.querySelectorAll('button').forEach((btn) => {
                btn.onclick = (ev) => {
                    ev.stopPropagation();
                    const act = btn.dataset.act;
                    if (act === 'zoom') this._zoomToInstance(inst);
                    else if (act === 'hide')  this._toggleInstanceVisibility(inst, btn);
                    else if (act === 'save')  this._saveTextInstance(inst);
                    else if (act === 'trash') this._removeTextInstance(inst);
                };
            });
        });
    }

    _highlightTextInstance(maskId) {
        this.textHighlightId = (this.textHighlightId === maskId) ? null : maskId;
        const others = this.textInstances.filter((x) => x.mask_id !== this.textHighlightId);
        const opacityHi = 1.0;
        const opacityLo = this.textHighlightId ? 0.25 : 1.0;

        // Set opacity per-layer (Leaflet image overlays support setOpacity)
        const setOp = (layerId, op) => {
            const layer = window.mapManager?.analysisLayers?.[layerId]
                       || this.platform.localImage?.localAnalysisOverlays?.[layerId];
            if (layer && typeof layer.setOpacity === 'function') layer.setOpacity(op);
        };

        if (this.textHighlightId) {
            setOp(this.textInstances.find((x) => x.mask_id === this.textHighlightId)?.layerId, opacityHi);
            others.forEach((x) => setOp(x.layerId, opacityLo));
        } else {
            this.textInstances.forEach((x) => setOp(x.layerId, 1.0));
        }

        // Visual cue on the row
        document.querySelectorAll('.sam3-instance-row').forEach((row) => {
            row.style.background = (row.dataset.maskId === this.textHighlightId) ? '#f3e9ff' : '';
        });
    }

    _zoomToInstance(inst) {
        const map = this.getMap();
        if (!map) return;
        const meta = inst.overlayMeta;
        if (!meta?.bounds) return;  // local mode has no geo-bounds
        const [s, w, n, e] = meta.bounds;
        map.flyToBounds([[s, w], [n, e]], { padding: [40, 40], duration: 0.4 });
    }

    _toggleInstanceVisibility(inst, btn) {
        if (inst.visible) {
            this._hideOverlayById(inst.layerId);
            inst.visible = false;
            btn.textContent = '🚫';
        } else {
            this._showOverlayById(inst.layerId, inst.overlayUrl, inst.overlayMeta, inst.name);
            inst.visible = true;
            btn.textContent = '👁';
        }
    }

    async _saveTextInstance(inst) {
        try {
            const r = await fetch('/api/sam3/save-mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mask_id: inst.mask_id }),
            });
            if (!r.ok) throw new Error('save failed');
            const entry = {
                mask_id: inst.mask_id,
                visible: true,
                layerId: inst.layerId,
                overlayUrl: inst.overlayUrl,
                overlayMeta: inst.overlayMeta,
                color_hex: inst.color_hex,
                name: inst.name,
            };
            this.savedMasks.push(entry);
            this._renderSavedMask(entry);

            // Register the saved instance so change-detection can pick it.
            // SAM3 outputs a binary mask directly, so hasMask is true on save.
            if (typeof this.platform.registerSlotAnalysis === 'function') {
                this.platform.registerSlotAnalysis({
                    id: inst.mask_id,
                    type: 'sam3',
                    name: inst.name || `SAM3 ${this.savedMasks.length}`,
                    hasMask: true,
                });
            }

            this.platform.showNotification(`Saved "${inst.name}"`, 'success');
        } catch (e) {
            this.platform.showNotification(`Save failed: ${e.message}`, 'error');
        }
    }

    async _removeTextInstance(inst) {
        this._hideOverlayById(inst.layerId);
        try { await fetch(`/api/sam3/masks/${inst.mask_id}`, { method: 'DELETE' }); } catch (_) {}
        this.textInstances = this.textInstances.filter((x) => x.mask_id !== inst.mask_id);
        if (this.textHighlightId === inst.mask_id) this.textHighlightId = null;
        this._renderTextInstances();
    }

    // ===========================================================================
    // SAVE & NEXT (point mode)
    // ===========================================================================
    async saveAndNext() {
        if (!this.currentMaskId) return;
        const saveBtn = document.getElementById('sam3-save');
        if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = '...'; }
        try {
            const r = await fetch('/api/sam3/save-mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mask_id: this.currentMaskId }),
            });
            if (!r.ok) throw new Error('Failed to save mask');
            const entry = {
                mask_id: this.currentMaskId,
                visible: true,
                layerId: this.currentOverlayId,
                overlayUrl: this._lastOverlayUrl,
                overlayMeta: this._lastOverlayMeta,
            };
            this.savedMasks.push(entry);
            this._renderSavedMask(entry);

            // Register the saved instance so change-detection can pick it.
            // SAM3 outputs a binary mask directly, so hasMask is true on save.
            if (typeof this.platform.registerSlotAnalysis === 'function') {
                this.platform.registerSlotAnalysis({
                    id: entry.mask_id,
                    type: 'sam3',
                    name: `SAM3 Mask ${this.savedMasks.length}`,
                    hasMask: true,
                });
            }

            this.currentMaskId = null;
            this.currentOverlayId = null;
            this.clearCurrentPoints();
            this._setStatus('');
            this.platform.showNotification('Mask saved. Place new points for the next object.', 'success');
        } catch (error) {
            this.platform.showNotification(`Failed to save: ${error.message}`, 'error');
        } finally {
            if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = '💾 Save'; }
        }
    }

    // ===========================================================================
    // SAVED MASKS LIST
    // ===========================================================================
    _renderSavedMask(maskEntry) {
        const item = document.querySelector('.analysis-item.sam3-option');
        const savedList = item?.querySelector('.sam3-saved-list');
        if (!savedList) return;
        if (savedList.querySelector(`[data-mask-id="${maskEntry.mask_id}"]`)) return;

        const idx = this.savedMasks.indexOf(maskEntry) + 1;
        const colorChip = maskEntry.color_hex
            ? `background:${maskEntry.color_hex};`
            : 'background: linear-gradient(135deg, #E7344C 0%, #941E26 100%);';
        const name = maskEntry.name || `SAM3 Mask ${idx}`;

        const maskItem = document.createElement('div');
        maskItem.className = 'sam3-mask-item';
        maskItem.dataset.maskId = maskEntry.mask_id;
        maskItem.dataset.visible = maskEntry.visible ? 'true' : 'false';
        maskItem.innerHTML = `
            <div class="sam3-mask-header">
                <div class="sam3-mask-color" style="${colorChip}"></div>
                <div class="sam3-mask-info">
                    <span class="sam3-mask-name">${name}</span>
                    <span class="sam3-mask-status">${maskEntry.visible ? 'Visible' : 'Hidden'}</span>
                </div>
                <div class="sam3-mask-actions">
                    <button class="sam3-mask-remove-btn" title="Remove">✕</button>
                </div>
            </div>
        `;
        savedList.appendChild(maskItem);

        maskItem.addEventListener('click', (e) => {
            if (e.target.closest('.sam3-mask-remove-btn')) return;
            this._toggleMaskOverlay(maskEntry.mask_id);
        });
        maskItem.querySelector('.sam3-mask-remove-btn').onclick = (e) => {
            e.stopPropagation();
            this._removeMask(maskEntry.mask_id);
        };
    }

    _toggleMaskOverlay(maskId) {
        const entry = this.savedMasks.find((m) => m.mask_id === maskId);
        if (!entry) return;
        const maskItem = document.querySelector(`.sam3-mask-item[data-mask-id="${maskId}"]`);
        if (entry.visible) {
            this._hideOverlayById(entry.layerId);
            entry.visible = false;
            if (maskItem) {
                maskItem.dataset.visible = 'false';
                maskItem.querySelector('.sam3-mask-status').textContent = 'Hidden';
            }
        } else {
            this._showOverlayById(entry.layerId, entry.overlayUrl, entry.overlayMeta, entry.name);
            entry.visible = true;
            if (maskItem) {
                maskItem.dataset.visible = 'true';
                maskItem.querySelector('.sam3-mask-status').textContent = 'Visible';
            }
        }
    }

    async _removeMask(maskId) {
        const entry = this.savedMasks.find((m) => m.mask_id === maskId);
        if (!entry) return;
        this._hideOverlayById(entry.layerId);
        try { await fetch(`/api/sam3/masks/${maskId}`, { method: 'DELETE' }); } catch (_) {}
        this.savedMasks = this.savedMasks.filter((m) => m.mask_id !== maskId);
        const el = document.querySelector(`.sam3-mask-item[data-mask-id="${maskId}"]`);
        if (el) el.remove();
    }

    // ===========================================================================
    // OVERLAY HELPERS
    // ===========================================================================
    _hideOverlayById(layerId) {
        if (!layerId) return;
        if (this.isLocalMode()) {
            this.platform.localImage?.hideLocalAnalysisLayer(layerId);
        } else if (window.mapManager) {
            window.mapManager.hideAnalysisLayer(layerId);
        }
    }

    _showOverlayById(layerId, overlayUrl, meta, name) {
        const lbl = name || 'SAM3';
        if (this.isLocalMode()) {
            this.platform.localImage?.showLocalAnalysisLayer(
                layerId, overlayUrl, meta?.width, meta?.height, lbl, true
            );
        } else if (window.mapManager) {
            window.mapManager.showAnalysisLayer(layerId, overlayUrl, lbl, null, true);
        }
    }

    _makeOverlayInteractive(layerId) {
        let layer = window.mapManager?.analysisLayers?.[layerId];
        if (!layer && this.isLocalMode()) {
            layer = this.platform.localImage?.localAnalysisOverlays?.[layerId];
        }
        if (!layer) return;
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

    _hideCurrentPreview() {
        this._hideOverlayById(this.currentOverlayId);
        this.currentMaskId = null;
        this.currentOverlayId = null;
        this._updatePointCounts();
    }

    // ===========================================================================
    // Per-slot stash (Time A / Time B)
    // ===========================================================================
    _freezeToSlot(slotId) {
        if (slotId !== 'A' && slotId !== 'B') return;
        this._segmentAbort = true;
        this._segmentPending = false;

        this._slotStash[slotId] = {
            mode: this.mode,
            positivePoints: this.positivePoints.slice(),
            negativePoints: this.negativePoints.slice(),
            markers: this.markers.slice(),
            clickHistory: this.clickHistory.slice(),
            savedMasks: this.savedMasks.slice(),
            currentMaskId: this.currentMaskId,
            currentOverlayId: this.currentOverlayId,
            _lastOverlayUrl: this._lastOverlayUrl,
            _lastOverlayMeta: this._lastOverlayMeta,
            clickMode: this.clickMode,
            textPrompt: this.textPrompt,
            textThreshold: this.textThreshold,
            textInstances: this.textInstances.slice(),
            textHighlightId: this.textHighlightId,
        };

        const map = this.getMap();
        this._slotStash[slotId].markers.forEach(({ marker, label }) => {
            if (marker && map?.hasLayer?.(marker)) map.removeLayer(marker);
            if (label && map?.hasLayer?.(label))   map.removeLayer(label);
        });
    }

    _thawFromSlot(slotId) {
        const stash = (slotId === 'A' || slotId === 'B') ? this._slotStash[slotId] : null;
        if (stash) {
            this.mode = stash.mode || 'point';
            this.positivePoints = (stash.positivePoints || []).slice();
            this.negativePoints = (stash.negativePoints || []).slice();
            this.markers = (stash.markers || []).slice();
            this.clickHistory = (stash.clickHistory || []).slice();
            this.savedMasks = (stash.savedMasks || []).slice();
            this.currentMaskId = stash.currentMaskId || null;
            this.currentOverlayId = stash.currentOverlayId || null;
            this._lastOverlayUrl = stash._lastOverlayUrl || null;
            this._lastOverlayMeta = stash._lastOverlayMeta || null;
            this.clickMode = stash.clickMode || 'positive';
            this.textPrompt = stash.textPrompt || '';
            this.textThreshold = stash.textThreshold ?? 0.5;
            this.textInstances = (stash.textInstances || []).slice();
            this.textHighlightId = stash.textHighlightId || null;

            const map = this.getMap();
            if (map) {
                this.markers.forEach(({ marker, label }) => {
                    if (marker && !map.hasLayer?.(marker)) marker.addTo(map);
                    if (label && !map.hasLayer?.(label))   label.addTo(map);
                });
            }
            this._slotStash[slotId] = null;
        } else {
            this.mode = 'point';
            this.positivePoints = [];
            this.negativePoints = [];
            this.markers = [];
            this.clickHistory = [];
            this.savedMasks = [];
            this.currentMaskId = null;
            this.currentOverlayId = null;
            this._lastOverlayUrl = null;
            this._lastOverlayMeta = null;
            this.clickMode = 'positive';
            this.textPrompt = '';
            this.textThreshold = 0.5;
            this.textInstances = [];
            this.textHighlightId = null;
        }
        this._segmentAbort = null;
    }

    handleSlotChange(oldSlot, newSlot) {
        if (oldSlot === newSlot) return;
        this.stopSelection();
        this._freezeToSlot(oldSlot);
        this._thawFromSlot(newSlot);
        const item = document.querySelector('.analysis-item.sam3-option');
        const list = item?.querySelector('.sam3-saved-list');
        if (list) {
            list.innerHTML = '';
            this.savedMasks.forEach((m) => this._renderSavedMask(m));
            this._updatePointCounts();
        }
        // If the panel is open, refresh both panes for the new slot.
        if (item?.querySelector('.sam3-ui')) {
            this._renderPointPane();
            this._renderTextPane();
        }
    }

    // ===========================================================================
    // CLEANUP
    // ===========================================================================
    reset() {
        this.stopSelection();
        this.clearCurrentPoints();
        this.savedMasks.forEach((m) => this._hideOverlayById(m.layerId));
        this.savedMasks = [];
        this._clearTextInstances();
        this.currentMaskId = null;
        this.currentOverlayId = null;
        const item = document.querySelector('.analysis-item.sam3-option');
        item?.querySelector('.sam3-ui')?.remove();
        item?.classList.remove('expanded', 'active');
        const infoEl = item?.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    cleanup() { this.reset(); }
}
