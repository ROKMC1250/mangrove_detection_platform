/**
 * Spectral Analysis Controller
 *
 * Supports both satellite (GEE) and local image modes.
 *
 * Flow:
 * 1. Click Spectral Analysis → Show index selector + Compute button
 * 2. Select index (preset or custom) → Click Compute
 * 3. Result appears in results list with colorbar + threshold + pixel inspect
 * 4. Multiple indices can be computed and stacked
 */

class SpectralAnalysisController {
    constructor(platformController) {
        this.platform = platformController;
        this.results = [];  // Array of computed index results
        this.overlayLayers = {};  // layerId → true (managed via mapManager or localImage)

        // Per-slot stash, frozen when the user flips Time A / Time B
        this._slotStash = { A: null, B: null };
        this._originSlot = null;  // stamped at computeIndex kickoff

        this.presetIndices = [
            { id: 'ndvi', name: 'NDVI', formula: '(NIR-RED)/(NIR+RED)' },
            { id: 'mvi',  name: 'MVI',  formula: '(NIR-GREEN)/(SWIR1-GREEN)' },
            { id: 'ndmi', name: 'NDMI', formula: '(NIR-SWIR)/(NIR+SWIR)' },
            { id: 'ndwi', name: 'NDWI', formula: '(GREEN-NIR)/(GREEN+NIR)' },
            { id: 'savi', name: 'SAVI', formula: '((NIR-RED)/(NIR+RED+L))*(1+L)' },
            { id: 'evi',  name: 'EVI',  formula: '2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)' },
            { id: 'custom', name: 'Custom Index' }
        ];

        this.allBands = [
            'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'
        ];
    }

    isLocalMode() {
        const li = this.platform.localImage;
        return !!li?.localMapActive || !!li?.isUploadedImageActive;
    }

    isUploadedMode() {
        const li = this.platform.localImage;
        return li?.currentMode === 'load-image' && !!li?.isUploadedImageActive;
    }

    getImageInfo() {
        if (this.isUploadedMode()) {
            const li = this.platform.localImage;
            return {
                local: true,
                uploaded: true,
                upload_id: li.uploadId,
                band_roles: li.uploadedBandRoles || {},
                bands: li.availableBands || []
            };
        }
        if (this.isLocalMode()) {
            const li = this.platform.localImage;
            if (!li?.currentImageDir || !li?.currentAlgoDir) return null;
            return {
                local: true,
                image_dir: li.currentImageDir,
                algorithm_dir: li.currentAlgoDir,
                bands: li.availableBands || []
            };
        }
        const imageId = this.platform.selectedImageId;
        if (!imageId) return null;
        return {
            id: imageId,
            bbox: window.mapManager?.getCurrentBounds(),
            geometry: window.mapManager?.getCurrentGeoJSON()?.geometry
        };
    }

    handleItemClick(item) {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification(this.isLocalMode() ? 'Load a local image first' : 'Process an image first', 'warning');
            return;
        }

        const hasUI = item?.querySelector('.sa-ui');
        if (hasUI) {
            this.hideUI(item);
        } else {
            this.showSetupUI(item);
        }
    }

    showSetupUI(item) {
        if (!item) return;

        item.querySelector('.sa-ui')?.remove();

        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        // In local mode, use available band names for custom index
        const localBands = this.isLocalMode()
            ? (this.platform.localImage?.availableBands || []).map(b => b.replace('after_', '').replace('.tif', ''))
            : this.allBands;

        const ui = document.createElement('div');
        ui.className = 'sa-ui sa-setup';
        ui.innerHTML = `
            <div class="sa-header">
                <span class="sa-title">Spectral Analysis</span>
            </div>
            <div class="sa-row">
                <select class="sa-index-select">
                    ${this.presetIndices.map(idx =>
                        `<option value="${idx.id}">${idx.name}${idx.formula ? ' — ' + idx.formula : ''}</option>`
                    ).join('')}
                </select>
                <button class="sa-run-btn sa-btn primary">Compute</button>
            </div>
            <div class="sa-custom-panel" style="display:none;">
                <div class="sa-custom-row">
                    <label>Band A</label>
                    <select class="sa-band-a">
                        ${localBands.map((b, i) => `<option value="${b}" ${i === 0 ? 'selected' : ''}>${b}</option>`).join('')}
                    </select>
                    <label>Band B</label>
                    <select class="sa-band-b">
                        ${localBands.map((b, i) => `<option value="${b}" ${i === 1 ? 'selected' : ''}>${b}</option>`).join('')}
                    </select>
                </div>
                <div class="sa-custom-row">
                    <label>Colormap</label>
                    <select class="sa-colormap">
                        <option value="RdYlGn">RdYlGn</option>
                        <option value="viridis" selected>viridis</option>
                        <option value="plasma">plasma</option>
                        <option value="coolwarm">coolwarm</option>
                        <option value="RdYlBu">RdYlBu</option>
                    </select>
                </div>
            </div>
            <div class="sa-results-list"></div>
        `;

        item.appendChild(ui);
        item.classList.add('expanded');

        // Re-render existing results
        this.results.forEach(r => this._reRenderResultItem(item, r));

        // Events
        const select = ui.querySelector('.sa-index-select');
        const customPanel = ui.querySelector('.sa-custom-panel');

        select.onchange = () => {
            customPanel.style.display = select.value === 'custom' ? 'block' : 'none';
        };

        ui.querySelector('.sa-run-btn').onclick = (e) => {
            e.stopPropagation();
            this.computeIndex(item);
        };
    }

    _reRenderResultItem(item, storedResult) {
        // Re-render a previously computed result when UI is re-opened
        const resultsList = item.querySelector('.sa-results-list');
        if (!resultsList) return;
        if (resultsList.querySelector(`[data-result-id="${storedResult.id}"]`)) return;
        this._buildResultItemDOM(resultsList, storedResult);
    }

    hideUI(item) {
        if (!item) return;
        item.querySelector('.sa-ui')?.remove();
        item.classList.remove('expanded');
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    async computeIndex(item) {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) return;

        const ui = item.querySelector('.sa-ui');
        if (!ui) return;

        // Remember which slot this compute belongs to — if the user switches
        // slots before the response arrives we must not leak the result.
        this._originSlot = this.platform.currentSlot;

        const select = ui.querySelector('.sa-index-select');
        const indexType = select.value;
        const runBtn = ui.querySelector('.sa-run-btn');

        let endpoint, payload;

        if (imageInfo.uploaded) {
            endpoint = '/api/local/uploaded/compute-spectral-index';
            payload = {
                upload_id: imageInfo.upload_id,
                index_type: indexType,
                band_roles: imageInfo.band_roles || {}
            };
        } else if (imageInfo.local) {
            endpoint = '/api/local/compute-spectral-index';
            payload = {
                image_dir: imageInfo.image_dir,
                algorithm_dir: imageInfo.algorithm_dir,
                index_type: indexType
            };
        } else {
            endpoint = '/api/compute-spectral-index';
            payload = {
                image_id: imageInfo.id,
                bbox: imageInfo.bbox,
                geometry: imageInfo.geometry,
                index_type: indexType
            };
        }

        if (indexType === 'custom') {
            payload.band_a = ui.querySelector('.sa-band-a').value;
            payload.band_b = ui.querySelector('.sa-band-b').value;
            payload.colormap = ui.querySelector('.sa-colormap').value;
        }

        runBtn.disabled = true;
        runBtn.textContent = '...';

        const indexLabel = (indexType || 'index').toUpperCase();
        const run = async () => {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to compute index');
            }
            const result = await response.json();
            this.addResultItem(item, result);
            // Hold the overlay until the freshly added analysis image actually paints.
            if (result && result.model_id) {
                await this._awaitOverlayPaint(result.model_id);
            }
        };

        try {
            if (this.platform && typeof this.platform.withLoading === 'function') {
                await this.platform.withLoading(`Computing ${indexLabel}...`, run);
            } else {
                await run();
            }
        } catch (err) {
            console.error('Spectral index error:', err);
            this.platform.showNotification(`Error: ${err.message}`, 'error');
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Compute';
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

    addResultItem(item, result) {
        // Drop completions belonging to a slot the user has since switched
        // away from. Origin is set by computeIndex at kickoff.
        const origin = this._originSlot;
        this._originSlot = null;
        if (origin && origin !== this.platform.currentSlot) {
            this.platform.showNotification(
                `Spectral index on Time ${origin} dropped — you switched slots mid-run.`,
                'warning'
            );
            return;
        }

        const resultsList = item.querySelector('.sa-results-list');
        if (!resultsList) return;

        const resultId = result.model_id;
        const storedResult = { id: resultId, ...result };
        this.results.push(storedResult);

        this._buildResultItemDOM(resultsList, storedResult);

        // Auto-show overlay
        this.toggleOverlay(resultId, storedResult);
    }

    _buildResultItemDOM(resultsList, result) {
        const resultId = result.id;
        const colormap = result.colormap || {};
        const minVal = colormap.min_val != null ? colormap.min_val.toFixed(3) : '0';
        const maxVal = colormap.max_val != null ? colormap.max_val.toFixed(3) : '1';
        const cmapName = colormap.name || 'viridis';
        const label = colormap.label || result.name;

        const resultItem = document.createElement('div');
        resultItem.className = 'sa-result-item';
        resultItem.dataset.resultId = resultId;
        resultItem.dataset.visible = 'false';

        resultItem.innerHTML = `
            <div class="sa-result-header">
                <img class="sa-result-thumb" src="${result.preview_url}" alt="${result.name}" />
                <div class="sa-result-info">
                    <span class="sa-result-name">${result.name}</span>
                    <span class="sa-result-status">Click to toggle overlay</span>
                </div>
                <div class="sa-result-actions">
                    <button class="sa-pixel-btn" title="Inspect pixel values">🖱️</button>
                    <button class="sa-remove-btn" title="Remove">✕</button>
                </div>
            </div>
            <div class="colorbar-container" data-model-id="${resultId}"
                 data-min-val="${colormap.min_val}" data-max-val="${colormap.max_val}">
                <div class="colorbar-header">
                    <div class="colorbar-label">${label}</div>
                </div>
                <div class="colorbar-with-threshold">
                    <div class="colorbar-track">
                        <div class="colorbar-gradient ${cmapName}"></div>
                        <div class="colorbar-selection" style="left:0%;right:0%;"></div>
                        <div class="colorbar-handle min-handle" style="left:0%"></div>
                        <div class="colorbar-handle max-handle" style="left:100%"></div>
                    </div>
                </div>
                <div class="colorbar-values">
                    <input type="number" class="colorbar-min-input" value="${minVal}" step="0.01">
                    <div class="colorbar-buttons">
                        <button class="colorbar-apply-btn">Apply</button>
                        <button class="colorbar-cancel-btn">Cancel</button>
                    </div>
                    <input type="number" class="colorbar-max-input" value="${maxVal}" step="0.01">
                </div>
            </div>
            <div class="sa-result-stats" style="display:none;">
                <span class="sa-result-pixels"></span>
                <span class="sa-result-pct"></span>
            </div>
        `;

        resultsList.appendChild(resultItem);

        // Toggle overlay on header click
        resultItem.querySelector('.sa-result-header').onclick = (e) => {
            if (e.target.closest('.sa-result-actions')) return;
            this.toggleOverlay(resultId, result);
        };

        // Pixel inspect
        resultItem.querySelector('.sa-pixel-btn').onclick = (e) => {
            e.stopPropagation();
            if (this.platform.thresholdController) {
                this.platform.thresholdController.enablePixelInspection(resultId, colormap);
            }
        };

        // Remove
        resultItem.querySelector('.sa-remove-btn').onclick = (e) => {
            e.stopPropagation();
            this.removeResult(resultId);
        };

        // Setup colorbar threshold
        this.setupColorbarThreshold(resultItem, resultId, result);

        // Rehydrate UI for results that already have a saved mask — happens
        // when the SA panel is rebuilt after a slot switch (`handleSlotChange`
        // → `_thawFromSlot` → loop over `this.results`). Without this the
        // card would default back to score-thumb / hidden-stats even though
        // the underlying mask state on `result` is still live.
        if (result.showingMask && result.mask_result) {
            const m = result.mask_result;
            const thumb = resultItem.querySelector('.sa-result-thumb');
            if (thumb && m.preview_url) thumb.src = m.preview_url;
            const stats = resultItem.querySelector('.sa-result-stats');
            if (stats) {
                stats.style.display = 'flex';
                const px = stats.querySelector('.sa-result-pixels');
                const pct = stats.querySelector('.sa-result-pct');
                if (px) px.textContent = `🎯 ${(m.detected_pixels ?? 0).toLocaleString()} px`;
                if (pct) pct.textContent = `${m.detection_percentage ?? 0}%`;
            }
            const status = resultItem.querySelector('.sa-result-status');
            if (status) {
                status.textContent = `Mask (${(m.min_threshold ?? 0).toFixed(3)} – ${(m.max_threshold ?? 0).toFixed(3)})`;
            }
            // If the overlay layer was carried over via the slot stash,
            // mark the card as visible so the next click toggles correctly.
            if (this.overlayLayers && this.overlayLayers[resultId]) {
                resultItem.dataset.visible = 'true';
            }
        }
    }

    setupColorbarThreshold(resultItem, modelId, result) {
        const container = resultItem.querySelector('.colorbar-container');
        if (!container) return;

        const colormap = result.colormap || {};
        const minVal = colormap.min_val ?? 0;
        const maxVal = colormap.max_val ?? 1;
        let currentMin = minVal;
        let currentMax = maxVal;

        const track = container.querySelector('.colorbar-track');
        const selection = container.querySelector('.colorbar-selection');
        const minHandle = container.querySelector('.min-handle');
        const maxHandle = container.querySelector('.max-handle');
        const minInput = container.querySelector('.colorbar-min-input');
        const maxInput = container.querySelector('.colorbar-max-input');
        const applyBtn = container.querySelector('.colorbar-apply-btn');
        const cancelBtn = container.querySelector('.colorbar-cancel-btn');

        const updateSelection = () => {
            const range = maxVal - minVal || 1;
            const leftPct = ((currentMin - minVal) / range) * 100;
            const rightPct = ((maxVal - currentMax) / range) * 100;
            selection.style.left = `${Math.max(0, leftPct)}%`;
            selection.style.right = `${Math.max(0, rightPct)}%`;
            minHandle.style.left = `${Math.max(0, leftPct)}%`;
            maxHandle.style.left = `${Math.min(100, 100 - rightPct)}%`;
        };

        const startDrag = (handle, isMin) => {
            const onMouseMove = (e) => {
                const rect = track.getBoundingClientRect();
                let pct = (e.clientX - rect.left) / rect.width;
                pct = Math.max(0, Math.min(1, pct));
                const val = minVal + pct * (maxVal - minVal);
                if (isMin) {
                    currentMin = Math.min(val, currentMax);
                    minInput.value = currentMin.toFixed(3);
                } else {
                    currentMax = Math.max(val, currentMin);
                    maxInput.value = currentMax.toFixed(3);
                }
                updateSelection();
            };
            const onMouseUp = () => {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            };
            handle.addEventListener('mousedown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            });
        };

        startDrag(minHandle, true);
        startDrag(maxHandle, false);

        minInput.onchange = () => {
            currentMin = parseFloat(minInput.value) || minVal;
            updateSelection();
        };
        maxInput.onchange = () => {
            currentMax = parseFloat(maxInput.value) || maxVal;
            updateSelection();
        };

        applyBtn.onclick = async (e) => {
            e.stopPropagation();
            await this.applyThreshold(modelId, currentMin, currentMax, colormap, result);
        };

        cancelBtn.onclick = (e) => {
            e.stopPropagation();
            currentMin = minVal;
            currentMax = maxVal;
            minInput.value = minVal.toFixed(3);
            maxInput.value = maxVal.toFixed(3);
            updateSelection();

            // Drop the mask state. After cancel the card is back to "score
            // map" mode — thumbnail, overlay, and stats follow.
            result.showingMask = false;
            result.mask_result = null;
            const card = document.querySelector(`.sa-result-item[data-result-id="${modelId}"]`);
            if (card) {
                const thumb = card.querySelector('.sa-result-thumb');
                if (thumb && result.preview_url) thumb.src = result.preview_url;
                const stats = card.querySelector('.sa-result-stats');
                if (stats) stats.style.display = 'none';
                const status = card.querySelector('.sa-result-status');
                if (status) status.textContent = `${result.name} — Score map`;
            }
            if (result.overlay_url) {
                this.showOverlayOnMap(modelId, result.overlay_url, result.overlay_meta);
            }
        };
    }

    async applyThreshold(modelId, min, max, colormap, result) {
        try {
            const imageInfo = this.getImageInfo();
            let endpoint, body;

            if (imageInfo?.uploaded) {
                endpoint = '/api/local/uploaded/apply-threshold-range';
                body = {
                    upload_id: imageInfo.upload_id,
                    model_id: modelId,
                    min_threshold: min,
                    max_threshold: max,
                    colormap: colormap
                };
            } else if (imageInfo?.local) {
                endpoint = '/api/local/apply-threshold-range';
                body = {
                    image_dir: imageInfo.image_dir,
                    algorithm_dir: imageInfo.algorithm_dir,
                    model_id: modelId,
                    min_threshold: min,
                    max_threshold: max,
                    colormap: colormap
                };
            } else {
                endpoint = '/api/apply-threshold-range';
                body = {
                    image_id: this.platform.selectedImageId,
                    model_id: modelId,
                    min_threshold: min,
                    max_threshold: max,
                    colormap: colormap
                };
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            if (!response.ok) throw new Error('Failed to apply threshold');

            const data = await response.json();

            // Save mask state onto the result so toggling the card off and
            // back on (or switching tabs and returning) keeps showing the
            // mask, not the score. Mirrors target-detection's `showingMask`
            // contract. The original score `preview_url`/`overlay_url` on
            // `result` are preserved so Cancel can restore them.
            result.mask_result = {
                preview_url: data.preview_url,
                overlay_url: data.overlay_url,
                overlay_meta: data.overlay_meta || result.overlay_meta,
                detected_pixels: data.detected_pixels,
                detection_percentage: data.detection_percentage,
                min_threshold: data.min_threshold,
                max_threshold: data.max_threshold,
                analysis_id: data.analysis_id,
            };
            result.showingMask = true;

            // Render the binary mask overlay on the map. Reuses `modelId`
            // as the layer id, so the mask cleanly replaces the score
            // overlay and gets exactly one entry in the layer panel.
            this.showOverlayOnMap(modelId, data.overlay_url, data.overlay_meta || result.overlay_meta, true);

            // Register the binary mask in the slot registry so change
            // detection can pick it up. Backend returns analysis_id (the
            // cache key under which the mask is stored).
            if (data.analysis_id && typeof this.platform.registerSlotAnalysis === 'function') {
                const label = colormap?.label || result?.name || modelId;
                this.platform.registerSlotAnalysis({
                    id: data.analysis_id,
                    type: 'spectral',
                    name: `${label} (${min.toFixed(3)}–${max.toFixed(3)})`,
                    hasMask: true,
                });
            }

            // Update the result card UI: thumbnail flips to the mask preview,
            // stats panel shows pixel count + percentage, status text reflects
            // the active threshold.
            const resultItem = document.querySelector(`.sa-result-item[data-result-id="${modelId}"]`);
            if (resultItem) {
                resultItem.dataset.visible = 'true';
                const thumb = resultItem.querySelector('.sa-result-thumb');
                if (thumb && data.preview_url) thumb.src = data.preview_url;
                const status = resultItem.querySelector('.sa-result-status');
                if (status) status.textContent = `Mask (${min.toFixed(3)} – ${max.toFixed(3)})`;
                const stats = resultItem.querySelector('.sa-result-stats');
                if (stats) {
                    stats.style.display = 'flex';
                    const pxEl = stats.querySelector('.sa-result-pixels');
                    const pctEl = stats.querySelector('.sa-result-pct');
                    if (pxEl) pxEl.textContent = `🎯 ${(data.detected_pixels ?? 0).toLocaleString()} px`;
                    if (pctEl) pctEl.textContent = `${data.detection_percentage ?? 0}%`;
                }
            }
        } catch (err) {
            console.error('Threshold error:', err);
            this.platform.showNotification(`Error: ${err.message}`, 'error');
        }
    }

    toggleOverlay(resultId, result) {
        const resultItem = document.querySelector(`.sa-result-item[data-result-id="${resultId}"]`);
        const isVisible = resultItem?.dataset.visible === 'true';

        if (isVisible) {
            this.removeOverlayFromMap(resultId);
            if (resultItem) {
                resultItem.dataset.visible = 'false';
                const status = resultItem.querySelector('.sa-result-status');
                if (status) status.textContent = 'Click to toggle overlay';
            }
        } else {
            // Once the user has applied a threshold, the card represents a
            // saved mask — bring back the mask overlay (not the score) on
            // toggle. Cancel is the only path that drops back to the score
            // overlay. This is what makes the mask "stick" across visibility
            // toggles and image switches.
            const useMask = result?.showingMask && result?.mask_result?.overlay_url;
            const url = useMask ? result.mask_result.overlay_url : result.overlay_url;
            const meta = useMask ? (result.mask_result.overlay_meta || result.overlay_meta) : result.overlay_meta;
            if (url) {
                this.showOverlayOnMap(resultId, url, meta, !!useMask);
            }
            if (resultItem) {
                resultItem.dataset.visible = 'true';
                const status = resultItem.querySelector('.sa-result-status');
                if (status) {
                    if (useMask) {
                        const lo = result.mask_result.min_threshold;
                        const hi = result.mask_result.max_threshold;
                        status.textContent = `Mask (${(lo ?? 0).toFixed(3)} – ${(hi ?? 0).toFixed(3)})`;
                    } else {
                        status.textContent = `${result.name} — Click to hide`;
                    }
                }
            }
        }
    }

    showOverlayOnMap(layerId, overlayUrl, meta, isBinary = false) {
        this.removeOverlayFromMap(layerId);

        if (this.isLocalMode()) {
            const li = this.platform.localImage;
            if (li) {
                li.showLocalAnalysisLayer(layerId, overlayUrl, li.preloadedSize?.width, li.preloadedSize?.height, undefined, isBinary);
                this.overlayLayers[layerId] = true;
            }
        } else if (window.mapManager) {
            window.mapManager.showAnalysisLayer(layerId, overlayUrl, layerId, null, isBinary);
            this.overlayLayers[layerId] = true;
        }
    }

    removeOverlayFromMap(layerId) {
        if (!this.overlayLayers[layerId]) return;

        if (this.isLocalMode()) {
            this.platform.localImage?.hideLocalAnalysisLayer(layerId);
        } else if (window.mapManager) {
            window.mapManager.hideAnalysisLayer(layerId);
        }
        delete this.overlayLayers[layerId];
    }

    removeResult(resultId) {
        this.removeOverlayFromMap(resultId);
        this.results = this.results.filter(r => r.id !== resultId);
        const el = document.querySelector(`.sa-result-item[data-result-id="${resultId}"]`);
        if (el) el.remove();
    }

    cleanup() {
        Object.keys(this.overlayLayers).forEach(id => this.removeOverlayFromMap(id));
        this.results = [];
    }

    // ========== Per-slot stash (Time A / Time B) ==========

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
        const item = document.querySelector('.analysis-item.spectral-analysis-option');
        const list = item?.querySelector('.sa-results-list');
        if (list) {
            list.innerHTML = '';
            this.results.forEach(r => this._buildResultItemDOM(list, r));
        }
    }

}
