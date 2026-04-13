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

        try {
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

        } catch (err) {
            console.error('Spectral index error:', err);
            this.platform.showNotification(`Error: ${err.message}`, 'error');
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Compute';
        }
    }

    addResultItem(item, result) {
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
            this.showOverlayOnMap(modelId, data.overlay_url, data.overlay_meta || result.overlay_meta);

            const resultItem = document.querySelector(`.sa-result-item[data-result-id="${modelId}"]`);
            if (resultItem) {
                resultItem.dataset.visible = 'true';
                const status = resultItem.querySelector('.sa-result-status');
                if (status) status.textContent = `Threshold applied (${min.toFixed(3)} - ${max.toFixed(3)})`;
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
            if (result.overlay_url) {
                this.showOverlayOnMap(resultId, result.overlay_url, result.overlay_meta);
            }
            if (resultItem) {
                resultItem.dataset.visible = 'true';
                const status = resultItem.querySelector('.sa-result-status');
                if (status) status.textContent = `${result.name} — Click to hide`;
            }
        }
    }

    showOverlayOnMap(layerId, overlayUrl, meta) {
        this.removeOverlayFromMap(layerId);

        if (this.isLocalMode()) {
            const li = this.platform.localImage;
            if (li) {
                li.showLocalAnalysisLayer(layerId, overlayUrl, li.preloadedSize?.width, li.preloadedSize?.height);
                this.overlayLayers[layerId] = true;
            }
        } else if (window.mapManager) {
            window.mapManager.showAnalysisLayer(layerId, overlayUrl, layerId);
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
}
