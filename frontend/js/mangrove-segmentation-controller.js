/**
 * Mangrove Segmentation Controller
 *
 * Flow:
 * 1. Click Mangrove Segmentation item -> Show: [TTA toggle] [Run]
 * 2. Click Run -> API call -> returns probability map with colorbar
 * 3. Adjust threshold (min/max handles) -> Apply -> binary mask overlay
 * 4. Cancel -> restore probability map
 */

class MangroveSegmentationController {
    constructor(platformController) {
        this.platform = platformController;
        this.results = [];
        this.overlayLayers = {};
    }

    getImageInfo() {
        const imageId = this.platform.selectedImageId;
        if (!imageId) return null;
        return {
            id: imageId,
            bbox: this.platform.processedBbox || window.mapManager?.getCurrentBounds(),
            geometry: this.platform.processedGeometry || window.mapManager?.getCurrentGeoJSON()?.geometry
        };
    }

    handleItemClick(item) {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification('Process an image first', 'warning');
            return;
        }

        this.itemEl = item || document.querySelector('.analysis-item.mangrove-segmentation-option');
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
                <label class="ms-tta-label">
                    <input type="checkbox" id="ms-tta"> TTA
                </label>
                <button id="ms-run" class="ms-btn primary">Run Segmentation</button>
            </div>
            <div class="ms-status" style="display:none;"></div>
            <div class="ms-results-list"></div>
        `;

        this.itemEl.appendChild(ui);
        this.itemEl.classList.add('expanded');

        // Re-render existing results
        this.results.forEach(r => this._renderResultItem(r));

        document.getElementById('ms-run').onclick = (e) => {
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

        const runBtn = document.getElementById('ms-run');
        const statusEl = this.itemEl?.querySelector('.ms-status');

        if (runBtn) {
            runBtn.disabled = true;
            runBtn.textContent = 'Running...';
        }
        if (statusEl) {
            statusEl.style.display = 'block';
            statusEl.textContent = 'Running mangrove segmentation... This may take a moment.';
            statusEl.className = 'ms-status running';
        }

        const useTta = document.getElementById('ms-tta')?.checked || false;

        try {
            const response = await fetch('/api/mangrove-segmentation/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: imageInfo.id,
                    bbox: imageInfo.bbox,
                    geometry: imageInfo.geometry,
                    use_tta: useTta,
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();

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

            this.results.push(resultEntry);
            this._renderResultItem(resultEntry);

            // Auto-show probability map overlay
            this._showOverlay(resultEntry.id, resultEntry.detection_result);

            if (statusEl) {
                statusEl.textContent = 'Segmentation complete!';
                statusEl.className = 'ms-status success';
            }

            this.platform.showNotification('Mangrove segmentation complete', 'success');

        } catch (err) {
            console.error('Mangrove segmentation failed:', err);
            if (statusEl) {
                statusEl.textContent = `Error: ${err.message}`;
                statusEl.className = 'ms-status error';
            }
            this.platform.showNotification(`Segmentation failed: ${err.message}`, 'error');
        } finally {
            if (runBtn) {
                runBtn.disabled = false;
                runBtn.textContent = 'Run Segmentation';
            }
        }
    }

    // ========== RESULT RENDERING ==========
    _renderResultItem(resultEntry) {
        const resultsList = this.itemEl?.querySelector('.ms-results-list');
        if (!resultsList) return;

        // Don't duplicate
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
                <img class="ms-result-thumb" src="${r.detection_result?.preview_url || ''}" alt="Mangrove Segmentation" />
                <div class="ms-result-info">
                    <span class="ms-result-name">Mangrove Probability</span>
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
        `;

        resultsList.appendChild(resultItem);

        // Toggle overlay on header click
        resultItem.querySelector('.ms-result-header').onclick = (e) => {
            if (e.target.closest('.ms-result-actions')) return;
            e.stopPropagation();
            this._toggleOverlay(r.id);
        };

        // Remove button
        resultItem.querySelector('.ms-result-remove-btn').onclick = (e) => {
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

        // Apply threshold -> show binary mask
        applyBtn.onclick = async (e) => {
            e.stopPropagation();
            await this._applyThreshold(resultEntry, currentMin, currentMax, resultItem);
        };

        // Cancel -> restore probability map
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
        };

        // Prevent colorbar clicks from toggling overlay
        resultItem.querySelector('.ms-result-colorbar').addEventListener('click', (e) => e.stopPropagation());
    }

    async _applyThreshold(resultEntry, min, max, resultItem) {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) return;

        try {
            const response = await fetch('/api/mangrove-segmentation/apply-threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    segmentation_id: resultEntry.id,
                    min_threshold: min,
                    max_threshold: max,
                    bbox: imageInfo.bbox,
                })
            });

            if (!response.ok) throw new Error('Failed to apply threshold');

            const result = await response.json();
            resultEntry.mask_result = result.mask_result;
            resultEntry.showingMask = true;

            // Show mask overlay
            this._showOverlay(resultEntry.id, result.mask_result);

            // Update UI
            if (resultItem) {
                const stats = resultItem.querySelector('.ms-result-stats');
                stats.style.display = 'flex';
                stats.querySelector('.ms-result-pixels').textContent = `${result.detected_pixels?.toLocaleString() || 0} px`;
                stats.querySelector('.ms-result-pct').textContent = `${result.detection_percentage || 0}%`;

                if (result.mask_result?.preview_url) {
                    resultItem.querySelector('.ms-result-thumb').src = result.mask_result.preview_url;
                }
                resultItem.querySelector('.ms-result-status').textContent = `Mask (${min.toFixed(3)} - ${max.toFixed(3)})`;
            }

            this.platform.showNotification(`Mangrove: ${result.detection_percentage}% detected`, 'success');
        } catch (error) {
            console.error('Threshold error:', error);
            this.platform.showNotification('Failed to apply threshold', 'error');
        }
    }

    // ========== OVERLAY MANAGEMENT ==========
    _showOverlay(resultId, overlayData) {
        if (!overlayData?.overlay_url) return;

        this._hideOverlay(resultId);

        const layerId = `ms-${resultId}`;

        if (window.mapManager) {
            window.mapManager.showAnalysisLayer(layerId, overlayData.overlay_url, 'Mangrove Segmentation');
            window.mapManager.outlineAOI();
        }

        this.overlayLayers[resultId] = layerId;

        const resultItem = document.querySelector(`.ms-result-item[data-result-id="${resultId}"]`);
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

        const resultItem = document.querySelector(`.ms-result-item[data-result-id="${resultId}"]`);
        if (resultItem) {
            resultItem.dataset.visible = 'false';
            const entry = this.results.find(r => r.id === resultId);
            if (entry) entry.visible = false;
        }
    }

    _toggleOverlay(resultId) {
        const entry = this.results.find(r => r.id === resultId);
        if (!entry) return;

        const resultItem = document.querySelector(`.ms-result-item[data-result-id="${resultId}"]`);
        const isVisible = resultItem?.dataset.visible === 'true';

        if (isVisible) {
            this._hideOverlay(resultId);
            if (resultItem) {
                resultItem.querySelector('.ms-result-status').textContent = 'Click to toggle overlay';
            }
        } else {
            const overlayData = entry.showingMask ? entry.mask_result : entry.detection_result;
            this._showOverlay(resultId, overlayData);
            if (resultItem) {
                resultItem.querySelector('.ms-result-status').textContent = 'Overlay active — click to hide';
            }
        }
    }

    _removeResult(resultId) {
        this._hideOverlay(resultId);

        this.results = this.results.filter(r => r.id !== resultId);

        const resultItem = document.querySelector(`.ms-result-item[data-result-id="${resultId}"]`);
        if (resultItem) resultItem.remove();
    }
}
