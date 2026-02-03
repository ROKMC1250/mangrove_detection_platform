/**
 * Target Detection Controller
 * 
 * Flow:
 * 1. Click Target Detection → Show: [Algorithm] [📍 Select] [Run Detection]
 * 2. Run Detection → Show score map + colorbar + threshold bar
 *    - Toggle: click to show/hide score map
 * 3. Apply threshold → Show mask
 *    - Toggle: click to show/hide mask
 *    - Cancel: go back to step 2 (score map)
 */

class TargetDetectionController {
    constructor(platformController) {
        this.platform = platformController;
        
        // States: 'setup' | 'score' | 'mask'
        this.state = 'setup';
        this.targetSelectMode = false;
        this.targetPoints = [];
        this.selectedAlgorithm = 'SAM';
        this.currentDetectionId = null;
        this.currentResult = null;
        this.layerVisible = false;
        this.targetMarkers = [];
        
        this.algorithms = [
            { id: 'sam', name: 'SAM' },
            { id: 'ace', name: 'ACE' },
            { id: 'mf', name: 'MF' },
            { id: 'cem', name: 'CEM' }
        ];
        
        // All available bands
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
        this.selectedBands = null; // null = all bands
        
        this.handleMapClick = this.handleMapClick.bind(this);
    }

    getImageInfo() {
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
            this.platform.showNotification('Process an image first', 'warning');
            return;
        }

        const item = document.querySelector('.analysis-item.target-detection-option');
        const hasUI = item?.querySelector('.td-ui');

        if (this.state === 'setup') {
            // Toggle setup UI
            if (hasUI) {
                this.hideUI();
            } else {
                this.showSetupUI();
            }
        } else if (this.state === 'score') {
            // Toggle score map visibility
            this.toggleScoreMap();
        } else if (this.state === 'mask') {
            // Toggle mask visibility
            this.toggleMask();
        }
    }

    // ========== SETUP STATE ==========
    showSetupUI() {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;

        // Clear previous points and markers when showing setup UI
        this.clearMarkers();
        this.targetPoints = [];
        this.stopTargetSelection();

        // Remove existing UI
        item.querySelector('.td-ui')?.remove();

        // Hide the analysis-info section when showing UI
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const ui = document.createElement('div');
        ui.className = 'td-ui td-setup';
        ui.innerHTML = `
            <div class="td-row">
                <select id="td-algo">
                    ${this.algorithms.map(a => `<option value="${a.id}">${a.name}</option>`).join('')}
                </select>
                <button id="td-pick" class="td-btn">📍 <span id="td-count">0</span></button>
                <button id="td-bands-toggle" class="td-btn">🔧 Bands</button>
                <button id="td-run" class="td-btn primary" disabled>Run</button>
            </div>
            <div id="td-bands-panel" class="td-bands-panel" style="display:none;">
                <div class="td-bands-grid">
                    ${this.allBands.map(b => `
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
        `;

        item.appendChild(ui);
        item.classList.add('expanded');

        // Initialize algorithm from dropdown
        const algoSelect = document.getElementById('td-algo');
        this.selectedAlgorithm = algoSelect.value.toUpperCase();
        
        // Events
        algoSelect.onchange = (e) => {
            this.selectedAlgorithm = e.target.value.toUpperCase();
        };

        document.getElementById('td-pick').onclick = (e) => {
            e.stopPropagation();
            this.toggleTargetSelection();
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

        document.getElementById('td-run').onclick = (e) => {
            e.stopPropagation();
            this.runDetection();
        };
    }

    getSelectedBands() {
        const checkboxes = document.querySelectorAll('#td-bands-panel input[type="checkbox"]:checked');
        if (checkboxes.length === 0 || checkboxes.length === this.allBands.length) {
            return null; // All bands
        }
        return Array.from(checkboxes).map(cb => parseInt(cb.value));
    }

    toggleTargetSelection() {
        if (this.targetSelectMode) {
            this.stopTargetSelection();
        } else {
            this.startTargetSelection();
        }
    }

    startTargetSelection() {
        this.targetSelectMode = true;
        document.getElementById('td-pick')?.classList.add('active');
        if (window.mapManager?.map) {
            window.mapManager.map.getContainer().style.cursor = 'crosshair';
            window.mapManager.map.on('click', this.handleMapClick);
        }
        this.platform.showNotification('Click map to select target', 'info');
    }

    stopTargetSelection() {
        this.targetSelectMode = false;
        document.getElementById('td-pick')?.classList.remove('active');
        if (window.mapManager?.map) {
            window.mapManager.map.getContainer().style.cursor = '';
            window.mapManager.map.off('click', this.handleMapClick);
        }
    }

    handleMapClick(e) {
        if (!this.targetSelectMode) return;
        this.targetPoints.push({ lat: e.latlng.lat, lng: e.latlng.lng });
        
        // Add marker with better visibility
        if (window.mapManager && window.L) {
            const pointNum = this.targetPoints.length;
            
            // Create a more visible marker
            const marker = L.circleMarker([e.latlng.lat, e.latlng.lng], {
                radius: 10,
                fillColor: '#ff0000',
                color: '#ffffff',
                weight: 3,
                opacity: 1,
                fillOpacity: 1
            }).addTo(window.mapManager.map);
            
            // Add number label
            const label = L.marker([e.latlng.lat, e.latlng.lng], {
                icon: L.divIcon({
                    className: 'td-marker-number',
                    html: `<span>${pointNum}</span>`,
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                })
            }).addTo(window.mapManager.map);
            
            this.targetMarkers.push({ marker, label });
        }
        
        // Update count
        const countEl = document.getElementById('td-count');
        if (countEl) countEl.textContent = this.targetPoints.length;
        
        // Enable run button
        const runBtn = document.getElementById('td-run');
        if (runBtn) runBtn.disabled = false;
    }

    clearMarkers() {
        this.targetMarkers.forEach(item => {
            if (item.marker) window.mapManager?.map?.removeLayer(item.marker);
            if (item.label) window.mapManager?.map?.removeLayer(item.label);
        });
        this.targetMarkers = [];
    }

    async runDetection() {
        if (this.targetPoints.length === 0) return;

        const imageInfo = this.getImageInfo();
        if (!imageInfo) return;

        this.stopTargetSelection();
        
        // Clear any previous detection layers before starting new one
        this.cleanupPreviousDetection();
        
        this.platform.showLoading(`Running ${this.selectedAlgorithm}...`);

        try {
            const selectedBands = this.getSelectedBands();
            const response = await fetch('/api/target-detection/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: imageInfo.id,
                    bbox: imageInfo.bbox,
                    geometry: imageInfo.geometry,
                    target_points: this.targetPoints,
                    algorithm: this.selectedAlgorithm,
                    auto_threshold: true,
                    selected_bands: selectedBands
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Detection failed' }));
                throw new Error(errorData.detail || `Detection failed with status ${response.status}`);
            }

            const result = await response.json();
            
            // Validate result has required fields
            if (!result || !result.detection_id) {
                throw new Error('Invalid response from server');
            }

            this.currentResult = result;
            this.currentDetectionId = this.currentResult.detection_id;
            
            this.clearMarkers();
            this.targetPoints = [];
            
            // Show score map
            this.state = 'score';
            this.showScoreUI();
            await this.showScoreMap();
            
            // Show charts box below
            if (this.currentResult.charts) {
                this.showChartsBox();
            }
            
            this.platform.hideLoading();

        } catch (error) {
            console.error('Target detection error:', error);
            this.platform.hideLoading();
            this.platform.showNotification(`Detection failed: ${error.message}`, 'error');
            
            // Reset to setup state on failure
            this.cancelToSetup();
        }
    }
    
    cleanupPreviousDetection() {
        // Remove all previous detection layers from the map
        if (window.mapManager) {
            window.mapManager.hideAnalysisLayer('target-detection');
            window.mapManager.hideAnalysisLayer('target-detection-mask');
            window.mapManager.hideAnalysisLayer('target-detection-score');
        }
        
        // Reset result state
        this.currentResult = null;
        this.currentDetectionId = null;
        this.layerVisible = false;
        
        // Hide any existing charts
        this.hideChartsBox();
    }

    // ========== SCORE STATE ==========
    showScoreUI() {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;

        item.querySelector('.td-ui')?.remove();
        
        // Update thumbnail if we have a preview
        const thumbEl = item.querySelector('.analysis-thumbnail');
        if (thumbEl && this.currentResult?.detection_result?.preview_url) {
            thumbEl.classList.remove('custom-placeholder');
            thumbEl.style.background = '';
            thumbEl.innerHTML = '';
            const img = document.createElement('img');
            img.src = this.currentResult.detection_result.preview_url;
            img.alt = 'Detection Result';
            img.style.cssText = 'width:100%;height:100%;object-fit:cover;border-radius:4px;';
            thumbEl.appendChild(img);
        }
        
        // Hide the analysis-info section
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const r = this.currentResult;
        const minVal = r.min_val;
        const maxVal = r.max_val;
        
        const ui = document.createElement('div');
        ui.className = 'td-ui td-score';
        ui.innerHTML = `
            <div class="td-colorbar-container">
                <div class="colorbar-with-threshold">
                    <div class="colorbar-track td-track">
                        <div class="colorbar-gradient td-gradient"></div>
                        <div class="colorbar-selection"></div>
                        <div class="colorbar-handle min-handle"></div>
                        <div class="colorbar-handle max-handle"></div>
                    </div>
                </div>
                <div class="colorbar-values">
                    <input type="number" class="colorbar-min-input" id="td-min" value="${minVal.toFixed(3)}" step="0.001">
                    <div class="colorbar-buttons">
                        <button id="td-apply" class="colorbar-apply-btn">Apply</button>
                        <button id="td-cancel-threshold" class="colorbar-cancel-btn" style="display:none;">Cancel</button>
                    </div>
                    <input type="number" class="colorbar-max-input" id="td-max" value="${maxVal.toFixed(3)}" step="0.001">
                </div>
            </div>
            <button id="td-try-another" class="td-btn-secondary">🔄 Try another model</button>
        `;

        item.appendChild(ui);
        item.classList.add('expanded');
        this.updateStatus(this.layerVisible ? `${r.algorithm} - Click to hide` : `${r.algorithm} - Click to show`);

        // Store references
        const track = ui.querySelector('.colorbar-track');
        const minHandle = ui.querySelector('.min-handle');
        const maxHandle = ui.querySelector('.max-handle');
        const selection = ui.querySelector('.colorbar-selection');
        const minInput = document.getElementById('td-min');
        const maxInput = document.getElementById('td-max');
        const cancelBtn = document.getElementById('td-cancel-threshold');

        // Store current values in closure
        let currentMin = minVal;
        let currentMax = maxVal;

        // Update handle positions and selection
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

        // Initialize positions after DOM render
        setTimeout(updateUI, 50);

        // Drag handling
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

        // Input change handlers
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

        // Store current values for apply
        this._currentThresholdMin = () => currentMin;
        this._currentThresholdMax = () => currentMax;

        // Apply button - show binary mask
        document.getElementById('td-apply').onclick = (e) => {
            e.stopPropagation();
            this.applyThreshold();
            if (cancelBtn) cancelBtn.style.display = 'inline-block';
        };

        // Cancel threshold button - restore score map
        if (cancelBtn) {
            cancelBtn.onclick = (e) => {
                e.stopPropagation();
                this.cancelThresholdToScore();
                // Reset handles
                currentMin = minVal;
                currentMax = maxVal;
                minInput.value = minVal.toFixed(3);
                maxInput.value = maxVal.toFixed(3);
                updateUI();
                cancelBtn.style.display = 'none';
            };
        }

        // Try another model button
        document.getElementById('td-try-another').onclick = (e) => {
            e.stopPropagation();
            this.cancelToSetup();
        };

        // Prevent colorbar clicks from toggling
        ui.querySelector('.td-colorbar-container').addEventListener('click', (e) => e.stopPropagation());
    }
    
    // Cancel threshold and go back to score map view
    async cancelThresholdToScore() {
        if (!this.currentResult?.detection_result?.overlay_url) return;
        
        // Hide mask, show score map
        window.mapManager?.hideAnalysisLayer('target-detection-mask');
        await window.mapManager?.showAnalysisLayer(
            'target-detection',
            this.currentResult.detection_result.overlay_url,
            'Target Detection Score'
        );
        
        this.state = 'score';
        this.layerVisible = true;
        this.updateItemActive(true);
        this.updateStatus(`${this.currentResult.algorithm} - Click to hide`);
        this.platform.showNotification('Restored score map', 'info');
    }

    async showScoreMap() {
        if (!this.currentResult?.detection_result?.overlay_url) return;
        await window.mapManager?.showAnalysisLayer(
            'target-detection',
            this.currentResult.detection_result.overlay_url,
            'Target Detection Score'
        );
        this.layerVisible = true;
        this.updateItemActive(true);
    }

    hideScoreMap() {
        window.mapManager?.hideAnalysisLayer('target-detection');
        this.layerVisible = false;
        this.updateItemActive(false);
    }

    toggleScoreMap() {
        if (this.layerVisible) {
            this.hideScoreMap();
            this.updateStatus(`${this.currentResult?.algorithm || 'Score'} - Click to show`);
        } else {
            this.showScoreMap();
            this.updateStatus(`${this.currentResult?.algorithm || 'Score'} - Click to hide`);
        }
    }

    async applyThreshold() {
        // Use closure values if available, otherwise fall back to input values
        const minThreshold = this._currentThresholdMin ? this._currentThresholdMin() : parseFloat(document.getElementById('td-min')?.value);
        const maxThreshold = this._currentThresholdMax ? this._currentThresholdMax() : parseFloat(document.getElementById('td-max')?.value);
        const imageInfo = this.getImageInfo();
        if (!imageInfo || !this.currentDetectionId) return;

        this.platform.showLoading('Applying threshold...');

        try {
            const response = await fetch('/api/target-detection/apply-threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    detection_id: this.currentDetectionId,
                    min_threshold: minThreshold,
                    max_threshold: maxThreshold,
                    bbox: imageInfo.bbox
                })
            });

            if (!response.ok) throw new Error('Failed');

            const result = await response.json();
            this.currentResult.mask_result = result.mask_result;
            this.currentResult.detected_pixels = result.detected_pixels;
            this.currentResult.detection_percentage = result.detection_percentage;
            
            // Switch to mask state
            this.state = 'mask';
            this.showMaskUI();
            await this.showMask();
            
            this.platform.hideLoading();
            this.platform.showNotification(
                `${result.detection_percentage}% detected`,
                'success'
            );

        } catch (error) {
            this.platform.hideLoading();
            this.platform.showNotification('Failed to apply threshold', 'error');
        }
    }

    cancelToSetup() {
        // Clean up all detection layers
        this.cleanupPreviousDetection();
        
        this.hideScoreMap();
        this.hideMask();
        this.hideChartsBox();
        this.clearMarkers();
        
        this.state = 'setup';
        this.currentResult = null;
        this.currentDetectionId = null;
        this.targetPoints = [];
        this.layerVisible = false;
        
        this.showSetupUI();
    }

    // ========== MASK STATE ==========
    showMaskUI() {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;

        item.querySelector('.td-ui')?.remove();
        
        // Update thumbnail if we have a mask preview
        const thumbEl = item.querySelector('.analysis-thumbnail');
        if (thumbEl && this.currentResult?.mask_result?.preview_url) {
            thumbEl.classList.remove('custom-placeholder');
            thumbEl.style.background = '';
            thumbEl.innerHTML = '';
            const img = document.createElement('img');
            img.src = this.currentResult.mask_result.preview_url;
            img.alt = 'Binary Mask';
            img.style.cssText = 'width:100%;height:100%;object-fit:cover;border-radius:4px;';
            thumbEl.appendChild(img);
        }
        
        // Hide the analysis-info section
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const r = this.currentResult;
        const minVal = r.min_val;
        const maxVal = r.max_val;
        const appliedMin = this._currentThresholdMin ? this._currentThresholdMin() : minVal;
        const appliedMax = this._currentThresholdMax ? this._currentThresholdMax() : maxVal;
        
        const ui = document.createElement('div');
        ui.className = 'td-ui td-mask';
        ui.innerHTML = `
            <div class="td-mask-info">
                <span>🎯 ${r.detected_pixels?.toLocaleString() || 0} pixels</span>
                <span>${r.detection_percentage || 0}%</span>
            </div>
            <div class="td-colorbar-container">
                <div class="colorbar-with-threshold">
                    <div class="colorbar-track td-track">
                        <div class="colorbar-gradient td-gradient"></div>
                        <div class="colorbar-selection"></div>
                        <div class="colorbar-handle min-handle"></div>
                        <div class="colorbar-handle max-handle"></div>
                    </div>
                </div>
                <div class="colorbar-values">
                    <input type="number" class="colorbar-min-input" id="td-min-mask" value="${appliedMin.toFixed(3)}" step="0.001">
                    <div class="colorbar-buttons">
                        <button id="td-apply-mask" class="colorbar-apply-btn">Apply</button>
                        <button id="td-cancel-mask" class="colorbar-cancel-btn">Cancel</button>
                    </div>
                    <input type="number" class="colorbar-max-input" id="td-max-mask" value="${appliedMax.toFixed(3)}" step="0.001">
                </div>
            </div>
            <button id="td-try-another-mask" class="td-btn-secondary">🔄 Try another model</button>
        `;

        item.appendChild(ui);
        item.classList.add('expanded');
        this.updateStatus(this.layerVisible ? 'Binary mask - Click to hide' : 'Binary mask - Click to show');

        // Store references
        const track = ui.querySelector('.colorbar-track');
        const minHandle = ui.querySelector('.min-handle');
        const maxHandle = ui.querySelector('.max-handle');
        const selection = ui.querySelector('.colorbar-selection');
        const minInput = document.getElementById('td-min-mask');
        const maxInput = document.getElementById('td-max-mask');

        // Store current values in closure
        let currentMin = appliedMin;
        let currentMax = appliedMax;

        // Update handle positions and selection
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

        // Initialize positions after DOM render
        setTimeout(updateUI, 50);

        // Drag handling
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

        // Input change handlers
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

        // Update threshold values for apply
        this._currentThresholdMin = () => currentMin;
        this._currentThresholdMax = () => currentMax;

        // Apply button - apply new threshold
        document.getElementById('td-apply-mask').onclick = (e) => {
            e.stopPropagation();
            this.applyThreshold();
        };

        // Cancel button - go back to score map
        document.getElementById('td-cancel-mask').onclick = (e) => {
            e.stopPropagation();
            this.backToScore();
        };

        // Try another model button
        document.getElementById('td-try-another-mask').onclick = (e) => {
            e.stopPropagation();
            this.cancelToSetup();
        };

        // Prevent colorbar clicks from toggling
        ui.querySelector('.td-colorbar-container').addEventListener('click', (e) => e.stopPropagation());
    }

    async showMask() {
        if (!this.currentResult?.mask_result?.overlay_url) return;
        window.mapManager?.hideAnalysisLayer('target-detection');
        await window.mapManager?.showAnalysisLayer(
            'target-detection-mask',
            this.currentResult.mask_result.overlay_url,
            'Target Detection Mask'
        );
        this.layerVisible = true;
        this.updateItemActive(true);
    }

    hideMask() {
        window.mapManager?.hideAnalysisLayer('target-detection-mask');
        this.layerVisible = false;
        this.updateItemActive(false);
    }

    toggleMask() {
        if (this.layerVisible) {
            this.hideMask();
            this.updateStatus('Binary mask - Click to show');
        } else {
            this.showMask();
            this.updateStatus('Binary mask - Click to hide');
        }
    }

    async backToScore() {
        this.hideMask();
        this.state = 'score';
        this.showScoreUI();
        await this.showScoreMap();
    }

    // ========== CHARTS BOXES ==========
    showChartsBox() {
        // Remove existing charts
        this.hideChartsBox();
        
        if (!this.currentResult?.charts) return;
        
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;
        
        // Create separate box for each chart
        const charts = [
            { 
                id: 'td-chart-spectrum', 
                label: 'Target vs Background', 
                src: this.currentResult.charts.spectrum_comparison 
            },
            { 
                id: 'td-chart-score', 
                label: 'Score Distribution', 
                src: this.currentResult.charts.score_distribution 
            }
        ];
        
        let insertAfter = item;
        charts.forEach(chart => {
            const chartBox = document.createElement('div');
            chartBox.className = 'td-chart-box';
            chartBox.id = chart.id;
            chartBox.innerHTML = `
                <div class="td-chart-box-label">${chart.label}</div>
                <img src="${chart.src}" alt="${chart.label}">
            `;
            
            insertAfter.insertAdjacentElement('afterend', chartBox);
            insertAfter = chartBox;
            
            // Click to show full image
            chartBox.querySelector('img').onclick = (e) => {
                e.stopPropagation();
                this.showChartModal(chart.src, chart.label);
            };
        });
    }
    
    showChartModal(src, title) {
        // Remove existing modal
        document.querySelector('.td-chart-modal')?.remove();
        
        const modal = document.createElement('div');
        modal.className = 'td-chart-modal';
        modal.innerHTML = `
            <div class="td-chart-modal-content">
                <div class="td-chart-modal-header">
                    <span>${title}</span>
                    <button class="td-chart-modal-close">✕</button>
                </div>
                <img src="${src}" alt="${title}">
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close handlers
        modal.querySelector('.td-chart-modal-close').onclick = () => modal.remove();
        modal.onclick = (e) => {
            if (e.target === modal) modal.remove();
        };
    }
    
    hideChartsBox() {
        document.querySelectorAll('.td-chart-box').forEach(el => el.remove());
    }

    // ========== HELPERS ==========
    updateStatus(text) {
        const item = document.querySelector('.analysis-item.target-detection-option');
        const statusEl = item?.querySelector('.analysis-status');
        if (statusEl) {
            statusEl.textContent = text;
            statusEl.className = 'analysis-status ' + (this.state !== 'setup' ? 'active' : 'inactive');
        }
    }

    updateItemActive(active) {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (item) {
            item.classList.toggle('active', active);
        }
    }

    // Reset everything
    reset() {
        this.stopTargetSelection();
        this.clearMarkers();
        
        // Clean up all detection layers from the map
        if (window.mapManager) {
            window.mapManager.hideAnalysisLayer('target-detection');
            window.mapManager.hideAnalysisLayer('target-detection-mask');
            window.mapManager.hideAnalysisLayer('target-detection-score');
        }
        
        this.hideScoreMap();
        this.hideMask();
        this.hideChartsBox();
        this.state = 'setup';
        this.targetPoints = [];
        this.currentResult = null;
        this.currentDetectionId = null;
        this.layerVisible = false;
        
        const item = document.querySelector('.analysis-item.target-detection-option');
        item?.querySelector('.td-ui')?.remove();
        item?.classList.remove('expanded', 'active');
        
        // Show the analysis-info section again
        const infoEl = item?.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
        
        console.log('🧹 Target detection reset complete');
    }
    
    // Hide UI and show info
    hideUI() {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;
        
        item.querySelector('.td-ui')?.remove();
        item.classList.remove('expanded');
        
        // Show the analysis-info section again
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }
}

if (typeof window !== 'undefined') {
    window.TargetDetectionController = TargetDetectionController;
}
