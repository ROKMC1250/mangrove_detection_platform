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
            { id: 'rxd', name: 'RXD' },
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

        // Events
        document.getElementById('td-algo').onchange = (e) => {
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

            if (!response.ok) throw new Error((await response.json()).detail || 'Failed');

            this.currentResult = await response.json();
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
            this.platform.hideLoading();
            this.platform.showNotification(`Failed: ${error.message}`, 'error');
        }
    }

    // ========== SCORE STATE ==========
    showScoreUI() {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;

        item.querySelector('.td-ui')?.remove();
        
        // Hide the analysis-info section
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const r = this.currentResult;
        const ui = document.createElement('div');
        ui.className = 'td-ui td-score';
        ui.innerHTML = `
            <div class="td-colorbar">
                <div class="td-colorbar-gradient"></div>
                <div class="td-colorbar-labels">
                    <span>${r.min_val.toFixed(2)}</span>
                    <span>${r.max_val.toFixed(2)}</span>
                </div>
            </div>
            <div class="td-threshold-row">
                <span>Threshold:</span>
                <input type="range" id="td-thresh" 
                    min="${r.min_val}" max="${r.max_val}" 
                    step="${(r.max_val - r.min_val) / 100}" 
                    value="${r.threshold}">
                <span id="td-thresh-val">${r.threshold.toFixed(3)}</span>
            </div>
            <div class="td-actions">
                <button id="td-apply" class="td-btn primary">Apply</button>
                <button id="td-cancel" class="td-btn">Cancel</button>
            </div>
        `;

        item.appendChild(ui);
        item.classList.add('expanded');
        this.updateStatus(`${r.algorithm} score map`);

        // Events
        document.getElementById('td-thresh').oninput = (e) => {
            document.getElementById('td-thresh-val').textContent = parseFloat(e.target.value).toFixed(3);
        };

        document.getElementById('td-apply').onclick = (e) => {
            e.stopPropagation();
            this.applyThreshold();
        };

        document.getElementById('td-cancel').onclick = (e) => {
            e.stopPropagation();
            this.cancelToSetup();
        };
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
        } else {
            this.showScoreMap();
        }
    }

    async applyThreshold() {
        const threshold = parseFloat(document.getElementById('td-thresh')?.value);
        const imageInfo = this.getImageInfo();
        if (!imageInfo || !this.currentDetectionId) return;

        this.platform.showLoading('Applying threshold...');

        try {
            const response = await fetch('/api/target-detection/apply-threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    detection_id: this.currentDetectionId,
                    threshold: threshold,
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
        this.hideScoreMap();
        this.hideMask();
        this.hideChartsBox();
        this.state = 'setup';
        this.currentResult = null;
        this.currentDetectionId = null;
        this.showSetupUI();
    }

    // ========== MASK STATE ==========
    showMaskUI() {
        const item = document.querySelector('.analysis-item.target-detection-option');
        if (!item) return;

        item.querySelector('.td-ui')?.remove();
        
        // Hide the analysis-info section
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const r = this.currentResult;
        const ui = document.createElement('div');
        ui.className = 'td-ui td-mask';
        ui.innerHTML = `
            <div class="td-mask-info">
                <span>${r.detected_pixels?.toLocaleString() || 0} pixels</span>
                <span>${r.detection_percentage || 0}%</span>
            </div>
            <button id="td-back" class="td-btn">← Back to Score</button>
        `;

        item.appendChild(ui);
        item.classList.add('expanded');
        this.updateStatus(`${r.detection_percentage}% detected`);

        document.getElementById('td-back').onclick = (e) => {
            e.stopPropagation();
            this.backToScore();
        };
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
        } else {
            this.showMask();
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
