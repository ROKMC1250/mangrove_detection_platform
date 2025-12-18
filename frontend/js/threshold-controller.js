/**
 * Threshold Controller Module
 * Handles threshold-based filtering and pixel value inspection
 */

class ThresholdController {
    constructor(platformController) {
        this.platform = platformController;
        this.activeThresholdModelId = null;
        this.inspectionEnabled = false;
        this.inspectionModelId = null;
        this.pixelPopup = null;
    }

    /**
     * Toggle pixel value inspection mode
     */
    togglePixelInspection(modelId, result, btnElement) {
        if (this.inspectionEnabled && this.inspectionModelId === modelId) {
            this.disablePixelInspection();
            btnElement.classList.remove('active');
        } else {
            this.enablePixelInspection(modelId, result);
            
            // Update button states
            document.querySelectorAll('.btn-inspect').forEach(btn => {
                btn.classList.remove('active');
            });
            btnElement.classList.add('active');
        }
    }

    /**
     * Enable pixel value inspection
     */
    enablePixelInspection(modelId, result) {
        this.inspectionEnabled = true;
        this.inspectionModelId = modelId;
        this.inspectionResult = result;

        if (window.mapManager && window.mapManager.map) {
            window.mapManager.map.getContainer().style.cursor = 'crosshair';
            window.mapManager.map.on('click', this.handleMapClick.bind(this));
        }

        this.platform.showNotification('Click on the map to inspect pixel values', 'info');
    }

    /**
     * Disable pixel value inspection
     */
    disablePixelInspection() {
        this.inspectionEnabled = false;
        this.inspectionModelId = null;
        this.inspectionResult = null;
        this.hidePixelPopup();

        if (window.mapManager && window.mapManager.map) {
            window.mapManager.map.getContainer().style.cursor = '';
            window.mapManager.map.off('click', this.handleMapClick.bind(this));
        }
    }

    /**
     * Handle map click for pixel inspection
     */
    async handleMapClick(e) {
        if (!this.inspectionEnabled) return;

        const latlng = e.latlng;
        const selectedImage = this.platform.imageSearch?.getSelectedImage();
        
        if (!selectedImage) {
            this.platform.showNotification('No image selected', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/get-pixel-value', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: selectedImage.id,
                    lat: latlng.lat,
                    lng: latlng.lng,
                    model_id: this.inspectionModelId
                })
            });

            const data = await response.json();
            
            if (data.error) {
                this.showPixelPopup(latlng, data.error, null);
            } else {
                this.showPixelPopup(latlng, data.value, this.inspectionResult?.colormap);
            }

        } catch (error) {
            console.error('Pixel value error:', error);
            this.showPixelPopup(latlng, 'Error', null);
        }
    }

    /**
     * Show pixel value popup
     */
    showPixelPopup(latlng, value, colormap) {
        this.hidePixelPopup();

        const label = colormap?.label || 'Value';
        const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;

        const content = `
            <div class="pixel-popup">
                <div class="pixel-label">${label}</div>
                <div class="pixel-value">${formattedValue}</div>
                <div class="pixel-coords">${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}</div>
            </div>
        `;

        if (window.mapManager && window.L) {
            this.pixelPopup = L.popup()
                .setLatLng(latlng)
                .setContent(content)
                .openOn(window.mapManager.map);
        }
    }

    /**
     * Hide pixel value popup
     */
    hidePixelPopup() {
        if (this.pixelPopup && window.mapManager?.map) {
            window.mapManager.map.closePopup(this.pixelPopup);
            this.pixelPopup = null;
        }
    }

    /**
     * Toggle threshold control
     */
    async toggleThresholdControl(modelId, result, btnElement) {
        if (this.activeThresholdModelId === modelId) {
            this.disableThresholdControl(btnElement);
        } else {
            await this.enableThresholdControl(modelId, result, btnElement);
        }
    }

    /**
     * Enable threshold control
     */
    async enableThresholdControl(modelId, result, btnElement) {
        // Disable any existing threshold control
        if (this.activeThresholdModelId) {
            const prevBtn = document.querySelector(`.analysis-item[data-model-id="${this.activeThresholdModelId}"] .btn-threshold`);
            if (prevBtn) this.disableThresholdControl(prevBtn);
        }

        this.activeThresholdModelId = modelId;
        btnElement.classList.add('active');

        const analysisItem = btnElement.closest('.analysis-item');
        this.createThresholdUI(analysisItem, modelId, result);
    }

    /**
     * Disable threshold control
     */
    disableThresholdControl(btnElement) {
        const analysisItem = btnElement.closest('.analysis-item');
        const thresholdUI = analysisItem?.querySelector('.threshold-controls');
        if (thresholdUI) thresholdUI.remove();

        btnElement.classList.remove('active');
        this.activeThresholdModelId = null;
    }

    /**
     * Create threshold UI
     */
    createThresholdUI(analysisItem, modelId, result) {
        // Remove existing UI
        const existing = analysisItem.querySelector('.threshold-controls');
        if (existing) existing.remove();

        const colormap = result.colormap || { min_val: -1, max_val: 1 };
        const minVal = colormap.min_val;
        const maxVal = colormap.max_val;
        const step = (maxVal - minVal) / 100;

        const thresholdUI = document.createElement('div');
        thresholdUI.className = 'threshold-controls';
        thresholdUI.innerHTML = `
            <div class="threshold-header">
                <span>Threshold Range</span>
                <button class="btn-small btn-close-threshold">✕</button>
            </div>
            <div class="threshold-inputs">
                <div class="threshold-input-group">
                    <label>Min</label>
                    <input type="number" id="threshold-min-${modelId}" 
                           value="${minVal.toFixed(3)}" step="${step.toFixed(4)}" 
                           min="${minVal}" max="${maxVal}">
                </div>
                <div class="threshold-input-group">
                    <label>Max</label>
                    <input type="number" id="threshold-max-${modelId}" 
                           value="${maxVal.toFixed(3)}" step="${step.toFixed(4)}" 
                           min="${minVal}" max="${maxVal}">
                </div>
            </div>
            <div class="threshold-slider">
                <input type="range" id="threshold-slider-min-${modelId}" 
                       min="${minVal}" max="${maxVal}" step="${step}" value="${minVal}">
                <input type="range" id="threshold-slider-max-${modelId}" 
                       min="${minVal}" max="${maxVal}" step="${step}" value="${maxVal}">
            </div>
            <button class="btn btn-primary btn-apply-threshold" id="apply-threshold-${modelId}">
                Apply Threshold
            </button>
        `;

        analysisItem.appendChild(thresholdUI);

        // Event listeners
        const closeBtn = thresholdUI.querySelector('.btn-close-threshold');
        closeBtn.addEventListener('click', () => {
            const thresholdBtn = analysisItem.querySelector('.btn-threshold');
            this.disableThresholdControl(thresholdBtn);
        });

        const minInput = thresholdUI.querySelector(`#threshold-min-${modelId}`);
        const maxInput = thresholdUI.querySelector(`#threshold-max-${modelId}`);
        const minSlider = thresholdUI.querySelector(`#threshold-slider-min-${modelId}`);
        const maxSlider = thresholdUI.querySelector(`#threshold-slider-max-${modelId}`);

        // Sync inputs and sliders
        minInput.addEventListener('input', () => minSlider.value = minInput.value);
        maxInput.addEventListener('input', () => maxSlider.value = maxInput.value);
        minSlider.addEventListener('input', () => minInput.value = parseFloat(minSlider.value).toFixed(3));
        maxSlider.addEventListener('input', () => maxInput.value = parseFloat(maxSlider.value).toFixed(3));

        // Apply button
        const applyBtn = thresholdUI.querySelector(`#apply-threshold-${modelId}`);
        applyBtn.addEventListener('click', () => {
            this.applyThresholdRange(
                modelId, 
                result, 
                parseFloat(minInput.value), 
                parseFloat(maxInput.value)
            );
        });
    }

    /**
     * Apply threshold range
     */
    async applyThresholdRange(modelId, result, minThreshold, maxThreshold) {
        const selectedImage = this.platform.imageSearch?.getSelectedImage();
        if (!selectedImage) {
            this.platform.showNotification('No image selected', 'warning');
            return;
        }

        this.platform.showLoading('Applying threshold...');

        try {
            const response = await fetch('/api/apply-threshold-range', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: selectedImage.id,
                    model_id: modelId,
                    min_threshold: minThreshold,
                    max_threshold: maxThreshold,
                    colormap: result.colormap || {}
                })
            });

            if (!response.ok) throw new Error(`Failed: ${response.status}`);
            const data = await response.json();

            // Update the overlay
            if (window.mapManager && data.overlay_url) {
                await window.mapManager.showAnalysisLayer(
                    `${modelId}-threshold`,
                    data.overlay_url,
                    data.name
                );
            }

            this.platform.hideLoading();
            this.platform.showNotification('Threshold applied', 'success');

        } catch (error) {
            console.error('Threshold error:', error);
            this.platform.hideLoading();
            this.platform.showNotification(`Failed: ${error.message}`, 'error');
        }
    }

    /**
     * Disable all active controls
     */
    disableAll() {
        this.disablePixelInspection();
        
        document.querySelectorAll('.btn-inspect.active').forEach(btn => {
            btn.classList.remove('active');
        });
        
        document.querySelectorAll('.btn-threshold.active').forEach(btn => {
            this.disableThresholdControl(btn);
        });
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.ThresholdController = ThresholdController;
}

