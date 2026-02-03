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
        this.originalOverlayUrl = null;
        this.originalModelName = null;
        
        // Bind methods for event handlers
        this._boundHandleMapClick = this._handleMapClickInternal.bind(this);
    }

    // ========== PIXEL VALUE INSPECTION ==========

    /**
     * Toggle pixel value inspection mode
     */
    togglePixelInspection(modelId, result, btnElement) {
        if (this.inspectionEnabled && this.inspectionModelId === modelId) {
            this.disablePixelInspection();
            btnElement.classList.remove('active');
        } else {
            this.enablePixelInspection(modelId, result, btnElement);
        }
    }

    /**
     * Enable pixel value inspection
     */
    enablePixelInspection(modelId, result, btnElement) {
        // Disable any previous inspection
        this.disablePixelInspection();
        
        this.inspectionEnabled = true;
        this.inspectionModelId = modelId;
        this.inspectionResult = result;

        // Update all inspect button states
        document.querySelectorAll('.pixel-value-btn, .btn-inspect').forEach(btn => {
            btn.classList.remove('active');
        });
        btnElement.classList.add('active');

        if (window.mapManager && window.mapManager.map) {
            window.mapManager.map.getContainer().style.cursor = 'crosshair';
            window.mapManager.map.on('click', this._boundHandleMapClick);
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

        // Remove active state from all inspect buttons
        document.querySelectorAll('.pixel-value-btn.active, .btn-inspect.active').forEach(btn => {
            btn.classList.remove('active');
        });

        if (window.mapManager && window.mapManager.map) {
            window.mapManager.map.getContainer().style.cursor = '';
            window.mapManager.map.off('click', this._boundHandleMapClick);
        }
    }

    /**
     * Internal map click handler for pixel inspection
     */
    async _handleMapClickInternal(e) {
        if (!this.inspectionEnabled) return;

        const latlng = e.latlng;
        
        // Get the selected image from image search
        let selectedImage = null;
        if (this.platform.imageSearch) {
            selectedImage = this.platform.imageSearch.getSelectedImage?.() || 
                           { id: this.platform.selectedImageId };
        } else {
            selectedImage = { id: this.platform.selectedImageId };
        }
        
        if (!selectedImage || !selectedImage.id) {
            this.platform.showNotification('No image selected', 'warning');
            return;
        }

        // Show loading popup
        this.showPixelPopup(latlng, 'Loading...', null);

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
                this.showPixelPopup(latlng, `Error: ${data.error}`, null);
            } else {
                this.showPixelPopup(latlng, data.value, this.inspectionResult?.colormap);
            }

        } catch (error) {
            console.error('Pixel value error:', error);
            this.showPixelPopup(latlng, 'Error fetching value', null);
        }
    }

    /**
     * Show pixel value popup
     */
    showPixelPopup(latlng, value, colormap) {
        this.hidePixelPopup();

        const label = colormap?.label || 'Value';
        let formattedValue = value;
        let valueClass = 'pixel-value';
        
        if (typeof value === 'number') {
            formattedValue = value.toFixed(4);
        } else if (typeof value === 'string') {
            // Handle special messages (not numbers)
            valueClass = 'pixel-value pixel-value-text';
        }

        const content = `
            <div class="pixel-popup">
                <div class="pixel-label">${label}</div>
                <div class="${valueClass}">${formattedValue}</div>
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

    // ========== THRESHOLD CONTROL ==========

    /**
     * Toggle threshold control
     */
    async toggleThresholdControl(modelId, result, btnElement) {
        if (this.activeThresholdModelId === modelId) {
            this.hideThresholdUI(btnElement);
        } else {
            await this.showThresholdUI(modelId, result, btnElement);
        }
    }

    /**
     * Show threshold UI below the colorbar
     */
    async showThresholdUI(modelId, result, btnElement) {
        // Hide any existing threshold UI
        if (this.activeThresholdModelId) {
            const prevBtn = document.querySelector(`.analysis-item[data-model-id="${this.activeThresholdModelId}"] .threshold-btn`);
            if (prevBtn) this.hideThresholdUI(prevBtn);
        }

        this.activeThresholdModelId = modelId;
        btnElement.classList.add('active');
        
        // Store original overlay URL
        const analysisItem = btnElement.closest('.analysis-item');
        this.originalOverlayUrl = result.overlay_url || analysisItem?.dataset.originalOverlayUrl;
        this.originalModelName = result.name || modelId;

        this.createThresholdUI(analysisItem, modelId, result);
    }

    /**
     * Hide threshold UI
     */
    hideThresholdUI(btnElement) {
        const analysisItem = btnElement?.closest('.analysis-item');
        const thresholdUI = analysisItem?.querySelector('.threshold-inline');
        if (thresholdUI) thresholdUI.remove();

        if (btnElement) btnElement.classList.remove('active');
        this.activeThresholdModelId = null;
    }

    /**
     * Create threshold UI - draggable handles on colormap
     */
    createThresholdUI(analysisItem, modelId, result) {
        // Remove existing UI
        const existing = analysisItem?.querySelector('.threshold-inline');
        if (existing) existing.remove();

        if (!analysisItem) return;

        const colormap = result.colormap || { min_val: -1, max_val: 1, name: 'RdYlGn' };
        const minVal = colormap.min_val;
        const maxVal = colormap.max_val;
        const colormapName = colormap.name || 'RdYlGn';

        // Get the gradient class from the existing colorbar
        const existingColorbar = analysisItem.querySelector('.colorbar-gradient');
        const gradientClass = existingColorbar ? existingColorbar.className.replace('colorbar-gradient', '').trim() : colormapName;

        const thresholdUI = document.createElement('div');
        thresholdUI.className = 'threshold-inline';
        thresholdUI.innerHTML = `
            <div class="threshold-colormap-slider">
                <div class="threshold-colormap-track">
                    <div class="threshold-colormap-gradient colorbar-gradient ${gradientClass}"></div>
                    <div class="threshold-selection"></div>
                    <div class="threshold-handle min-handle" data-handle="min"></div>
                    <div class="threshold-handle max-handle" data-handle="max"></div>
                </div>
            </div>
            <div class="threshold-values-row">
                <div class="threshold-value-group">
                    <span class="threshold-value-label">Min:</span>
                    <input type="number" class="threshold-input min-input" 
                           value="${minVal.toFixed(3)}" step="0.001" 
                           min="${minVal}" max="${maxVal}">
                </div>
                <div class="threshold-value-group">
                    <span class="threshold-value-label">Max:</span>
                    <input type="number" class="threshold-input max-input" 
                           value="${maxVal.toFixed(3)}" step="0.001" 
                           min="${minVal}" max="${maxVal}">
                </div>
            </div>
            <div class="threshold-actions">
                <button class="threshold-btn-apply">Apply</button>
                <button class="threshold-btn-cancel">Cancel</button>
            </div>
        `;

        // Insert after colorbar-container within analysis-info
        const colorbarContainer = analysisItem.querySelector('.colorbar-container');
        if (colorbarContainer) {
            colorbarContainer.after(thresholdUI);
        } else {
            const analysisInfo = analysisItem.querySelector('.analysis-info');
            if (analysisInfo) {
                analysisInfo.appendChild(thresholdUI);
            } else {
                analysisItem.appendChild(thresholdUI);
            }
        }

        // Store references
        const track = thresholdUI.querySelector('.threshold-colormap-track');
        const minHandle = thresholdUI.querySelector('.min-handle');
        const maxHandle = thresholdUI.querySelector('.max-handle');
        const selection = thresholdUI.querySelector('.threshold-selection');
        const minInput = thresholdUI.querySelector('.min-input');
        const maxInput = thresholdUI.querySelector('.max-input');
        const applyBtn = thresholdUI.querySelector('.threshold-btn-apply');
        const cancelBtn = thresholdUI.querySelector('.threshold-btn-cancel');

        // Current values
        let currentMin = minVal;
        let currentMax = maxVal;

        // Update handle positions and selection
        const updateUI = () => {
            const trackWidth = track.offsetWidth;
            const range = maxVal - minVal;
            const minPos = ((currentMin - minVal) / range) * trackWidth;
            const maxPos = ((currentMax - minVal) / range) * trackWidth;
            
            minHandle.style.left = `${minPos}px`;
            maxHandle.style.left = `${maxPos}px`;
            selection.style.left = `${minPos}px`;
            selection.style.width = `${maxPos - minPos}px`;
        };

        // Initialize positions
        setTimeout(updateUI, 0);

        // Drag handling
        const startDrag = (handle, isMin) => {
            const onMove = (e) => {
                e.preventDefault();
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

        minHandle.addEventListener('mousedown', (e) => { e.preventDefault(); startDrag(minHandle, true); });
        maxHandle.addEventListener('mousedown', (e) => { e.preventDefault(); startDrag(maxHandle, false); });
        minHandle.addEventListener('touchstart', (e) => { e.preventDefault(); startDrag(minHandle, true); });
        maxHandle.addEventListener('touchstart', (e) => { e.preventDefault(); startDrag(maxHandle, false); });

        // Input change handlers
        minInput.addEventListener('change', () => {
            currentMin = Math.max(minVal, Math.min(parseFloat(minInput.value), currentMax - 0.001));
            minInput.value = currentMin.toFixed(3);
            updateUI();
        });

        maxInput.addEventListener('change', () => {
            currentMax = Math.min(maxVal, Math.max(parseFloat(maxInput.value), currentMin + 0.001));
            maxInput.value = currentMax.toFixed(3);
            updateUI();
        });

        // Prevent click propagation to analysis item
        thresholdUI.addEventListener('click', (e) => e.stopPropagation());

        // Apply button
        applyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.applyThreshold(analysisItem, modelId, result, currentMin, currentMax);
        });

        // Cancel button - closes UI and restores colormap
        cancelBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.restoreColormap(analysisItem, modelId, result);
            const thresholdBtn = analysisItem.querySelector('.threshold-btn');
            this.hideThresholdUI(thresholdBtn);
        });
    }

    /**
     * Apply threshold and show binary mask
     */
    async applyThreshold(analysisItem, modelId, result, minThreshold, maxThreshold) {
        // Get the selected image
        let imageId = this.platform.selectedImageId;
        if (!imageId && this.platform.imageSearch?.getSelectedImage) {
            const img = this.platform.imageSearch.getSelectedImage();
            imageId = img?.id;
        }
        
        if (!imageId) {
            this.platform.showNotification('No image selected', 'warning');
            return;
        }

        this.platform.showLoading('Applying threshold...');

        try {
            const response = await fetch('/api/apply-threshold-range', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: imageId,
                    model_id: modelId,
                    min_threshold: minThreshold,
                    max_threshold: maxThreshold,
                    colormap: result.colormap || {}
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Failed: ${response.status}`);
            }
            
            const data = await response.json();

            if (data.overlay_url && window.mapManager) {
                // Hide original colormap layer
                window.mapManager.hideAnalysisLayer(modelId);
                
                // Show binary mask
                await window.mapManager.showAnalysisLayer(
                    modelId,
                    data.overlay_url,
                    `${result.name} (Binary)`
                );

                // Update analysis item state to use binary map
                if (analysisItem) {
                    analysisItem.dataset.currentOverlayUrl = data.overlay_url;
                    analysisItem.dataset.isBinary = 'true';
                    analysisItem.dataset.active = 'true';
                    analysisItem.classList.add('active');
                    
                    const statusEl = analysisItem.querySelector('.analysis-status');
                    if (statusEl) {
                        statusEl.textContent = 'Binary mask active';
                        statusEl.classList.remove('inactive');
                        statusEl.classList.add('active');
                    }
                }
            }

            this.platform.hideLoading();
            this.platform.showNotification(`Threshold applied: ${minThreshold.toFixed(3)} ~ ${maxThreshold.toFixed(3)}`, 'success');

        } catch (error) {
            console.error('Threshold error:', error);
            this.platform.hideLoading();
            this.platform.showNotification(`Failed: ${error.message}`, 'error');
        }
    }

    /**
     * Restore original colormap
     */
    async restoreColormap(analysisItem, modelId, result) {
        const originalUrl = this.originalOverlayUrl || analysisItem?.dataset.originalOverlayUrl || result.overlay_url;
        
        if (!originalUrl) {
            this.platform.showNotification('Original colormap not found', 'error');
            return;
        }

        if (window.mapManager) {
            // Hide current layer
            window.mapManager.hideAnalysisLayer(modelId);
            
            // Show original colormap
            await window.mapManager.showAnalysisLayer(
                modelId,
                originalUrl,
                this.originalModelName || result.name
            );

            // Update analysis item state
            if (analysisItem) {
                analysisItem.dataset.currentOverlayUrl = originalUrl;
                analysisItem.dataset.isBinary = 'false';
                analysisItem.dataset.active = 'true';
                analysisItem.classList.add('active');
                
                const statusEl = analysisItem.querySelector('.analysis-status');
                if (statusEl) {
                    statusEl.textContent = 'Overlay active';
                    statusEl.classList.remove('inactive');
                    statusEl.classList.add('active');
                }
            }
        }
        
        this.platform.showNotification('Restored original colormap', 'info');
    }

    /**
     * Disable all active controls
     */
    disableAll() {
        this.disablePixelInspection();
        
        // Remove all threshold control panels
        document.querySelectorAll('.threshold-inline, .threshold-controls').forEach(panel => {
            panel.remove();
        });
        
        // Remove active states from buttons
        document.querySelectorAll('.threshold-btn.active, .pixel-value-btn.active, .btn-inspect.active').forEach(btn => {
            btn.classList.remove('active');
        });
        
        this.activeThresholdModelId = null;
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.ThresholdController = ThresholdController;
}
