/**
 * Image Processor Module
 * Handles image processing, analysis display, and colorbar rendering
 */

class ImageProcessorController {
    constructor(platformController) {
        this.platform = platformController;
        this.lastAnalysisResults = null;
        this._modelMaskUrls = null;
        this._currentAbortController = null;
    }

    /**
     * Cancel any ongoing request
     */
    cancelCurrentRequest() {
        if (this._currentAbortController) {
            this._currentAbortController.abort();
            this._currentAbortController = null;
            console.log('[PROCESS] Request cancelled');
        }
    }

    async processImage(imageId) {
        if (!imageId) {
            this.platform.showNotification('Please select an image first', 'error');
            return;
        }

        // Cancel any previous request
        this.cancelCurrentRequest();

        // Create new AbortController
        this._currentAbortController = new AbortController();
        const signal = this._currentAbortController.signal;

        // Listen for cancel event
        const cancelHandler = () => {
            this.cancelCurrentRequest();
        };
        window.addEventListener('loadingCancelled', cancelHandler, { once: true });

        this.platform.showLoading('Processing image on server (export + analysis)...');

        try {
            const jobId = `${imageId}-${Date.now()}`;
            console.log(`[PROCESS] Starting process-image with job_id: ${jobId}`);
            this.platform._progressStop = false;

            // Start polling
            this.platform.pollProgress(jobId).catch((e) => {
                console.error('[PROCESS] Polling failed:', e);
            });

            const currentBbox = window.mapManager.getCurrentBounds();
            const currentGeometry = window.mapManager.getCurrentGeoJSON()?.geometry;

            // Save bbox/geometry for later use by mangrove segmentation
            this.platform.processedBbox = currentBbox;
            this.platform.processedGeometry = currentGeometry;

            const response = await fetch('/api/process-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    item_id: imageId,
                    bbox: currentBbox,
                    geometry: currentGeometry,
                    intensity_multiplier: 1.5,
                    job_id: jobId
                }),
                signal: signal
            });

            if (response.ok) {
                const data = await response.json();

                this.platform.currentProcessedImageId = imageId;

                // A fresh process on the active slot invalidates any
                // previously-registered analysis result ids for this slot —
                // those ids point at cached rasters for the prior image and
                // would fail the shape check inside change-detection. Clear
                // the registry; leave Leaflet layers alone (they'll be
                // replaced by modelId as the user re-runs each analysis).
                const slotState = this.platform._slots?.[this.platform.currentSlot];
                if (slotState) slotState.analyses = [];
                if (typeof this.platform._updateSlotToolbar === 'function') {
                    this.platform._updateSlotToolbar();
                }

                // Display analysis results in Analysis tab
                if (data.analysis_results) {
                    this.showAnalysisResults(data.analysis_results);
                }

                this.platform.showNotification('Image processing complete! Check Analysis Results tab.', 'success');
                this.platform._progressStop = true;
                this.platform.hideLoading();
                
                // Enable analyze button and switch to analysis
                const openBtn = document.getElementById('open-analysis-btn');
                if (openBtn) openBtn.disabled = false;
                this.platform.switchToAnalysisTab();

            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Processing failed');
            }

        } catch (error) {
            // Handle abort (user cancelled)
            if (error.name === 'AbortError') {
                console.log('[PROCESS] Request was cancelled by user');
                this.platform._progressStop = true;
                // Notification is already shown by UIManager
                return;
            }
            
            console.error('Error processing image:', error);
            this.platform.showNotification(`Error processing image: ${error.message}`, 'error');
            this.platform._progressStop = true;
            this.platform.hideLoading();
        } finally {
            // Cleanup
            this._currentAbortController = null;
            window.removeEventListener('loadingCancelled', cancelHandler);
        }
    }

    showAnalysisResults(analysisResults) {
        this.lastAnalysisResults = analysisResults;
        this.platform.lastAnalysisResults = analysisResults;

        // Switch to analysis tab
        if (this.platform.uiManager) {
            this.platform.uiManager.switchToAnalysisTab();
        } else {
            this.platform.switchToAnalysisTab();
        }

        const analysisList = document.querySelector('.analysis-list');
        if (!analysisList) return;

        analysisList.innerHTML = '';

        // Slot-switch may call us with null/empty results when the target
        // slot hasn't been processed yet. Show an empty state instead of
        // crashing on `analysisResults[modelId]`.
        if (!analysisResults || typeof analysisResults !== 'object'
            || Object.keys(analysisResults).length === 0) {
            const slot = this.platform.currentSlot || 'A';
            analysisList.innerHTML = `
                <div class="no-analysis-results">
                    <div class="empty-state-icon">📊</div>
                    <h4 class="empty-state-title">Time ${slot}: No analysis yet</h4>
                    <p class="empty-state-message">
                        Select an image and click <b>Analyze</b> to populate Time ${slot}.
                    </p>
                </div>`;
            return;
        }

        // Structured layout: Cloud Mask, AlphaEarth, Spectral Analysis, Target Detection
        const renderOrder = ['cloud_mask', 'model5'];

        renderOrder.forEach(modelId => {
            const result = analysisResults[modelId];
            if (!result) return;
            this._addOverlayItem(analysisList, modelId, result);
        });

        // Spectral Analysis section
        this.addSpectralAnalysisOption(analysisList);

        // Target Detection section
        this.addTargetDetectionOption(analysisList);

        // Mangrove Segmentation section
        this.addMangroveSegmentationOption(analysisList);

        // SAM3 Segmentation section
        this.addSAM3Option(analysisList);

        console.log('Analysis results displayed in list format');
    }

    _addOverlayItem(analysisList, modelId, result) {
        const analysisItem = document.createElement('div');
        analysisItem.className = 'analysis-item';
        analysisItem.dataset.modelId = modelId;
        analysisItem.dataset.active = 'false';

        let innerHTML = `
            <img class="analysis-thumbnail" src="${result.preview_url || result.thumbnail_url || ''}" alt="${result.name}" />
            <div class="analysis-info">
                <h4 class="analysis-title">${result.name}</h4>
                <p class="analysis-subtitle">${(result.bands || []).join(', ')}</p>
                <div class="analysis-status inactive">Click to toggle overlay</div>
            </div>
        `;

        analysisItem.innerHTML = innerHTML;

        analysisItem.dataset.originalModelId = modelId;
        analysisItem.dataset.currentLayerId = modelId;
        analysisItem.dataset.currentOverlayUrl = result.overlay_url;
        analysisItem.dataset.originalOverlayUrl = result.overlay_url;

        // Click to toggle overlay
        analysisItem.onclick = async () => {
            const isActive = analysisItem.dataset.active === 'true';
            const statusEl = analysisItem.querySelector('.analysis-status');
            const currentOverlayUrl = analysisItem.dataset.currentOverlayUrl;

            if (!currentOverlayUrl) {
                this.platform.showNotification('Overlay not available yet', 'error');
                return;
            }

            if (!isActive) {
                if (window.mapManager) {
                    await window.mapManager.showAnalysisLayer(modelId, currentOverlayUrl, result.name);
                    window.mapManager.outlineAOI();
                }
                analysisItem.dataset.active = 'true';
                analysisItem.classList.add('active');
                statusEl.textContent = 'Overlay active';
                statusEl.classList.remove('inactive');
                statusEl.classList.add('active');
            } else {
                if (window.mapManager) {
                    window.mapManager.hideAnalysisLayer(modelId);
                }
                analysisItem.dataset.active = 'false';
                analysisItem.classList.remove('active');
                statusEl.textContent = 'Click to toggle overlay';
                statusEl.classList.remove('active');
                statusEl.classList.add('inactive');
            }
        };

        analysisList.appendChild(analysisItem);
    }

    addSpectralAnalysisOption(analysisList) {
        const saItem = document.createElement('div');
        saItem.className = 'analysis-item spectral-analysis-option';
        saItem.dataset.modelId = 'spectral-analysis';
        saItem.dataset.active = 'false';

        saItem.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder" style="background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);">
                <div class="custom-icon">📊</div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">Spectral Analysis</h4>
                <div class="analysis-status inactive">Click to start</div>
            </div>
        `;

        saItem.onclick = (e) => {
            if (e.target.closest('.sa-ui')) return;
            if (!this.platform.spectralAnalysis) {
                this.platform.spectralAnalysis = new SpectralAnalysisController(this.platform);
            }
            this.platform.spectralAnalysis.handleItemClick(saItem);
        };

        analysisList.appendChild(saItem);
    }

    addTargetDetectionOption(analysisList) {
        const targetItem = document.createElement('div');
        targetItem.className = 'analysis-item target-detection-option';
        targetItem.dataset.modelId = 'target-detection';
        targetItem.dataset.active = 'false';

        targetItem.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder" style="background: linear-gradient(135deg, #1565c0 0%, #00838f 100%);">
                <div class="custom-icon td-icon-svg">
                    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="white" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <circle cx="12" cy="12" r="6"/>
                        <circle cx="12" cy="12" r="2"/>
                        <line x1="12" y1="1" x2="12" y2="4"/>
                        <line x1="12" y1="20" x2="12" y2="23"/>
                        <line x1="1" y1="12" x2="4" y2="12"/>
                        <line x1="20" y1="12" x2="23" y2="12"/>
                    </svg>
                </div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">Target Detection</h4>
                <div class="analysis-status inactive">Click to start</div>
            </div>
        `;

        targetItem.onclick = (e) => {
            if (e.target.closest('.td-ui')) {
                return;
            }
            
            if (!this.platform.targetDetection) {
                this.platform.showNotification('Target Detection module not loaded', 'error');
                return;
            }

            this.platform.targetDetection.handleItemClick();
        };

        analysisList.appendChild(targetItem);
    }

    addMangroveSegmentationOption(analysisList) {
        const msItem = document.createElement('div');
        msItem.className = 'analysis-item mangrove-segmentation-option';
        msItem.dataset.modelId = 'mangrove-segmentation';
        msItem.dataset.active = 'false';

        msItem.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder" style="background: linear-gradient(135deg, #00897b 0%, #00bcd4 100%);">
                <div class="custom-icon">🌿</div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">Mangrove Segmentation</h4>
                <div class="analysis-status inactive">Click to start</div>
            </div>
        `;

        msItem.onclick = (e) => {
            if (e.target.closest('.ms-ui')) return;
            try {
                if (!this.platform.mangroveSegmentation) {
                    this.platform.mangroveSegmentation = new MangroveSegmentationController(this.platform);
                }
                this.platform.mangroveSegmentation.handleItemClick(msItem);
            } catch (err) {
                console.error('Mangrove Segmentation error:', err);
                this.platform.showNotification(`Mangrove Segmentation error: ${err.message}`, 'error');
            }
        };

        analysisList.appendChild(msItem);
    }

    addSAM3Option(analysisList) {
        const sam3Item = document.createElement('div');
        sam3Item.className = 'analysis-item sam3-option';
        sam3Item.dataset.modelId = 'sam3';
        sam3Item.dataset.active = 'false';

        sam3Item.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder" style="background: linear-gradient(135deg, #7b1fa2 0%, #e91e63 100%);">
                <div class="custom-icon sam3-icon-svg">
                    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="white" stroke-width="2">
                        <path d="M6 9l6 6 6-6"/>
                        <rect x="3" y="3" width="18" height="18" rx="3"/>
                        <circle cx="8" cy="8" r="1.5" fill="white"/>
                        <circle cx="16" cy="16" r="1.5" fill="white"/>
                    </svg>
                </div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">SAM3 Segmentation</h4>
                <div class="analysis-status inactive">Click to start</div>
            </div>
        `;

        sam3Item.onclick = (e) => {
            if (e.target.closest('.sam3-ui')) return;
            if (!this.platform.sam3Controller) {
                this.platform.showNotification('SAM3 module not loaded', 'error');
                return;
            }
            this.platform.sam3Controller.handleItemClick();
        };

        analysisList.appendChild(sam3Item);
    }

    addCustomVisualizationOption(analysisList) {
        const customItem = document.createElement('div');
        customItem.className = 'analysis-item custom-visualization';
        customItem.dataset.modelId = 'custom';
        customItem.dataset.active = 'false';

        customItem.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder">
                <div class="custom-icon">🛠️</div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">✏️ Custom Visualization</h4>
                <p class="analysis-subtitle">Create custom visualizations</p>
                <div class="analysis-status inactive">Click to create custom visualization</div>
            </div>
        `;

        customItem.onclick = () => {
            this.platform.showCustomVisualizationModal();
        };

        analysisList.appendChild(customItem);
    }

    addCustomVisualizationResult(result) {
        const analysisList = document.querySelector('.analysis-list');
        if (!analysisList) return;

        const customItem = analysisList.querySelector('.custom-visualization');
        
        const resultItem = document.createElement('div');
        resultItem.className = 'analysis-item custom-result';
        resultItem.dataset.modelId = result.model_id;
        resultItem.dataset.active = 'false';

        resultItem.innerHTML = `
            <img class="analysis-thumbnail" src="${result.preview}" alt="${result.name}" />
            <div class="analysis-info">
                <h4 class="analysis-title">${result.name}</h4>
                <div class="analysis-status inactive">Click to toggle overlay</div>
            </div>
        `;

        resultItem.onclick = async () => {
            const isActive = resultItem.dataset.active === 'true';
            const statusEl = resultItem.querySelector('.analysis-status');

            if (!isActive) {
                if (window.mapManager && result.overlay_url) {
                    await window.mapManager.showAnalysisLayer(result.model_id, result.overlay_url, result.name);
                    window.mapManager.outlineAOI();
                }
                resultItem.dataset.active = 'true';
                resultItem.classList.add('active');
                statusEl.textContent = 'Overlay active';
                statusEl.classList.remove('inactive');
                statusEl.classList.add('active');
            } else {
                if (window.mapManager) {
                    window.mapManager.hideAnalysisLayer(result.model_id);
                }
                resultItem.dataset.active = 'false';
                resultItem.classList.remove('active');
                statusEl.textContent = 'Click to toggle overlay';
                statusEl.classList.remove('active');
                statusEl.classList.add('inactive');
            }
        };

        if (customItem) {
            analysisList.insertBefore(resultItem, customItem);
        } else {
            analysisList.appendChild(resultItem);
        }
    }

    createColorbar(colormap) {
        const minFormatted = colormap.min_val.toFixed(3);
        const maxFormatted = colormap.max_val.toFixed(3);

        return `
            <div class="colorbar-container">
                <div class="colorbar-label">${colormap.label}</div>
                <div class="colorbar">
                    <div class="colorbar-gradient ${colormap.name}"></div>
                </div>
                <div class="colorbar-values">
                    <span class="colorbar-min">${minFormatted}</span>
                    <span class="colorbar-max">${maxFormatted}</span>
                </div>
            </div>
        `;
    }

    setupColorbarThreshold(analysisItem, modelId, result) {
        const container = analysisItem.querySelector('.colorbar-container');
        if (!container) return;

        const track = container.querySelector('.colorbar-track');
        const minHandle = container.querySelector('.min-handle');
        const maxHandle = container.querySelector('.max-handle');
        const selection = container.querySelector('.colorbar-selection');
        const minInput = container.querySelector('.colorbar-min-input');
        const maxInput = container.querySelector('.colorbar-max-input');
        const applyBtn = container.querySelector('.colorbar-apply-btn');
        const cancelBtn = container.querySelector('.colorbar-cancel-btn');

        if (!track || !minHandle || !maxHandle) return;

        const colormap = result.colormap || { min_val: -1, max_val: 1 };
        const minVal = colormap.min_val;
        const maxVal = colormap.max_val;

        // Current threshold values
        let currentMin = minVal;
        let currentMax = maxVal;

        // Update UI positions
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

        // Initialize after DOM render
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
        minInput.addEventListener('change', (e) => {
            e.stopPropagation();
            currentMin = Math.max(minVal, Math.min(parseFloat(minInput.value), currentMax - 0.001));
            minInput.value = currentMin.toFixed(3);
            updateUI();
        });
        minInput.addEventListener('click', (e) => e.stopPropagation());

        maxInput.addEventListener('change', (e) => {
            e.stopPropagation();
            currentMax = Math.min(maxVal, Math.max(parseFloat(maxInput.value), currentMin + 0.001));
            maxInput.value = currentMax.toFixed(3);
            updateUI();
        });
        maxInput.addEventListener('click', (e) => e.stopPropagation());

        // Apply button
        applyBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await this.applyColorbarThreshold(analysisItem, modelId, result, currentMin, currentMax);
            // Show cancel button after applying
            if (cancelBtn) cancelBtn.style.display = 'inline-block';
        });

        // Cancel button - restore original colormap
        if (cancelBtn) {
            cancelBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                await this.cancelColorbarThreshold(analysisItem, modelId, result);
                // Reset handles to full range
                currentMin = minVal;
                currentMax = maxVal;
                minInput.value = minVal.toFixed(3);
                maxInput.value = maxVal.toFixed(3);
                updateUI();
                // Hide cancel button
                cancelBtn.style.display = 'none';
            });
        }

        // Prevent colorbar clicks from toggling the overlay
        container.addEventListener('click', (e) => e.stopPropagation());
    }

    async cancelColorbarThreshold(analysisItem, modelId, result) {
        const originalUrl = analysisItem.dataset.originalOverlayUrl;
        
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
                result.name
            );

            // Update analysis item state
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
        
        this.platform.showNotification('Restored original colormap', 'info');
    }

    async applyColorbarThreshold(analysisItem, modelId, result, minThreshold, maxThreshold) {
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
                    `${result.name} (Binary)`,
                    null,
                    true
                );

                // Update analysis item state
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

            this.platform.hideLoading();
            this.platform.showNotification(`Threshold: ${minThreshold.toFixed(3)} ~ ${maxThreshold.toFixed(3)}`, 'success');

        } catch (error) {
            console.error('Threshold error:', error);
            this.platform.hideLoading();
            this.platform.showNotification(`Failed: ${error.message}`, 'error');
        }
    }

    createColorbarWithControls(colormap, modelId) {
        const minVal = colormap.min_val;
        const maxVal = colormap.max_val;
        const minFormatted = minVal.toFixed(3);
        const maxFormatted = maxVal.toFixed(3);

        return `
            <div class="colorbar-container" data-model-id="${modelId}" 
                 data-min-val="${minVal}" data-max-val="${maxVal}"
                 data-colormap="${colormap.name}" data-label="${colormap.label}">
                <div class="colorbar-header">
                    <div class="colorbar-label">${colormap.label}</div>
                    <div class="colorbar-controls">
                        <button class="pixel-value-btn" title="Inspect pixel values">🖱️</button>
                    </div>
                </div>
                <div class="colorbar-with-threshold">
                    <div class="colorbar-track">
                        <div class="colorbar-gradient ${colormap.name}"></div>
                        <div class="colorbar-selection"></div>
                        <div class="colorbar-handle min-handle" data-handle="min"></div>
                        <div class="colorbar-handle max-handle" data-handle="max"></div>
                    </div>
                </div>
                <div class="colorbar-values">
                    <input type="number" class="colorbar-min-input" value="${minFormatted}" step="0.001">
                    <div class="colorbar-buttons">
                        <button class="colorbar-apply-btn">Apply</button>
                        <button class="colorbar-cancel-btn" style="display:none;">Cancel</button>
                    </div>
                    <input type="number" class="colorbar-max-input" value="${maxFormatted}" step="0.001">
                </div>
            </div>
        `;
    }
}
