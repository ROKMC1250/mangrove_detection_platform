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

            const response = await fetch('/api/process-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    item_id: imageId,
                    bbox: window.mapManager.getCurrentBounds(),
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    intensity_multiplier: 1.5,
                    job_id: jobId
                }),
                signal: signal
            });

            if (response.ok) {
                const data = await response.json();

                this.platform.currentProcessedImageId = imageId;

                // Display analysis results in Analysis tab
                if (data.analysis_results) {
                    this.showAnalysisResults(data.analysis_results);
                }

                this.platform.showNotification('Image processing complete! Check Analysis Results tab.', 'success');
                this.platform._progressStop = true;
                this.platform.hideLoading();
                
                // Switch to Analysis tab to show results
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

    async processEmitImage(imageId, bandSelection) {
        if (!imageId || !bandSelection) {
            this.platform.showNotification('Please select an image and bands first', 'error');
            return;
        }

        this.platform.showLoading('Processing EMIT image with selected bands...');

        try {
            const jobId = `emit-${imageId}-${Date.now()}`;
            this.platform._progressStop = false;

            this.platform.pollProgress(jobId).catch((e) => {
                console.error('[EMIT PROCESS] Polling failed:', e);
            });

            const response = await fetch('/api/process-emit-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    item_id: imageId,
                    bbox: window.mapManager.getCurrentBounds(),
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    selected_bands: bandSelection.selected_bands,
                    visualization_type: bandSelection.visualization_type,
                    rgb_bands: bandSelection.rgb_bands,
                    index_bands: bandSelection.index_bands,
                    colormap: bandSelection.colormap,
                    job_id: jobId
                })
            });

            if (response.ok) {
                const data = await response.json();
                
                if (data.visualization) {
                    const viz = data.visualization;
                    
                    if (viz.preview) {
                        const customResult = {
                            model_id: `emit-${bandSelection.visualization_type}`,
                            name: `EMIT ${bandSelection.visualization_type === 'rgb' ? 'RGB Composite' : 'Custom Index'}`,
                            preview: viz.preview,
                            overlay_url: viz.overlay_url,
                            overlay_meta: viz.overlay_meta,
                            range: viz.range
                        };
                        
                        if (this.platform.analysisController) {
                            this.platform.analysisController.addCustomVisualizationResult(customResult);
                        }
                    }
                }

                this.platform.showNotification('EMIT image processing complete!', 'success');
                this.platform._progressStop = true;
                this.platform.hideLoading();
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'EMIT processing failed');
            }

        } catch (error) {
            console.error('Error processing EMIT image:', error);
            this.platform.showNotification(`Error processing EMIT image: ${error.message}`, 'error');
            this.platform._progressStop = true;
            this.platform.hideLoading();
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

        const sortedEntries = Object.entries(analysisResults).sort(([keyA], [keyB]) => {
            if (keyA === 'overlay_meta') return 1;
            if (keyB === 'overlay_meta') return -1;
            if (keyA === 'cloud_mask') return -1;
            if (keyB === 'cloud_mask') return 1;
            return 0;
        });

        sortedEntries.forEach(([modelId, result]) => {
            if (modelId === 'overlay_meta') return;

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
            `;

            const isIndexResult = result.colormap || modelId.includes('model2') || modelId.includes('model3') || modelId.includes('model4');
            const isDeepLearningModel = modelId === 'model1';
            const isCloudMask = modelId === 'cloud_mask';
            const isAlphaEarth = modelId === 'model5';

            if (isIndexResult && result.colormap) {
                innerHTML += this.createColorbarWithControls(result.colormap, modelId);
            } else if (result.colormap) {
                innerHTML += this.createColorbar(result.colormap);
            } else if (isDeepLearningModel || isAlphaEarth) {
                innerHTML += `
                    <div class="model-controls">
                        <button class="change-monitoring-btn" title="Click to start change monitoring">🌏</button>
                    </div>
                `;
            }

            innerHTML += `</div>`;
            analysisItem.innerHTML = innerHTML;

            analysisItem.dataset.originalModelId = modelId;
            analysisItem.dataset.currentLayerId = modelId;
            analysisItem.dataset.currentOverlayUrl = result.overlay_url;
            analysisItem.dataset.originalOverlayUrl = result.overlay_url;

            // Click to toggle overlay
            analysisItem.onclick = async () => {
                const isActive = analysisItem.dataset.active === 'true';
                const statusEl = analysisItem.querySelector('.analysis-status');
                const currentLayerId = analysisItem.dataset.currentLayerId;
                const currentOverlayUrl = analysisItem.dataset.currentOverlayUrl;
                const isBinary = analysisItem.dataset.isBinary === 'true';

                if (!currentOverlayUrl) {
                    this.platform.showNotification('Overlay not available yet', 'error');
                    return;
                }

                if (!isActive) {
                    if (window.mapManager) {
                        const layerName = isBinary ? `${result.name} (Binary)` : result.name;
                        await window.mapManager.showAnalysisLayer(currentLayerId, currentOverlayUrl, layerName);
                        window.mapManager.outlineAOI();
                    }
                    analysisItem.dataset.active = 'true';
                    analysisItem.classList.add('active');
                    statusEl.textContent = isBinary ? 'Binary mask active' : 'Overlay active';
                    statusEl.classList.remove('inactive');
                    statusEl.classList.add('active');
                } else {
                    if (window.mapManager) {
                        window.mapManager.hideAnalysisLayer(currentLayerId);
                    }
                    analysisItem.dataset.active = 'false';
                    analysisItem.classList.remove('active');
                    statusEl.textContent = 'Click to toggle overlay';
                    statusEl.classList.remove('active');
                    statusEl.classList.add('inactive');
                }
            };

            // Add control button event handlers
            if (isIndexResult && result.colormap) {
                const pixelValueBtn = analysisItem.querySelector('.pixel-value-btn');
                if (pixelValueBtn) {
                    pixelValueBtn.onclick = (e) => {
                        e.stopPropagation();
                        this.platform.togglePixelValueInspection(modelId, result, pixelValueBtn);
                    };
                }

                const changeMonitoringBtn = analysisItem.querySelector('.change-monitoring-btn');
                if (changeMonitoringBtn) {
                    changeMonitoringBtn.onclick = (e) => {
                        e.stopPropagation();
                        this.platform.startChangeMonitoring(modelId, result, changeMonitoringBtn);
                    };
                }

                // Setup colorbar threshold handles
                this.setupColorbarThreshold(analysisItem, modelId, result);
            }

            if (isDeepLearningModel || isAlphaEarth) {
                const changeMonitoringBtn = analysisItem.querySelector('.change-monitoring-btn');
                if (changeMonitoringBtn) {
                    changeMonitoringBtn.onclick = (e) => {
                        e.stopPropagation();
                        this.platform.startChangeMonitoring(modelId, result, changeMonitoringBtn);
                    };
                }
            }

            analysisList.appendChild(analysisItem);
        });

        this.addCustomVisualizationOption(analysisList);
        this.addTargetDetectionOption(analysisList);

        console.log('Analysis results displayed in list format');
    }

    addTargetDetectionOption(analysisList) {
        const targetItem = document.createElement('div');
        targetItem.className = 'analysis-item target-detection-option';
        targetItem.dataset.modelId = 'target-detection';
        targetItem.dataset.active = 'false';

        targetItem.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder" style="background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);">
                <div class="custom-icon">🎯</div>
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
                    `${result.name} (Binary)`
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
                        <button class="change-monitoring-btn" title="Change monitoring">🌏</button>
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
