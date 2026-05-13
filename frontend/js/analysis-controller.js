/**
 * Analysis Controller Module
 * Handles display and interaction with analysis results
 */

class AnalysisController {
    constructor(platformController) {
        this.platform = platformController;
        this.currentResults = null;
        this.activeModelId = null;
    }

    /**
     * Show analysis results in the UI
     */
    showAnalysisResults(analysisResults) {
        this.currentResults = analysisResults;

        const analysisSection = document.getElementById('analysis-section');
        const analysisList = document.querySelector('.analysis-list');

        if (!analysisList) {
            console.error('Analysis list not found');
            return;
        }

        analysisSection.style.display = 'block';
        analysisList.innerHTML = '';

        // Define model order
        const modelOrder = ['cloud_mask', 'model1', 'model2', 'model3', 'model4', 'model5'];
        
        for (const modelId of modelOrder) {
            const result = analysisResults[modelId];
            if (!result || !result.preview_url) continue;

            const item = this.createAnalysisItem(modelId, result);
            analysisList.appendChild(item);
        }

        // Add custom visualization option
        this.addCustomVisualizationOption(analysisList);

        // Switch to analysis tab
        this.switchToAnalysisTab();
    }

    /**
     * Create a single analysis item element
     */
    createAnalysisItem(modelId, result) {
        const item = document.createElement('div');
        item.className = 'analysis-item';
        item.dataset.modelId = modelId;

        const hasColormap = result.colormap && result.colormap.name;

        item.innerHTML = `
            <div class="analysis-preview">
                <img src="${result.preview_url}" alt="${result.name}" loading="lazy">
            </div>
            <div class="analysis-info">
                <div class="analysis-name">${result.name}</div>
                ${hasColormap ? this.createColorbar(result.colormap) : ''}
                <div class="analysis-status inactive">Click to toggle overlay</div>
            </div>
            <div class="analysis-controls">
                ${hasColormap ? `
                    <button class="btn-small btn-inspect" title="Inspect pixel values">🔍</button>
                    <button class="btn-small btn-threshold" title="Apply threshold">📊</button>
                ` : ''}
            </div>
        `;

        // Toggle overlay on click
        item.addEventListener('click', (e) => {
            if (e.target.closest('.analysis-controls')) return;
            this.toggleOverlay(modelId, result, item);
        });

        // Inspect button
        const inspectBtn = item.querySelector('.btn-inspect');
        if (inspectBtn) {
            inspectBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.platform.togglePixelValueInspection(modelId, result, inspectBtn);
            });
        }

        // Threshold button
        const thresholdBtn = item.querySelector('.btn-threshold');
        if (thresholdBtn) {
            thresholdBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.platform.toggleThresholdControl(modelId, result, thresholdBtn);
            });
        }

        return item;
    }

    /**
     * Toggle overlay visibility
     */
    async toggleOverlay(modelId, result, itemElement) {
        const statusEl = itemElement.querySelector('.analysis-status');
        const isActive = itemElement.classList.contains('active');

        if (!isActive) {
            // Show overlay
            if (window.mapManager && result.overlay_url) {
                await window.mapManager.showAnalysisLayer(
                    modelId, 
                    result.overlay_url, 
                    result.name
                );
                window.mapManager.outlineAOI();
            }
            
            itemElement.classList.add('active');
            if (statusEl) {
                statusEl.textContent = 'Overlay active';
                statusEl.classList.remove('inactive');
                statusEl.classList.add('active');
            }
            this.activeModelId = modelId;
        } else {
            // Hide overlay
            if (window.mapManager) {
                window.mapManager.hideAnalysisLayer(modelId);
            }
            
            itemElement.classList.remove('active');
            if (statusEl) {
                statusEl.textContent = 'Click to toggle overlay';
                statusEl.classList.remove('active');
                statusEl.classList.add('inactive');
            }
            
            if (this.activeModelId === modelId) {
                this.activeModelId = null;
            }
        }
    }

    /**
     * Create colorbar HTML
     */
    createColorbar(colormap) {
        if (!colormap) return '';
        
        const { name, min_val, max_val, label } = colormap;
        const minFormatted = Number(min_val).toFixed(3);
        const maxFormatted = Number(max_val).toFixed(3);

        return `
            <div class="colorbar-container">
                <div class="colorbar-label">${label || name}</div>
                <div class="colorbar">
                    <div class="colorbar-gradient ${name}"></div>
                </div>
                <div class="colorbar-values">
                    <span class="colorbar-min">${minFormatted}</span>
                    <span class="colorbar-max">${maxFormatted}</span>
                </div>
            </div>
        `;
    }

    /**
     * Add custom visualization option
     */
    addCustomVisualizationOption(analysisList) {
        const customItem = document.createElement('div');
        customItem.className = 'analysis-item custom-viz-option';
        customItem.innerHTML = `
            <div class="custom-viz-icon">✨</div>
            <div class="analysis-info">
                <div class="analysis-name">Custom Visualization</div>
                <div class="analysis-status">Create your own index or RGB composite</div>
            </div>
        `;

        customItem.addEventListener('click', () => {
            this.platform.showCustomVisualizationModal();
        });

        analysisList.appendChild(customItem);
    }

    /**
     * Add custom visualization result
     */
    addCustomVisualizationResult(result) {
        const analysisList = document.querySelector('.analysis-list');
        if (!analysisList) return;

        const customId = result.custom_id || `custom-${Date.now()}`;
        
        const item = document.createElement('div');
        item.className = 'analysis-item custom-result';
        item.dataset.modelId = customId;
        item.dataset.customId = customId;
        item.dataset.currentOverlayUrl = result.overlay_url;
        item.dataset.originalOverlayUrl = result.overlay_url;
        item.dataset.active = 'false';

        const hasColormap = result.colormap && result.colormap.name;

        item.innerHTML = `
            <div class="analysis-preview">
                <img src="${result.preview_url}" alt="${result.name}" loading="lazy">
            </div>
            <div class="analysis-info">
                <div class="analysis-name">${result.name}</div>
                ${hasColormap ? this.createColorbar(result.colormap) : ''}
                <div class="analysis-status inactive">Click to toggle overlay</div>
            </div>
            <div class="analysis-controls">
                ${hasColormap ? `
                    <button class="btn-small btn-inspect" title="Inspect pixel values">🔍</button>
                    <button class="btn-small btn-threshold" title="Apply threshold">📊</button>
                ` : ''}
                <button class="btn-small btn-remove" title="Remove">🗑️</button>
            </div>
        `;

        // Toggle overlay
        item.addEventListener('click', (e) => {
            if (e.target.closest('.analysis-controls')) return;
            this.toggleCustomOverlay(customId, result, item);
        });

        // Remove button
        const removeBtn = item.querySelector('.btn-remove');
        if (removeBtn) {
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (window.mapManager) {
                    window.mapManager.hideAnalysisLayer(customId);
                }
                item.remove();
            });
        }

        // Insert before the custom viz option
        const customOption = analysisList.querySelector('.custom-viz-option');
        if (customOption) {
            analysisList.insertBefore(item, customOption);
        } else {
            analysisList.appendChild(item);
        }
    }

    /**
     * Toggle custom visualization overlay
     */
    async toggleCustomOverlay(customId, result, itemElement) {
        const statusEl = itemElement.querySelector('.analysis-status');
        const isActive = itemElement.dataset.active === 'true';
        const overlayUrl = itemElement.dataset.currentOverlayUrl;

        if (!isActive) {
            if (window.mapManager && overlayUrl) {
                await window.mapManager.showAnalysisLayer(customId, overlayUrl, result.name);
                window.mapManager.outlineAOI();
            }
            
            itemElement.dataset.active = 'true';
            itemElement.classList.add('active');
            if (statusEl) {
                statusEl.textContent = 'Overlay active';
                statusEl.classList.remove('inactive');
                statusEl.classList.add('active');
            }
        } else {
            if (window.mapManager) {
                window.mapManager.hideAnalysisLayer(customId);
            }
            
            itemElement.dataset.active = 'false';
            itemElement.classList.remove('active');
            if (statusEl) {
                statusEl.textContent = 'Click to toggle overlay';
                statusEl.classList.remove('active');
                statusEl.classList.add('inactive');
            }
        }
    }

    /**
     * Switch to analysis tab
     */
    switchToAnalysisTab() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabBtns.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        const analysisTabBtn = document.querySelector('.tab-btn[data-tab="analysis"]');
        const analysisTabContent = document.getElementById('analysis-tab-content');
        
        if (analysisTabBtn) analysisTabBtn.classList.add('active');
        if (analysisTabContent) analysisTabContent.classList.add('active');
    }

    /**
     * Get current results
     */
    getCurrentResults() {
        return this.currentResults;
    }

    /**
     * Get result by model ID
     */
    getResult(modelId) {
        return this.currentResults ? this.currentResults[modelId] : null;
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.AnalysisController = AnalysisController;
}

