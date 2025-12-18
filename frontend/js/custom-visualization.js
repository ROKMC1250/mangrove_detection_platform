/**
 * Custom Visualization Module
 * Handles custom band composites and index calculations
 */

class CustomVisualizationController {
    constructor(platformController) {
        this.platform = platformController;
        this.modalElement = null;
        this.vizType = 'composite';
    }

    /**
     * Get current image and AOI info
     */
    getImageInfo() {
        const imageId = this.platform.selectedImageId;
        if (!imageId) return null;
        
        const bbox = window.mapManager?.getCurrentBounds();
        const geometry = window.mapManager?.getCurrentGeoJSON()?.geometry;
        
        return {
            id: imageId,
            bbox: bbox,
            geometry: geometry
        };
    }

    /**
     * Show the custom visualization modal
     */
    showModal() {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification('Please process an image first', 'warning');
            return;
        }

        if (!this.modalElement) {
            this.createModal();
        }
        this.modalElement.style.display = 'flex';
    }

    /**
     * Hide the modal
     */
    hideModal() {
        if (this.modalElement) {
            this.modalElement.style.display = 'none';
        }
    }

    /**
     * Create the modal element
     */
    createModal() {
        this.modalElement = document.createElement('div');
        this.modalElement.className = 'modal-overlay custom-viz-modal';
        this.modalElement.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Custom Visualization</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="viz-type-tabs">
                        <button class="viz-type-btn active" data-type="composite">RGB Composite</button>
                        <button class="viz-type-btn" data-type="index">Custom Index</button>
                        <button class="viz-type-btn" data-type="script">Script</button>
                    </div>
                    
                    <div class="viz-type-content" id="composite-content">
                        <div class="form-group">
                            <label>Name</label>
                            <input type="text" id="composite-name" value="Custom RGB" placeholder="Visualization name">
                        </div>
                        <div class="form-group">
                            <label>Red Band</label>
                            <select id="composite-red">
                                <option value="B4">B4 (Red)</option>
                                <option value="B8">B8 (NIR)</option>
                                <option value="B3">B3 (Green)</option>
                                <option value="B2">B2 (Blue)</option>
                                <option value="B11">B11 (SWIR1)</option>
                                <option value="B12">B12 (SWIR2)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Green Band</label>
                            <select id="composite-green">
                                <option value="B3" selected>B3 (Green)</option>
                                <option value="B4">B4 (Red)</option>
                                <option value="B8">B8 (NIR)</option>
                                <option value="B2">B2 (Blue)</option>
                                <option value="B11">B11 (SWIR1)</option>
                                <option value="B12">B12 (SWIR2)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Blue Band</label>
                            <select id="composite-blue">
                                <option value="B2" selected>B2 (Blue)</option>
                                <option value="B4">B4 (Red)</option>
                                <option value="B3">B3 (Green)</option>
                                <option value="B8">B8 (NIR)</option>
                                <option value="B11">B11 (SWIR1)</option>
                                <option value="B12">B12 (SWIR2)</option>
                            </select>
                        </div>
                        <div class="preset-buttons">
                            <button class="btn-preset" data-preset="natural">Natural Color</button>
                            <button class="btn-preset" data-preset="false-color">False Color IR</button>
                            <button class="btn-preset" data-preset="swir">SWIR Composite</button>
                        </div>
                    </div>
                    
                    <div class="viz-type-content" id="index-content" style="display: none;">
                        <div class="form-group">
                            <label>Name</label>
                            <input type="text" id="index-name" value="Custom Index" placeholder="Index name">
                        </div>
                        <div class="form-group">
                            <label>Formula: (A - B) / (A + B)</label>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label>Band A</label>
                                <select id="index-band-a">
                                    <option value="B8" selected>B8 (NIR)</option>
                                    <option value="B4">B4 (Red)</option>
                                    <option value="B3">B3 (Green)</option>
                                    <option value="B2">B2 (Blue)</option>
                                    <option value="B11">B11 (SWIR1)</option>
                                    <option value="B12">B12 (SWIR2)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Band B</label>
                                <select id="index-band-b">
                                    <option value="B4" selected>B4 (Red)</option>
                                    <option value="B8">B8 (NIR)</option>
                                    <option value="B3">B3 (Green)</option>
                                    <option value="B2">B2 (Blue)</option>
                                    <option value="B11">B11 (SWIR1)</option>
                                    <option value="B12">B12 (SWIR2)</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Colormap</label>
                            <select id="index-colormap">
                                <option value="RdYlGn">Red-Yellow-Green</option>
                                <option value="viridis">Viridis</option>
                                <option value="plasma">Plasma</option>
                                <option value="inferno">Inferno</option>
                                <option value="coolwarm">Cool-Warm</option>
                            </select>
                        </div>
                        <div class="preset-buttons">
                            <button class="btn-preset" data-preset="ndvi">NDVI</button>
                            <button class="btn-preset" data-preset="ndmi">NDMI</button>
                            <button class="btn-preset" data-preset="ndwi">NDWI</button>
                        </div>
                    </div>
                    
                    <div class="viz-type-content" id="script-content" style="display: none;">
                        <div class="form-group">
                            <label>Name</label>
                            <input type="text" id="script-name" value="Custom Script" placeholder="Script name">
                        </div>
                        <div class="form-group">
                            <label>JavaScript Expression</label>
                            <textarea id="script-code" rows="6" placeholder="// Example: return [B8, B4, B3];">return [B8, B4, B3];</textarea>
                        </div>
                        <div class="script-help">
                            <p>Available bands: B2, B3, B4, B8, B11, B12</p>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" id="cancel-custom-viz">Cancel</button>
                    <button class="btn btn-primary" id="apply-custom-viz">Apply</button>
                </div>
            </div>
        `;

        document.body.appendChild(this.modalElement);
        this.setupEventListeners();
    }

    /**
     * Setup modal event listeners
     */
    setupEventListeners() {
        // Close button
        const closeBtn = this.modalElement.querySelector('.modal-close');
        closeBtn.addEventListener('click', () => this.hideModal());

        // Cancel button
        const cancelBtn = this.modalElement.querySelector('#cancel-custom-viz');
        cancelBtn.addEventListener('click', () => this.hideModal());

        // Click outside to close
        this.modalElement.addEventListener('click', (e) => {
            if (e.target === this.modalElement) this.hideModal();
        });

        // Apply button
        const applyBtn = this.modalElement.querySelector('#apply-custom-viz');
        applyBtn.addEventListener('click', () => this.applyVisualization());

        // Tab switching
        const tabBtns = this.modalElement.querySelectorAll('.viz-type-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                this.vizType = btn.dataset.type;
                this.showVizTypeContent(this.vizType);
            });
        });

        // Presets
        this.modalElement.querySelectorAll('.btn-preset').forEach(btn => {
            btn.addEventListener('click', () => this.applyPreset(btn.dataset.preset));
        });
    }

    /**
     * Show content for selected visualization type
     */
    showVizTypeContent(type) {
        const contents = this.modalElement.querySelectorAll('.viz-type-content');
        contents.forEach(c => c.style.display = 'none');
        
        const activeContent = this.modalElement.querySelector(`#${type}-content`);
        if (activeContent) activeContent.style.display = 'block';
    }

    /**
     * Apply a preset configuration
     */
    applyPreset(preset) {
        switch (preset) {
            case 'natural':
                document.getElementById('composite-red').value = 'B4';
                document.getElementById('composite-green').value = 'B3';
                document.getElementById('composite-blue').value = 'B2';
                document.getElementById('composite-name').value = 'Natural Color';
                break;
            case 'false-color':
                document.getElementById('composite-red').value = 'B8';
                document.getElementById('composite-green').value = 'B4';
                document.getElementById('composite-blue').value = 'B3';
                document.getElementById('composite-name').value = 'False Color IR';
                break;
            case 'swir':
                document.getElementById('composite-red').value = 'B12';
                document.getElementById('composite-green').value = 'B8';
                document.getElementById('composite-blue').value = 'B4';
                document.getElementById('composite-name').value = 'SWIR Composite';
                break;
            case 'ndvi':
                document.getElementById('index-band-a').value = 'B8';
                document.getElementById('index-band-b').value = 'B4';
                document.getElementById('index-name').value = 'NDVI';
                document.getElementById('index-colormap').value = 'RdYlGn';
                break;
            case 'ndmi':
                document.getElementById('index-band-a').value = 'B8';
                document.getElementById('index-band-b').value = 'B11';
                document.getElementById('index-name').value = 'NDMI';
                document.getElementById('index-colormap').value = 'RdYlGn';
                break;
            case 'ndwi':
                document.getElementById('index-band-a').value = 'B3';
                document.getElementById('index-band-b').value = 'B8';
                document.getElementById('index-name').value = 'NDWI';
                document.getElementById('index-colormap').value = 'coolwarm';
                break;
        }
    }

    /**
     * Apply the custom visualization
     */
    async applyVisualization() {
        const imageInfo = this.getImageInfo();
        if (!imageInfo) {
            this.platform.showNotification('No processed image available', 'warning');
            return;
        }

        let customViz = {};
        let customName = '';

        switch (this.vizType) {
            case 'composite':
                customName = document.getElementById('composite-name').value || 'Custom RGB';
                customViz = {
                    type: 'composite',
                    bands: [
                        document.getElementById('composite-red').value,
                        document.getElementById('composite-green').value,
                        document.getElementById('composite-blue').value
                    ]
                };
                break;
            case 'index':
                customName = document.getElementById('index-name').value || 'Custom Index';
                customViz = {
                    type: 'index',
                    bandA: document.getElementById('index-band-a').value,
                    bandB: document.getElementById('index-band-b').value,
                    colormap: document.getElementById('index-colormap').value
                };
                break;
            case 'script':
                customName = document.getElementById('script-name').value || 'Custom Script';
                customViz = {
                    type: 'script',
                    script: document.getElementById('script-code').value
                };
                break;
        }

        this.hideModal();
        this.platform.showLoading('Creating custom visualization...');

        try {
            const response = await fetch('/api/custom-visualization', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: imageInfo.id,
                    bbox: imageInfo.bbox,
                    geometry: imageInfo.geometry,
                    custom_visualization: customViz,
                    custom_name: customName
                })
            });

            if (!response.ok) throw new Error(`Failed: ${response.status}`);
            const result = await response.json();

            // Add result to analysis list using platform's method
            if (this.platform.addCustomVisualizationResult) {
                this.platform.addCustomVisualizationResult(result);
            }

            this.platform.hideLoading();
            this.platform.showNotification('Custom visualization created', 'success');

        } catch (error) {
            console.error('Custom viz error:', error);
            this.platform.hideLoading();
            this.platform.showNotification(`Failed: ${error.message}`, 'error');
        }
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.CustomVisualizationController = CustomVisualizationController;
}
