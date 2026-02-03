/**
 * Image Search Module
 * Handles satellite image search and selection functionality
 */

class ImageSearchController {
    constructor(platformController) {
        this.platform = platformController;
        this.currentSatellite = 's2';
        this.s2Images = [];
        this.s1Images = [];
        this.selectedImageId = null;
    }

    /**
     * Reset search state
     */
    resetSearchState() {
        this.selectedImageId = null;
        this.s2Images = [];
        this.s1Images = [];
    }

    /**
     * Switch between satellite result tabs
     */
    switchSatelliteResults(satellite) {
        this.currentSatellite = satellite;
        const s2Btn = document.querySelector('.satellite-toggle-btn[data-satellite="s2"]');
        const s1Btn = document.querySelector('.satellite-toggle-btn[data-satellite="s1"]');
        const s2Results = document.getElementById('inline-results-s2');
        const s1Results = document.getElementById('inline-results-s1');

        // Hide all results
        if (s2Results) s2Results.classList.add('hidden');
        if (s1Results) s1Results.classList.add('hidden');

        // Show selected results
        if (satellite === 's2') {
            s2Btn?.classList.add('active');
            s1Btn?.classList.remove('active');
            if (s2Results) s2Results.classList.remove('hidden');
        } else if (satellite === 's1') {
            s1Btn?.classList.add('active');
            s2Btn?.classList.remove('active');
            if (s1Results) s1Results.classList.remove('hidden');
        }
    }

    /**
     * Main search function
     */
    async searchImages() {
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        const cloudCoverInput = document.getElementById('cloud-cover');
        const cloudCover = cloudCoverInput ? parseInt(cloudCoverInput.value) || 20 : 20;

        if (!startDate || !endDate) {
            this.platform.showNotification('Please select start and end dates', 'warning');
            return;
        }

        const aoi = this.platform.getAOI();
        if (!aoi || !aoi.bbox) {
            this.platform.showNotification('Please define an AOI first', 'warning');
            return;
        }

        // Validate bbox
        if (!Array.isArray(aoi.bbox) || aoi.bbox.length !== 4) {
            this.platform.showNotification('Invalid AOI bounds', 'error');
            return;
        }

        this.platform.showLoading('Searching for satellite images...');

        try {
            // Search Sentinel-2
            const s2Request = {
                bbox: aoi.bbox,
                start_date: startDate,
                end_date: endDate,
                cloud_cover_max: cloudCover,
                limit: 50
            };
            
            // Add geometry if available
            if (aoi.geometry) {
                s2Request.geometry = aoi.geometry;
            }
            
            const s2Response = await fetch('/api/search-images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(s2Request)
            });

            if (!s2Response.ok) {
                const errorData = await s2Response.json().catch(() => ({ detail: `S2 search failed: ${s2Response.status}` }));
                throw new Error(errorData.detail || `S2 search failed: ${s2Response.status}`);
            }
            const s2Data = await s2Response.json();
            this.s2Images = s2Data.images || [];

            // Search Sentinel-1
            const s1Request = {
                bbox: aoi.bbox,
                start_date: startDate,
                end_date: endDate,
                limit: 50
            };
            
            if (aoi.geometry) {
                s1Request.geometry = aoi.geometry;
            }
            
            const s1Response = await fetch('/api/search-s1-images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(s1Request)
            });

            if (!s1Response.ok) {
                const errorData = await s1Response.json().catch(() => ({ detail: `S1 search failed: ${s1Response.status}` }));
                throw new Error(errorData.detail || `S1 search failed: ${s1Response.status}`);
            }
            const s1Data = await s1Response.json();
            this.s1Images = s1Data.images || [];

            this.showSearchResults();
            this.platform.hideLoading();
            
            this.platform.showNotification(
                `Found ${this.s2Images.length} Sentinel-2 and ${this.s1Images.length} Sentinel-1 images`,
                'success'
            );

        } catch (error) {
            console.error('Search error:', error);
            this.platform.hideLoading();
            this.platform.showNotification(`Search failed: ${error.message}`, 'error');
        }
    }

    /**
     * Display search results in the UI
     */
    showSearchResults() {
        const s2Container = document.getElementById('search-results-s2');
        const s1Container = document.getElementById('search-results-s1');

        // Always show all result panels, even if empty
        const s2Results = document.getElementById('inline-results-s2');
        const s1Results = document.getElementById('inline-results-s1');
        
        if (s2Container) {
            s2Container.innerHTML = '';
            if (this.s2Images.length > 0) {
                this.s2Images.forEach(image => {
                    s2Container.appendChild(this.createImageItem(image, 's2'));
                });
            } else {
                s2Container.innerHTML = '<p class="no-results">No Sentinel-2 images found</p>';
            }
            const s2Count = document.getElementById('results-count-s2');
            if (s2Count) s2Count.textContent = `${this.s2Images.length} images found`;
        }
        if (s2Results) s2Results.classList.remove('hidden');

        if (s1Container) {
            s1Container.innerHTML = '';
            if (this.s1Images.length > 0) {
                this.s1Images.forEach(image => {
                    s1Container.appendChild(this.createImageItem(image, 's1'));
                });
            } else {
                s1Container.innerHTML = '<p class="no-results">No Sentinel-1 images found</p>';
            }
            const s1Count = document.getElementById('results-count-s1');
            if (s1Count) s1Count.textContent = `${this.s1Images.length} images found`;
        }
        if (s1Results) s1Results.classList.remove('hidden');
    }

    /**
     * Create a single image item element
     */
    createImageItem(image, satellite) {
        const item = document.createElement('div');
        item.className = 'image-item';
        item.dataset.imageId = image.id;
        item.dataset.satellite = satellite;

        const dateStr = image.datetime ? 
            new Date(image.datetime).toLocaleDateString() : 'Unknown date';
        
        const cloudInfo = image.cloud_cover !== null && image.cloud_cover !== undefined ?
            `☁️ ${image.cloud_cover.toFixed(1)}%` : '';
        
        const overlapInfo = image.aoi_overlap !== null && image.aoi_overlap !== undefined ?
            `📍 ${(image.aoi_overlap * 100).toFixed(0)}%` : '';

        const isS1 = satellite === 's1';

        item.innerHTML = `
            <div class="image-header">
                <div class="image-id">${image.id}</div>
                <div class="cloud-cover">${cloudInfo || (image.orbit ? `🛰️ ${image.orbit}` : '')}</div>
            </div>
            <div class="image-details">
                <div class="label">Date:</div>
                <div class="value">${dateStr}</div>
                <div class="label">AOI Coverage:</div>
                <div class="value">${overlapInfo || 'N/A'}</div>
                <div class="label">Collection:</div>
                <div class="value">${image.collection || (isS1 ? 'Sentinel-1' : 'Sentinel-2')}</div>
            </div>
            <div class="image-actions">
                ${isS1 ? `
                    <div class="download-dropdown">
                        <button class="download-btn control-btn primary" data-image-id="${image.id}">
                            ⬇️ Download ▾
                        </button>
                        <div class="download-options hidden">
                            <button class="download-option" data-type="visualization" data-image-id="${image.id}">
                                🎨 With Visualization
                            </button>
                            <button class="download-option" data-type="original" data-image-id="${image.id}">
                                📦 Original (Raw)
                            </button>
                        </div>
                    </div>
                ` : `
                    <button class="process-btn control-btn primary" data-image-id="${image.id}">
                        🔄 Process
                    </button>
                    <div class="download-dropdown">
                        <button class="download-btn control-btn" data-image-id="${image.id}">
                            ⬇️ Download ▾
                        </button>
                        <div class="download-options hidden">
                            <button class="download-option" data-type="visualization" data-image-id="${image.id}">
                                🎨 With Visualization
                            </button>
                            <button class="download-option" data-type="original" data-image-id="${image.id}">
                                📦 Original (Raw)
                            </button>
                        </div>
                    </div>
                `}
            </div>
        `;

        // View on click
        item.addEventListener('click', () => {
            this.viewImage(image, satellite);
        });

        // Process button handler
        const processBtn = item.querySelector('.process-btn');
        if (processBtn) {
            processBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectImage(image.id, satellite);
            });
        }

        // Download button and dropdown handlers
        const downloadBtn = item.querySelector('.download-btn');
        const downloadDropdown = item.querySelector('.download-dropdown');
        const downloadOptions = item.querySelectorAll('.download-option');

        if (downloadBtn && downloadDropdown) {
            downloadBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const options = downloadDropdown.querySelector('.download-options');
                // Close all other dropdowns first
                document.querySelectorAll('.download-options').forEach(opt => {
                    if (opt !== options) opt.classList.add('hidden');
                });
                options.classList.toggle('hidden');
            });
        }

        // Handle download option clicks
        downloadOptions.forEach(optBtn => {
            optBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const type = optBtn.dataset.type;
                this.platform.downloadSatelliteImage(image, type);
                // Hide dropdown after click
                const options = downloadDropdown.querySelector('.download-options');
                if (options) options.classList.add('hidden');
            });
        });

        return item;
    }

    /**
     * View image on the map (toggle selection)
     */
    async viewImage(image, satellite) {
        const aoi = this.platform.getAOI();
        if (!aoi) {
            this.platform.showNotification('Please define an AOI first', 'warning');
            return;
        }

        const imageId = image.id;
        const layerId = `preview-${imageId}`;

        // Toggle logic: if clicking the same image, hide it
        if (this.selectedImageId === imageId) {
            // Deselect
            this.selectedImageId = null;
            this.selectedSatellite = null;
            this.platform.selectedImageId = null;
            this.platform.selectedSatelliteType = null;
            
            // Remove selection highlight
            document.querySelectorAll('.image-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            // Remove all layers related to this image from map
            if (window.mapManager) {
                // Remove preview layer
                window.mapManager.hideTileLayer(layerId);
                // Remove visualization layers (various patterns that might be used)
                window.mapManager.hideTileLayer(`s2-vis-${imageId}`);
                window.mapManager.hideTileLayer(`s1-vis-${imageId}`);
                // Also clear any image layers
                window.mapManager.clearImageLayers();
            }
            
            this.platform.showNotification('Image hidden from map', 'info');
            return;
        }

        // Select new image
        this.selectedImageId = imageId;
        this.selectedSatellite = satellite;
        this.platform.selectedImageId = imageId;
        this.platform.selectedSatelliteType = satellite;

        // Highlight selected item
        document.querySelectorAll('.image-item').forEach(item => {
            item.classList.remove('selected');
        });
        const selectedItem = document.querySelector(`.image-item[data-image-id="${imageId}"]`);
        if (selectedItem) selectedItem.classList.add('selected');

        try {
            const endpoint = satellite === 's1' ? '/api/get-s1-tile' : '/api/get-gee-tile';
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: image.id,
                    bbox: aoi.bbox,
                    geometry: aoi.geometry
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to get tile URL' }));
                throw new Error(errorData.detail || 'Failed to get tile URL');
            }
            const data = await response.json();

            if (window.mapManager && data.tile_template) {
                const satelliteName = satellite === 's1' ? 'Sentinel-1' : 'Sentinel-2';
                window.mapManager.showTileLayer(
                    layerId,
                    data.tile_template,
                    data.bounds,
                    `${satelliteName}: ${image.datetime || 'Unknown date'}`
                );
                this.platform.showNotification(`${satelliteName} image loaded on map`, 'success');
            }
        } catch (error) {
            console.error('View image error:', error);
            this.platform.showNotification(`Failed to load image preview: ${error.message}`, 'error');
        }
    }

    /**
     * Select image for processing
     */
    async selectImage(imageId, satellite = 's2') {
        this.selectedImageId = imageId;
        
        // Highlight selected item
        document.querySelectorAll('.image-item').forEach(item => {
            item.classList.remove('selected');
        });
        const selectedItem = document.querySelector(`.image-item[data-image-id="${imageId}"]`);
        if (selectedItem) selectedItem.classList.add('selected');

        // Process the image
        if (satellite === 's2') {
            await this.platform.processImage(imageId);
        } else {
            this.platform.showNotification('Sentinel-1 processing not yet implemented', 'info');
        }
    }

    /**
     * Get currently selected image
     */
    getSelectedImage() {
        if (!this.selectedImageId) return null;
        
        let image = this.s2Images.find(img => img.id === this.selectedImageId);
        if (!image) {
            image = this.s1Images.find(img => img.id === this.selectedImageId);
        }
        return image;
    }

    /**
     * Find image by ID
     */
    findImageById(imageId) {
        let image = this.s2Images.find(img => img.id === imageId);
        if (!image) {
            image = this.s1Images.find(img => img.id === imageId);
        }
        return image;
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.ImageSearchController = ImageSearchController;
}

