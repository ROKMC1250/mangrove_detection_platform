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
        const s2Results = document.getElementById('s2-results');
        const s1Results = document.getElementById('s1-results');

        if (satellite === 's2') {
            s2Btn?.classList.add('active');
            s1Btn?.classList.remove('active');
            if (s2Results) s2Results.style.display = 'block';
            if (s1Results) s1Results.style.display = 'none';
        } else {
            s1Btn?.classList.add('active');
            s2Btn?.classList.remove('active');
            if (s1Results) s1Results.style.display = 'block';
            if (s2Results) s2Results.style.display = 'none';
        }
    }

    /**
     * Main search function
     */
    async searchImages() {
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        const cloudCover = parseInt(document.getElementById('cloud-cover').value);

        if (!startDate || !endDate) {
            this.platform.showNotification('Please select start and end dates', 'warning');
            return;
        }

        const aoi = this.platform.getAOI();
        if (!aoi) {
            this.platform.showNotification('Please define an AOI first', 'warning');
            return;
        }

        this.platform.showLoading('Searching for satellite images...');

        try {
            // Search Sentinel-2
            const s2Response = await fetch('/api/search-images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    bbox: aoi.bbox,
                    geometry: aoi.geometry,
                    start_date: startDate,
                    end_date: endDate,
                    cloud_cover_max: cloudCover,
                    limit: 50
                })
            });

            if (!s2Response.ok) throw new Error(`S2 search failed: ${s2Response.status}`);
            const s2Data = await s2Response.json();
            this.s2Images = s2Data.images || [];

            // Search Sentinel-1
            const s1Response = await fetch('/api/search-s1-images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    bbox: aoi.bbox,
                    geometry: aoi.geometry,
                    start_date: startDate,
                    end_date: endDate,
                    limit: 50
                })
            });

            if (!s1Response.ok) throw new Error(`S1 search failed: ${s1Response.status}`);
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
        const s2Container = document.getElementById('s2-results');
        const s1Container = document.getElementById('s1-results');

        if (s2Container) {
            s2Container.innerHTML = '';
            this.s2Images.forEach(image => {
                s2Container.appendChild(this.createImageItem(image, 's2'));
            });
        }

        if (s1Container) {
            s1Container.innerHTML = '';
            this.s1Images.forEach(image => {
                s1Container.appendChild(this.createImageItem(image, 's1'));
            });
        }

        // Show results panel
        const resultsPanel = document.getElementById('results-section');
        if (resultsPanel) resultsPanel.style.display = 'block';
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

        item.innerHTML = `
            <div class="image-info">
                <div class="image-date">${dateStr}</div>
                <div class="image-meta">
                    ${cloudInfo} ${overlapInfo}
                    ${image.orbit ? `🛰️ ${image.orbit}` : ''}
                </div>
            </div>
            <div class="image-actions">
                <button class="btn-small btn-view" title="View on map">👁️</button>
                <button class="btn-small btn-process" title="Process">⚡</button>
            </div>
        `;

        // View button handler
        const viewBtn = item.querySelector('.btn-view');
        viewBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.viewImage(image, satellite);
        });

        // Process button handler
        const processBtn = item.querySelector('.btn-process');
        processBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectImage(image.id, satellite);
        });

        // Item click handler
        item.addEventListener('click', () => {
            this.viewImage(image, satellite);
        });

        return item;
    }

    /**
     * View image on the map
     */
    async viewImage(image, satellite) {
        const aoi = this.platform.getAOI();
        if (!aoi) return;

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

            if (!response.ok) throw new Error('Failed to get tile URL');
            const data = await response.json();

            if (window.mapManager && data.tile_template) {
                window.mapManager.showTileLayer(
                    `preview-${image.id}`,
                    data.tile_template,
                    data.bounds,
                    `${satellite.toUpperCase()}: ${image.datetime}`
                );
            }
        } catch (error) {
            console.error('View image error:', error);
            this.platform.showNotification('Failed to load image preview', 'error');
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

