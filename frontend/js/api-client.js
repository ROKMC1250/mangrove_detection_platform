/**
 * API Client for Backend Communication
 */
class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    if (typeof errorData.detail === 'string') {
                        errorMessage = errorData.detail;
                    } else if (errorData.detail && typeof errorData.detail === 'object') {
                        errorMessage = JSON.stringify(errorData.detail);
                    } else if (errorData.message) {
                        errorMessage = errorData.message;
                    }
                } catch (parseError) {
                    // Use default HTTP error message
                }
                throw new Error(errorMessage);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    // Satellite Search
    async searchImages(bbox, startDate, endDate, cloudCoverMax = 30) {
        return this.request('/api/search', {
            method: 'POST',
            body: JSON.stringify({
                bbox,
                start_date: startDate,
                end_date: endDate,
                cloud_cover_max: cloudCoverMax
            })
        });
    }

    // Image Processing
    async processImage(imageId, bbox) {
        return this.request('/api/process-image', {
            method: 'POST',
            body: JSON.stringify({
                item_id: imageId,
                bbox
            })
        });
    }

    // Custom Visualization
    async createCustomVisualization(params) {
        return this.request('/api/custom-visualization', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    }

    // Change Monitoring
    async startChangeMonitoring(params) {
        return this.request('/api/change-monitoring', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    }

    // Pixel Value Inspection
    async getPixelValue(lat, lng, modelId) {
        return this.request('/api/get-pixel-value', {
            method: 'POST',
            body: JSON.stringify({
                lat,
                lng,
                model_id: modelId
            })
        });
    }

    // Threshold Operations
    async applyThreshold(imageId, modelId, threshold, colormap) {
        return this.request('/api/apply-threshold', {
            method: 'POST',
            body: JSON.stringify({
                image_id: imageId,
                model_id: modelId,
                threshold,
                colormap
            })
        });
    }

    async applyThresholdRange(imageId, modelId, minThreshold, maxThreshold, colormap) {
        return this.request('/api/apply-threshold-range', {
            method: 'POST',
            body: JSON.stringify({
                image_id: imageId,
                model_id: modelId,
                min_threshold: minThreshold,
                max_threshold: maxThreshold,
                colormap
            })
        });
    }

    // Progress Monitoring
    async getProgress(jobId) {
        return this.request(`/api/progress?job_id=${jobId}`);
    }

    // Get GEE Tile URL
    async getGeeTile(itemId, bbox, geometry = null) {
        return this.request('/api/get-gee-tile', {
            method: 'POST',
            body: JSON.stringify({
                item_id: itemId,
                bbox,
                geometry
            })
        });
    }

    // Get S1 Tile URL
    async getS1Tile(itemId, bbox, geometry = null, bands = null, min = -25, max = 0) {
        return this.request('/api/get-s1-tile', {
            method: 'POST',
            body: JSON.stringify({
                item_id: itemId,
                bbox,
                geometry,
                bands,
                min,
                max
            })
        });
    }

    // Get S2 Custom Tile URL
    async getS2TileCustom(itemId, bbox, geometry = null, bands = null, min = 0, max = 3000) {
        return this.request('/api/get-s2-tile-custom', {
            method: 'POST',
            body: JSON.stringify({
                item_id: itemId,
                bbox,
                geometry,
                bands,
                min,
                max
            })
        });
    }

    // Search S1 Images
    async searchS1Images(bbox, startDate, endDate, geometry = null, limit = 50) {
        return this.request('/api/search-s1-images', {
            method: 'POST',
            body: JSON.stringify({
                bbox,
                geometry,
                start_date: startDate,
                end_date: endDate,
                limit
            })
        });
    }

    // Search S2 Images
    async searchS2Images(bbox, startDate, endDate, cloudCoverMax = 30, geometry = null, limit = 50) {
        return this.request('/api/search-images', {
            method: 'POST',
            body: JSON.stringify({
                bbox,
                geometry,
                start_date: startDate,
                end_date: endDate,
                cloud_cover_max: cloudCoverMax,
                limit
            })
        });
    }

    // Process Spectral Image
    async processSpectralImage(itemId, bbox, geometry = null, spectralMethod = 'ndvi') {
        return this.request('/api/process-spectral-image', {
            method: 'POST',
            body: JSON.stringify({
                item_id: itemId,
                bbox,
                geometry,
                spectral_method: spectralMethod
            })
        });
    }

    // Download S1 Image
    async downloadS1Image(itemId, bbox, geometry = null, asVisualization = false, bands = null) {
        const response = await fetch('/api/download-s1-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                item_id: itemId,
                bbox,
                geometry,
                as_visualization: asVisualization,
                bands
            })
        });
        return response;
    }

    // Download S2 Image
    async downloadS2Image(itemId, bbox, geometry = null, asVisualization = false, bands = null) {
        const response = await fetch('/api/download-s2-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                item_id: itemId,
                bbox,
                geometry,
                as_visualization: asVisualization,
                bands
            })
        });
        return response;
    }
} 