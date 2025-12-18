/**
 * Change Monitoring Module
 * Handles time-series change analysis functionality
 */

class ChangeMonitoringController {
    constructor(platformController) {
        this.platform = platformController;
        this.selectedDates = new Set();
        this.changeImages = [];
        this.analysisResults = null;
        this.spectralMethod = 'ndvi';
    }

    /**
     * Search for images for change monitoring
     */
    async searchChangeImages() {
        const startDate = document.getElementById('change-start-date')?.value;
        const endDate = document.getElementById('change-end-date')?.value;
        const cloudCover = parseInt(document.getElementById('change-cloud-cover')?.value || '30');

        if (!startDate || !endDate) {
            this.platform.showNotification('Please select start and end dates', 'warning');
            return;
        }

        const aoi = this.platform.getAOI();
        if (!aoi) {
            this.platform.showNotification('Please define an AOI first', 'warning');
            return;
        }

        this.platform.showLoading('Searching for images...');

        try {
            const response = await fetch('/api/search-images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    bbox: aoi.bbox,
                    geometry: aoi.geometry,
                    start_date: startDate,
                    end_date: endDate,
                    cloud_cover_max: cloudCover,
                    limit: 100
                })
            });

            if (!response.ok) throw new Error(`Search failed: ${response.status}`);
            const data = await response.json();
            this.changeImages = data.images || [];

            this.renderChangeCalendar();
            this.platform.hideLoading();
            this.platform.showNotification(`Found ${this.changeImages.length} images`, 'success');

        } catch (error) {
            console.error('Change search error:', error);
            this.platform.hideLoading();
            this.platform.showNotification(`Search failed: ${error.message}`, 'error');
        }
    }

    /**
     * Render calendar view of available images
     */
    renderChangeCalendar() {
        const container = document.getElementById('change-calendar');
        if (!container) return;

        container.innerHTML = '';

        if (this.changeImages.length === 0) {
            container.innerHTML = '<p class="no-results">No images found</p>';
            return;
        }

        // Group images by date
        const imagesByDate = {};
        this.changeImages.forEach(image => {
            const date = image.datetime ? image.datetime.split('T')[0] : 'Unknown';
            if (!imagesByDate[date]) {
                imagesByDate[date] = [];
            }
            imagesByDate[date].push(image);
        });

        // Create date items
        const sortedDates = Object.keys(imagesByDate).sort().reverse();
        
        const listContainer = document.createElement('div');
        listContainer.className = 'change-date-list';

        sortedDates.forEach(date => {
            const images = imagesByDate[date];
            const item = this.createDateItem(date, images);
            listContainer.appendChild(item);
        });

        container.appendChild(listContainer);
        this.updateSelectedCount();
    }

    /**
     * Create a date item for the calendar
     */
    createDateItem(date, images) {
        const item = document.createElement('div');
        item.className = 'change-date-item';
        item.dataset.date = date;

        const bestImage = images[0]; // Already sorted by cloud cover
        const cloudCover = bestImage.cloud_cover !== null ? 
            `${bestImage.cloud_cover.toFixed(1)}%` : 'N/A';

        item.innerHTML = `
            <div class="date-checkbox">
                <input type="checkbox" id="date-${date}" data-date="${date}" 
                       data-image-id="${bestImage.id}">
            </div>
            <div class="date-info">
                <div class="date-label">${date}</div>
                <div class="date-meta">☁️ ${cloudCover} | ${images.length} image(s)</div>
            </div>
            <div class="date-actions">
                <button class="btn-small btn-preview" title="Preview">👁️</button>
            </div>
        `;

        // Checkbox handler
        const checkbox = item.querySelector('input[type="checkbox"]');
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                this.selectedDates.add(date);
            } else {
                this.selectedDates.delete(date);
            }
            this.updateSelectedCount();
        });

        // Preview button
        const previewBtn = item.querySelector('.btn-preview');
        previewBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await this.previewImage(bestImage);
        });

        return item;
    }

    /**
     * Preview an image on the map
     */
    async previewImage(image) {
        const aoi = this.platform.getAOI();
        if (!aoi) return;

        try {
            const response = await fetch('/api/get-gee-tile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: image.id,
                    bbox: aoi.bbox,
                    geometry: aoi.geometry
                })
            });

            if (!response.ok) throw new Error('Failed to get tile');
            const data = await response.json();

            if (window.mapManager && data.tile_template) {
                window.mapManager.showTileLayer(
                    `change-preview-${image.id}`,
                    data.tile_template,
                    data.bounds,
                    image.datetime
                );
            }
        } catch (error) {
            console.error('Preview error:', error);
            this.platform.showNotification('Failed to load preview', 'error');
        }
    }

    /**
     * Update selected count display
     */
    updateSelectedCount() {
        const countEl = document.getElementById('change-selected-count');
        if (countEl) {
            countEl.textContent = `${this.selectedDates.size} dates selected`;
        }

        const runBtn = document.getElementById('run-change-analysis');
        if (runBtn) {
            runBtn.disabled = this.selectedDates.size < 2;
        }
    }

    /**
     * Toggle select all dates
     */
    toggleSelectAll() {
        const checkboxes = document.querySelectorAll('.change-date-item input[type="checkbox"]');
        const allSelected = this.selectedDates.size === checkboxes.length;

        checkboxes.forEach(checkbox => {
            checkbox.checked = !allSelected;
            const date = checkbox.dataset.date;
            if (!allSelected) {
                this.selectedDates.add(date);
            } else {
                this.selectedDates.delete(date);
            }
        });

        this.updateSelectedCount();
    }

    /**
     * Run change analysis on selected dates
     */
    async runChangeAnalysis() {
        if (this.selectedDates.size < 2) {
            this.platform.showNotification('Please select at least 2 dates', 'warning');
            return;
        }

        const aoi = this.platform.getAOI();
        if (!aoi) {
            this.platform.showNotification('Please define an AOI first', 'warning');
            return;
        }

        // Get image IDs for selected dates
        const imageIds = [];
        this.changeImages.forEach(image => {
            const date = image.datetime ? image.datetime.split('T')[0] : null;
            if (date && this.selectedDates.has(date)) {
                imageIds.push(image.id);
            }
        });

        const jobId = `change-${Date.now()}`;
        this.platform.showLoading('Running change analysis...');

        try {
            const response = await fetch('/api/change-monitoring', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    bbox: aoi.bbox,
                    geometry: aoi.geometry,
                    image_ids: imageIds,
                    job_id: jobId,
                    model_id: `model2` // NDVI by default
                })
            });

            if (!response.ok) throw new Error(`Analysis failed: ${response.status}`);
            const data = await response.json();
            
            this.analysisResults = data;
            this.displayResults(data);
            this.platform.hideLoading();
            this.platform.showNotification('Change analysis completed', 'success');

        } catch (error) {
            console.error('Change analysis error:', error);
            this.platform.hideLoading();
            this.platform.showNotification(`Analysis failed: ${error.message}`, 'error');
        }
    }

    /**
     * Display analysis results
     */
    displayResults(data) {
        const resultsContainer = document.getElementById('change-results');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = '';
        resultsContainer.style.display = 'block';

        if (!data.results || data.results.length === 0) {
            resultsContainer.innerHTML = '<p class="no-results">No results available</p>';
            return;
        }

        // Statistics summary
        if (data.statistics) {
            const stats = data.statistics;
            const statsHtml = `
                <div class="change-statistics">
                    <h4>Analysis Summary</h4>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <span class="stat-label">Total Change</span>
                            <span class="stat-value">${stats.total_change_km2?.toFixed(4) || 'N/A'} km²</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Max Area</span>
                            <span class="stat-value">${stats.max_area_km2?.toFixed(4) || 'N/A'} km²</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Min Area</span>
                            <span class="stat-value">${stats.min_area_km2?.toFixed(4) || 'N/A'} km²</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Avg ${data.results[0]?.index_name || 'Index'}</span>
                            <span class="stat-value">${stats.avg_spectral?.toFixed(3) || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            `;
            resultsContainer.innerHTML += statsHtml;
        }

        // Time series data
        const timeSeriesHtml = `
            <div class="change-timeseries">
                <h4>Time Series</h4>
                <div class="timeseries-list">
                    ${data.results.map(result => `
                        <div class="timeseries-item" data-date="${result.date}" data-image-id="${result.image_id}">
                            <div class="ts-date">${result.date}</div>
                            <div class="ts-values">
                                <span class="ts-area">${result.area_km2?.toFixed(4)} km²</span>
                                <span class="ts-index">${result.index_name}: ${result.mean_spectral?.toFixed(3)}</span>
                            </div>
                            <button class="btn-small btn-view-ts" title="View on map">👁️</button>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        resultsContainer.innerHTML += timeSeriesHtml;

        // Add view handlers
        resultsContainer.querySelectorAll('.btn-view-ts').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const item = btn.closest('.timeseries-item');
                const imageId = item.dataset.imageId;
                await this.viewTimeseriesImage(imageId);
            });
        });
    }

    /**
     * View a specific timeseries image
     */
    async viewTimeseriesImage(imageId) {
        const aoi = this.platform.getAOI();
        if (!aoi) return;

        try {
            const response = await fetch('/api/process-spectral-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: imageId,
                    bbox: aoi.bbox,
                    geometry: aoi.geometry,
                    spectral_method: this.spectralMethod
                })
            });

            if (!response.ok) throw new Error('Failed to get spectral image');
            const data = await response.json();

            if (window.mapManager && data.url) {
                const bounds = [
                    [data.bounds[1], data.bounds[0]],
                    [data.bounds[3], data.bounds[2]]
                ];
                window.mapManager.showAnalysisLayer(
                    `spectral-${imageId}`,
                    data.url,
                    `${this.spectralMethod.toUpperCase()} - ${imageId}`,
                    bounds
                );
            }
        } catch (error) {
            console.error('View spectral error:', error);
            this.platform.showNotification('Failed to load spectral image', 'error');
        }
    }

    /**
     * Download results as CSV
     */
    downloadCSV() {
        if (!this.analysisResults || !this.analysisResults.results) {
            this.platform.showNotification('No results to download', 'warning');
            return;
        }

        const headers = ['Date', 'Image ID', 'Area (km²)', 'Mean Index', 'Index Name'];
        const rows = this.analysisResults.results.map(r => [
            r.date,
            r.image_id,
            r.area_km2?.toFixed(6) || '',
            r.mean_spectral?.toFixed(6) || '',
            r.index_name || ''
        ]);

        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `change_analysis_${Date.now()}.csv`;
        link.click();
    }

    /**
     * Set spectral method
     */
    setSpectralMethod(method) {
        this.spectralMethod = method;
    }

    /**
     * Clear results
     */
    clearResults() {
        this.selectedDates.clear();
        this.analysisResults = null;
        
        const resultsContainer = document.getElementById('change-results');
        if (resultsContainer) resultsContainer.style.display = 'none';
        
        this.updateSelectedCount();
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.ChangeMonitoringController = ChangeMonitoringController;
}

