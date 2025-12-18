/**
 * Main Application Controller
 * Uses the new modular architecture
 */
class MainController {
    constructor() {
        this.apiClient = new ApiClient();
        this.selectedImageId = null;
        this.currentBounds = null;
        
        this.initializeApp();
    }
    
    initializeApp() {
        console.log('🚀 Initializing Mangrove Detection Platform...');
        
        // Initialize tabs
        const tabsContainer = document.querySelector('.tab-navigation');
        if (tabsContainer) {
            UIComponents.initializeTabs(tabsContainer);
        }
        
        // Setup event listeners
        this.setupEventListeners();
        
        console.log('✅ Application initialized');
    }
    
    setupEventListeners() {
        // Search button
        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) {
            searchBtn.addEventListener('click', () => this.handleSearch());
        }
        
        // Process image button
        const processBtn = document.getElementById('process-btn');
        if (processBtn) {
            processBtn.addEventListener('click', () => this.handleProcessImage());
        }
        
        // Change monitoring button
        const changeMonitoringBtn = document.getElementById('change-monitoring-btn');
        if (changeMonitoringBtn) {
            changeMonitoringBtn.addEventListener('click', () => this.handleChangeMonitoring());
        }
    }
    
    async handleSearch() {
        try {
            UIComponents.showLoading('Searching satellite images...');
            
            const bbox = this.getCurrentBounds();
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            const cloudCover = parseFloat(document.getElementById('cloud-cover').value) || 30;
            
            if (!bbox) {
                throw new Error('Please draw an area of interest first');
            }
            
            const results = await this.apiClient.searchImages(bbox, startDate, endDate, cloudCover);
            
            UIComponents.hideLoading();
            UIComponents.showNotification('Search completed successfully!', 'success');
            
            this.displaySearchResults(results.images);
            
        } catch (error) {
            UIComponents.hideLoading();
            UIComponents.showNotification(`Search failed: ${error.message}`, 'error');
            console.error('Search error:', error);
        }
    }
    
    async handleProcessImage() {
        try {
            if (!this.selectedImageId) {
                throw new Error('Please select an image first');
            }
            
            UIComponents.showLoading('Processing satellite image...');
            
            const bbox = this.getCurrentBounds();
            const results = await this.apiClient.processImage(this.selectedImageId, bbox);
            
            UIComponents.hideLoading();
            UIComponents.showNotification('Image processed successfully!', 'success');
            
            this.displayAnalysisResults(results.analysis_results);
            
        } catch (error) {
            UIComponents.hideLoading();
            UIComponents.showNotification(`Processing failed: ${error.message}`, 'error');
            console.error('Processing error:', error);
        }
    }
    
    async handleChangeMonitoring() {
        try {
            UIComponents.showLoading('Starting change monitoring...');
            
            const bbox = this.getCurrentBounds();
            const startDate = document.getElementById('cm-start-date').value;
            const endDate = document.getElementById('cm-end-date').value;
            const frequency = document.getElementById('cm-frequency').value;
            
            if (!bbox) {
                throw new Error('Please draw an area of interest first');
            }
            
            const results = await this.apiClient.startChangeMonitoring({
                bbox,
                start_date: startDate,
                end_date: endDate,
                frequency,
                selected_model: this.getSelectedModel()
            });
            
            UIComponents.hideLoading();
            UIComponents.showNotification('Change monitoring completed!', 'success');
            
            this.displayChangeMonitoringResults(results);
            
        } catch (error) {
            UIComponents.hideLoading();
            UIComponents.showNotification(`Change monitoring failed: ${error.message}`, 'error');
            console.error('Change monitoring error:', error);
        }
    }
    
    displaySearchResults(images) {
        const container = document.getElementById('search-results');
        if (!container) return;
        
        container.innerHTML = '';
        
        images.forEach(image => {
            const imageElement = document.createElement('div');
            imageElement.className = 'image-item';
            imageElement.innerHTML = `
                <div class="image-info">
                    <strong>Date:</strong> ${image.date}<br>
                    <strong>Cloud:</strong> ${image.cloud_coverage.toFixed(1)}%
                </div>
            `;
            
            imageElement.addEventListener('click', () => {
                this.selectImage(image.id);
                // Update UI to show selection
                document.querySelectorAll('.image-item').forEach(item => 
                    item.classList.remove('selected'));
                imageElement.classList.add('selected');
            });
            
            container.appendChild(imageElement);
        });
    }
    
    displayAnalysisResults(results) {
        const container = document.getElementById('analysis-results');
        if (!container) return;
        
        container.innerHTML = '';
        
        results.forEach(result => {
            const resultElement = document.createElement('div');
            resultElement.className = 'analysis-item';
            
            // Create colorbar if available
            const colorbarHtml = result.colormap ? 
                UIComponents.createColorbar(result.colormap) : '';
            
            resultElement.innerHTML = `
                <div class="analysis-info">
                    <strong>${result.name}</strong>
                    <p>Type: ${result.type}</p>
                    ${colorbarHtml}
                </div>
                <div class="analysis-thumbnail">
                    <img src="${result.thumbnail_url}" alt="${result.name}">
                </div>
            `;
            
            container.appendChild(resultElement);
        });
        
        UIComponents.showNotification(`${results.length} analysis results available`, 'info');
    }
    
    displayChangeMonitoringResults(results) {
        const container = document.getElementById('change-monitoring-results');
        if (!container) return;
        
        container.innerHTML = `
            <h3>Change Monitoring Results</h3>
            <p>Total periods: ${results.total_periods}</p>
            <p>Frequency: ${results.frequency}</p>
        `;
        
        if (results.statistics) {
            const stats = results.statistics;
            container.innerHTML += `
                <div class="statistics">
                    <h4>Statistics</h4>
                    <p>Total change: ${stats.total_change_km2?.toFixed(4)} km²</p>
                    <p>Average area: ${stats.average_area_km2?.toFixed(4)} km²</p>
                    <p>Change percentage: ${stats.change_percentage?.toFixed(1)}%</p>
                </div>
            `;
        }
    }
    
    selectImage(imageId) {
        this.selectedImageId = imageId;
        console.log(`Selected image: ${imageId}`);
    }
    
    getCurrentBounds() {
        // This would integrate with the map manager
        return this.currentBounds;
    }
    
    getSelectedModel() {
        // This would get the currently selected model for change monitoring
        return {
            modelId: 'model1',
            type: 'segmentation'
        };
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mainController = new MainController();
}); 