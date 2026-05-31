/**
 * Search Results Module
 * Handles search result rendering and image selection
 */

class SearchResultsController {
    constructor(platformController) {
        this.platform = platformController;
    }

    showSearchResults(images, satellite = null) {
        const sat = satellite || this.platform.selectedSatellite || 's2';
        const containerId = sat === 's1' ? 'inline-results-s1' : 'inline-results-s2';
        const countId = sat === 's1' ? 'results-count-s1' : 'results-count-s2';
        const resultsId = sat === 's1' ? 'search-results-s1' : 'search-results-s2';
        
        const inlineResults = document.getElementById(containerId);
        const resultsCount = document.getElementById(countId);
        const searchResults = document.getElementById(resultsId);

        if (!inlineResults || !resultsCount || !searchResults) {
            console.warn(`Search results elements not found for satellite: ${sat}`);
            return;
        }

        resultsCount.textContent = `${images.length} images found`;
        searchResults.innerHTML = '';

        if (images.length === 0) {
            searchResults.innerHTML = '<p class="no-results">No images match your search criteria. Try adjusting the parameters.</p>';
        } else {
            images.forEach(image => {
                const imageItem = this.createImageItem(image);
                searchResults.appendChild(imageItem);
            });
        }

        inlineResults.classList.remove('hidden');
    }

    createImageItem(image) {
        const item = document.createElement('div');
        item.className = 'image-item';
        item.dataset.imageId = image.id;

        const date = image.datetime ? new Date(image.datetime).toLocaleDateString() : 'Unknown';
        const aoiOverlap = image.aoi_overlap ? (image.aoi_overlap * 100).toFixed(1) : 'N/A';
        const isS1 = image.collection === 'Sentinel-1';
        
        let infoText = '';
        if (isS1) {
            const orbit = image.orbit || 'N/A';
            infoText = `📡 ${orbit}`;
        } else {
            const cloudCover = image.cloud_cover ? image.cloud_cover.toFixed(1) : 'N/A';
            infoText = `☁️ ${cloudCover}%`;
        }

        item.innerHTML = `
            <div class="image-header">
                <div class="image-id">${image.id}</div>
                <div class="cloud-cover">${infoText}</div>
            </div>
            <div class="image-details">
                <div class="label">Date:</div>
                <div class="value">${date}</div>
                <div class="label">AOI Coverage:</div>
                <div class="value">${aoiOverlap}%</div>
                <div class="label">Collection:</div>
                <div class="value">${image.collection || 'Sentinel-2'}</div>
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
                `}
            </div>
        `;

        item.addEventListener('click', () => {
            this.platform.selectImage(image.id);
        });

        const downloadBtn = item.querySelector('.download-btn');
        const downloadOptions = item.querySelectorAll('.download-option');
        const downloadDropdown = item.querySelector('.download-dropdown');
        
        if (downloadBtn && downloadDropdown) {
            downloadBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const options = downloadDropdown.querySelector('.download-options');
                document.querySelectorAll('.download-options').forEach(opt => {
                    if (opt !== options) opt.classList.add('hidden');
                });
                options.classList.toggle('hidden');
            });
        }
        
        downloadOptions.forEach(optBtn => {
            optBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const type = optBtn.dataset.type;
                this.downloadSatelliteImage(image, type);
                const options = optBtn.closest('.download-options');
                if (options) options.classList.add('hidden');
            });
        });

        return item;
    }

    setupDownloadDropdownClose() {
        document.addEventListener('click', () => {
            document.querySelectorAll('.download-options').forEach(opt => {
                opt.classList.add('hidden');
            });
        });
    }

    async downloadSatelliteImage(image, downloadType = 'original') {
        const isS1 = image.collection === 'Sentinel-1';
        const satelliteName = isS1 ? 'Sentinel-1' : 'Sentinel-2';
        const isVisualization = downloadType === 'visualization';
        
        const typeLabel = isVisualization ? 'visualization' : 'raw';
        this.platform.showLoading(`Downloading ${satelliteName} ${typeLabel} image...`);
        
        try {
            const bounds = window.mapManager.getCurrentBounds();
            const geometry = window.mapManager.getCurrentGeoJSON()?.geometry;
            
            const requestBody = {
                item_id: image.id,
                bbox: bounds,
                geometry: geometry,
                as_visualization: isVisualization
            };
            
            if (isVisualization) {
                if (isS1) {
                    const visParams = this.platform.getS1VisualizationParams();
                    requestBody.bands = visParams.bands;
                    requestBody.min = visParams.min;
                    requestBody.max = visParams.max;
                } else {
                    const visParams = this.platform.getS2VisualizationParams();
                    requestBody.bands = visParams.bands;
                    requestBody.min = visParams.min;
                    requestBody.max = visParams.max;
                }
            }
            
            const apiEndpoint = isS1 ? '/api/download-s1-image' : '/api/download-s2-image';
            
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            
            if (response.ok) {
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = `${image.id.split('/').pop()}_${typeLabel}.tif`;
                if (contentDisposition) {
                    const match = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
                    if (match && match[1]) {
                        filename = match[1].replace(/['"]/g, '');
                    }
                }
                
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
                
                this.platform.showNotification(`${satelliteName} ${typeLabel} downloaded: ${filename}`, 'success');
            } else {
                const errorData = await response.json().catch(() => ({ detail: 'Download failed' }));
                throw new Error(errorData.detail || 'Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.platform.showNotification(`Error downloading image: ${error.message}`, 'error');
        } finally {
            this.platform.hideLoading();
        }
    }

    renderCandidateList(images) {
        const candidateList = document.getElementById('candidate-list');
        if (!candidateList) return;

        candidateList.innerHTML = '';

        images.forEach(image => {
            const date = image.datetime ? new Date(image.datetime).toLocaleDateString() : 'Unknown';
            const cloudCover = image.cloud_cover ? image.cloud_cover.toFixed(1) : 'N/A';
            const aoiCoverage = image.aoi_overlap ? (image.aoi_overlap * 100).toFixed(1) : 'N/A';

            const item = document.createElement('div');
            item.className = 'candidate-item';
            item.innerHTML = `
                <input type="checkbox" class="candidate-checkbox" data-image-id="${image.id}">
                <div class="candidate-info">
                    <div class="candidate-date">${date}</div>
                    <div class="candidate-details">☁️ ${cloudCover}% | AOI: ${aoiCoverage}%</div>
                </div>
            `;
            candidateList.appendChild(item);
        });
    }

    updateRunButtonState() {
        const runBtn = document.getElementById('run-change-btn');
        if (!runBtn) return;

        const selectedCount = document.querySelectorAll('.candidate-checkbox:checked').length;
        runBtn.disabled = selectedCount < 2;
        runBtn.textContent = selectedCount >= 2 
            ? `📈 Analyze ${selectedCount} Images` 
            : '📈 Select at least 2 images';
    }

    formatAnalysisData(analysisData) {
        if (!analysisData || analysisData.error) {
            return `<span style="color: #E7344C;">Error: ${analysisData?.error || 'Analysis failed'}</span>`;
        }

        let html = `<strong style="color: #34a853;">📊 ${analysisData.analysis_type}</strong><br>`;

        if (analysisData.resolution) {
            html += `<em style="color: #1976d2;">📏 ${analysisData.resolution}</em><br>`;
        }

        if (analysisData.vegetation_percentage !== undefined) {
            html += `🌿 <strong>Vegetation: ${analysisData.vegetation_percentage.toFixed(1)}%</strong><br>`;
        }

        if (analysisData.ndvi_mean !== undefined) {
            html += `📈 <strong>NDVI: ${analysisData.ndvi_mean.toFixed(3)}</strong><br>`;
        }

        if (analysisData.gndvi_mean !== undefined) {
            html += `🌱 <strong>GNDVI: ${analysisData.gndvi_mean.toFixed(3)}</strong><br>`;
        }

        if (analysisData.evi_mean !== undefined) {
            html += `🌿 <strong>EVI: ${analysisData.evi_mean.toFixed(3)}</strong><br>`;
        }

        if (analysisData.savi_mean !== undefined) {
            html += `📊 <strong>SAVI: ${analysisData.savi_mean.toFixed(3)}</strong><br>`;
        }

        if (analysisData.ndvi_red_edge_mean !== undefined) {
            html += `🔴 <strong>NDVI-RE: ${analysisData.ndvi_red_edge_mean.toFixed(3)}</strong><br>`;
        }

        if (analysisData.ndwi_mean !== undefined) {
            html += `💧 <strong>NDWI: ${analysisData.ndwi_mean.toFixed(3)}</strong><br>`;
        }

        if (analysisData.water_percentage !== undefined) {
            html += `🌊 <strong>Water: ${analysisData.water_percentage.toFixed(1)}%</strong><br>`;
        }

        if (analysisData.mangrove_probability !== undefined) {
            html += `🌲 <strong>Mangrove: ${(analysisData.mangrove_probability * 100).toFixed(1)}%</strong><br>`;
        }

        if (analysisData.classification) {
            html += `🏷️ <strong>${analysisData.classification}</strong><br>`;
        }

        if (analysisData.water_bodies_detected !== undefined) {
            html += `💧 <strong>Water: ${analysisData.water_bodies_detected ? 'Detected' : 'Not detected'}</strong><br>`;
        }

        if (analysisData.moisture_level) {
            html += `💦 <strong>Moisture: ${analysisData.moisture_level}</strong><br>`;
        }

        if (analysisData.total_pixels) {
            html += `<em style="color: #666;">${analysisData.total_pixels.toLocaleString()} pixels analyzed</em><br>`;
        }

        return html;
    }
}
