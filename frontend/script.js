/**
 * ===== MANGROVE DETECTION PLATFORM =====
 * Google Maps Style Interface Controller
 */

class PlatformController {
    constructor() {
        // 상대 경로 사용으로 CORS 문제 방지
        this.apiBaseUrl = '';  // 빈 문자열로 설정하면 같은 호스트의 상대 경로 사용

        console.log(`🔗 API Base URL: ${this.apiBaseUrl || 'Same origin (relative)'}`);

        this.currentImages = [];
        this.s1Images = [];  // Separate storage for Sentinel-1 results
        this.s2Images = [];  // Separate storage for Sentinel-2 results
        this.selectedImageId = null;
        this.currentNotification = null;
        
        // Compare mode state
        this.compareMode = false;
        this.compareS1Layer = null;
        this.compareS2Layer = null;
        this.sideBySideControl = null;

        // Initialize modular controllers
        this.initModules();

        this.init();
    }

    /**
     * Initialize modular controllers for better code organization
     */
    initModules() {
        // Initialize API client if available
        if (typeof ApiClient !== 'undefined') {
            this.apiClient = new ApiClient(this.apiBaseUrl);
        }

        // Initialize image search controller if available
        if (typeof ImageSearchController !== 'undefined') {
            this.imageSearch = new ImageSearchController(this);
        }

        // Initialize analysis controller if available
        if (typeof AnalysisController !== 'undefined') {
            this.analysisController = new AnalysisController(this);
        }

        // Initialize change monitoring controller if available
        if (typeof ChangeMonitoringController !== 'undefined') {
            this.changeMonitoring = new ChangeMonitoringController(this);
        }

        // Initialize custom visualization controller if available
        if (typeof CustomVisualizationController !== 'undefined') {
            this.customViz = new CustomVisualizationController(this);
        }

        // Initialize threshold controller if available
        if (typeof ThresholdController !== 'undefined') {
            this.thresholdController = new ThresholdController(this);
        }

        // Initialize target detection controller if available
        if (typeof TargetDetectionController !== 'undefined') {
            this.targetDetection = new TargetDetectionController(this);
        }

        console.log('📦 Modular controllers initialized');
    }

    async init() {
        console.log('🚀 Initializing Platform Controller...');

        // Load configuration
        await this.loadConfig();

        // Setup event listeners
        this.setupEventListeners();
        
        // Setup download dropdown close
        this.setupDownloadDropdownClose();

        // Initialize date inputs
        this.initializeDateInputs();

        console.log('✅ Platform Controller Ready!');
    }

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            if (response.ok) {
                const config = await response.json();

                // Update data source info (if element exists)
                const dataSourceElement = document.getElementById('data-source-info');
                if (dataSourceElement) {
                    if (config.use_copernicus && config.use_public_stac) {
                        dataSourceElement.textContent = 'Data: Copernicus + Public STAC';
                    } else if (config.use_copernicus) {
                        dataSourceElement.textContent = 'Data: Copernicus STAC';
                    } else {
                        dataSourceElement.textContent = 'Data: Public STAC';
                    }
                }

                console.log('✅ Configuration loaded:', config);
            }
        } catch (error) {
            console.error('Failed to load configuration:', error);
            // this.showNotification('Failed to load configuration', 'error');
        }
    }

    setupEventListeners() {
        // Map drawing controls
        document.getElementById('draw-rectangle').addEventListener('click', () => {
            if (window.mapManager) {
                window.mapManager.startDrawing();
                this.showNotification('Click and drag on the map to draw AOI', 'info');
            } else {
                console.error('❌ MapManager not initialized yet');
                this.showNotification('Map is not ready. Please wait...', 'error');
            }
        });

        document.getElementById('clear-drawings').addEventListener('click', () => {
            if (window.mapManager) {
                window.mapManager.clearDrawings();
                this.resetSearchState();
                this.showNotification('AOI cleared', 'info');
            } else {
                console.error('❌ MapManager not initialized yet');
                this.showNotification('Map is not ready. Please wait...', 'error');
            }
        });

        // SHP Upload controls
        const shpUploadInput = document.getElementById('shp-upload');
        const dropZone = document.getElementById('drop-zone');

        if (dropZone && shpUploadInput) {
            // Click to upload
            dropZone.addEventListener('click', () => {
                shpUploadInput.click();
            });

            // Drag and Drop events
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, unhighlight, false);
            });

            function highlight(e) {
                dropZone.classList.add('dragover');
            }

            function unhighlight(e) {
                dropZone.classList.remove('dragover');
            }

            dropZone.addEventListener('drop', async (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                handleFiles(files);
            });

            shpUploadInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });

            const handleFiles = async (files) => {
                if (files.length > 0) {
                    const file = files[0];
                    const fileName = file.name.toLowerCase();

                    // Validate file type (must be zip or kml)
                    const isZip = fileName.endsWith('.zip');
                    const isKml = fileName.endsWith('.kml');
                    
                    if (!isZip && !isKml) {
                        this.showNotification('Please upload a .zip (Shapefile) or .kml file', 'error');
                        return;
                    }

                    if (window.mapManager) {
                        const fileType = isKml ? 'KML' : 'SHP';
                        this.showLoading(`Parsing ${fileType} file...`);
                        try {
                            if (isKml) {
                                await window.mapManager.loadKmlFile(file);
                            } else {
                                await window.mapManager.loadShpFile(file);
                            }
                            this.showNotification(`${fileType} file loaded successfully`, 'success');

                            // Update drop zone text
                            const dropText = dropZone.querySelector('.drop-text');
                            if (dropText) dropText.textContent = file.name;

                        } catch (error) {
                            console.error(`${fileType} Load Error:`, error);
                            this.showNotification(`Failed to load ${fileType} file: ` + error.message, 'error');
                        } finally {
                            this.hideLoading();
                            // Reset input
                            shpUploadInput.value = '';
                        }
                    }
                }
            };
        }

        // Search controls
        document.getElementById('cloud-cover').addEventListener('input', (e) => {
            document.getElementById('cloud-cover-value').textContent = `${e.target.value}%`;
        });

        document.getElementById('search-btn').addEventListener('click', () => this.searchImages());

        // Satellite selector buttons
        document.getElementById('satellite-s2-btn')?.addEventListener('click', () => {
            this.selectedSatellite = 's2';
            document.getElementById('satellite-s2-btn').classList.add('active');
            document.getElementById('satellite-s1-btn').classList.remove('active');
            // Show cloud cover for S2
            const cloudCoverRow = document.getElementById('cloud-cover')?.closest('.param-row');
            if (cloudCoverRow) cloudCoverRow.style.display = 'flex';
            // Show S2 options, hide S1 options
            document.getElementById('s2-channel-options')?.classList.remove('hidden');
            document.getElementById('s1-channel-options')?.classList.add('hidden');
            // Switch displayed results
            this.switchSatelliteResults('s2');
        });

        document.getElementById('satellite-s1-btn')?.addEventListener('click', () => {
            this.selectedSatellite = 's1';
            document.getElementById('satellite-s1-btn').classList.add('active');
            document.getElementById('satellite-s2-btn').classList.remove('active');
            // Hide cloud cover for S1
            const cloudCoverRow = document.getElementById('cloud-cover')?.closest('.param-row');
            if (cloudCoverRow) cloudCoverRow.style.display = 'none';
            // Hide S2 options, show S1 options
            document.getElementById('s2-channel-options')?.classList.add('hidden');
            document.getElementById('s1-channel-options')?.classList.remove('hidden');
            // Switch displayed results
            this.switchSatelliteResults('s1');
        });

        // Compare mode controls
        document.getElementById('compare-btn')?.addEventListener('click', () => this.startCompareMode());
        document.getElementById('compare-exit-btn')?.addEventListener('click', () => this.exitCompareMode());
        document.getElementById('compare-s1-select')?.addEventListener('change', () => this.updateCompareButtonState());
        document.getElementById('compare-s2-select')?.addEventListener('change', () => this.updateCompareButtonState());

        // S1 preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = btn.dataset.preset;
                this.applyS1Preset(preset);
                // Update active state
                document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Initialize satellite selection and S1/S2 channels
        this.selectedSatellite = 's2';
        this.s1Channels = { r: 'VV', g: 'VH', b: 'VV' };
        this.s1MinMax = { min: -25, max: 0 };
        this.s2Channels = { r: 'B4', g: 'B3', b: 'B2' };
        this.s2MinMax = { min: 0, max: 3000 };

        // Apply S1 visualization button
        document.getElementById('apply-s1-vis-btn')?.addEventListener('click', () => {
            this.applyS1VisualizationToCurrentImage();
        });

        // Apply S2 visualization button
        document.getElementById('apply-s2-vis-btn')?.addEventListener('click', () => {
            this.applyS2VisualizationToCurrentImage();
        });

        // S2 preset buttons
        document.querySelectorAll('.s2-preset').forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = btn.dataset.preset;
                this.applyS2Preset(preset);
                // Update active state
                document.querySelectorAll('.s2-preset').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Change Monitoring Events
        // Change Monitoring Events - Old UI removed
        // document.getElementById('search-candidates-btn')?.addEventListener('click', () => this.searchChangeCandidates());
        // document.getElementById('run-change-btn')?.addEventListener('click', () => this.runChangeMonitoring());

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabId = e.target.dataset.tab;
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                e.target.classList.add('active');
                document.getElementById(tabId + '-tab-content').classList.add('active');
            });
        });

        // Change monitoring controls
        // Change monitoring controls
        // Removed old start-change-monitoring listener as it is replaced by new flow

        // Panel controls
        const closeResultsBtn = document.getElementById('close-results');
        if (closeResultsBtn) {
            closeResultsBtn.addEventListener('click', () => {
                this.hidePanel('results-panel');
            });
        }

        const closeAnalysisBtn = document.getElementById('close-analysis');
        if (closeAnalysisBtn) {
            closeAnalysisBtn.addEventListener('click', () => {
                this.hidePanel('analysis-panel');
            });
        }

        // Tab controls
        this.setupTabControls();



        document.getElementById('prev-month')?.addEventListener('click', () => {
            if (this.changeMonitoringContext) {
                this.changeMonitoringContext.currentMonth--;
                if (this.changeMonitoringContext.currentMonth < 0) {
                    this.changeMonitoringContext.currentMonth = 11;
                    this.changeMonitoringContext.currentYear--;
                }
                this.renderCalendar(this.changeMonitoringContext.currentYear, this.changeMonitoringContext.currentMonth, this.changeMonitoringContext.availableImages);
                this.fetchCalendarData(this.changeMonitoringContext.currentYear, this.changeMonitoringContext.currentMonth);
            }
        });

        document.getElementById('next-month')?.addEventListener('click', () => {
            if (this.changeMonitoringContext) {
                this.changeMonitoringContext.currentMonth++;
                if (this.changeMonitoringContext.currentMonth > 11) {
                    this.changeMonitoringContext.currentMonth = 0;
                    this.changeMonitoringContext.currentYear++;
                }
                this.renderCalendar(this.changeMonitoringContext.currentYear, this.changeMonitoringContext.currentMonth, this.changeMonitoringContext.availableImages);
                this.fetchCalendarData(this.changeMonitoringContext.currentYear, this.changeMonitoringContext.currentMonth);
            }
        });

        document.getElementById('run-context-monitoring')?.addEventListener('click', () => {
            this.runContextualChangeMonitoring();
        });

        // Change Monitoring - Cloud Cover Slider
        document.getElementById('change-cloud-cover')?.addEventListener('input', (e) => {
            document.getElementById('change-cloud-cover-value').textContent = `${e.target.value}%`;
        });

        // Change Monitoring - AOI Coverage Slider
        document.getElementById('change-aoi-coverage')?.addEventListener('input', (e) => {
            document.getElementById('change-aoi-coverage-value').textContent = `${e.target.value}%`;
        });

        // Change Monitoring - Search Images Button
        document.getElementById('search-change-images-btn')?.addEventListener('click', () => {
            this.searchChangeImages();
        });

        // Change Monitoring - Run Analysis Button
        document.getElementById('run-change-analysis-btn')?.addEventListener('click', () => {
            this.runChangeAnalysis();
        });

        // Change Monitoring - Select All Button
        document.getElementById('select-all-dates-btn')?.addEventListener('click', () => {
            this.toggleSelectAllDates();
        });

        // Change Monitoring - Download CSV Button
        document.getElementById('download-csv-btn')?.addEventListener('click', () => {
            this.downloadChangeAnalysisCSV();
        });

        // AOI change listener
        const updateAOIButtons = (hasAOI) => {
            const searchBtn = document.getElementById('search-btn');
            const candidatesBtn = document.getElementById('search-candidates-btn');
            const changeImagesBtn = document.getElementById('search-change-images-btn');
            
            if (searchBtn) searchBtn.disabled = !hasAOI;
            if (candidatesBtn) candidatesBtn.disabled = !hasAOI;
            if (changeImagesBtn) changeImagesBtn.disabled = !hasAOI;
            
            if (hasAOI) {
                this.showNotification('AOI defined. You can now search for images.', 'success');
            }
        };
        
        if (window.mapManager) {
            window.mapManager.onAOIChange = updateAOIButtons;
        } else {
            // Set up the callback later when mapManager is ready
            setTimeout(() => {
                if (window.mapManager) {
                    window.mapManager.onAOIChange = updateAOIButtons;
                }
            }, 200);
        }
    }

    initializeDateInputs() {
        const today = new Date();
        const oneMonthAgo = new Date(today.getFullYear(), today.getMonth() - 1, today.getDate());
        const sixMonthsAgo = new Date(today.getFullYear(), today.getMonth() - 6, today.getDate());

        // Search tab dates
        document.getElementById('start-date').value = oneMonthAgo.toISOString().split('T')[0];
        document.getElementById('end-date').value = today.toISOString().split('T')[0];

        // Change monitoring dates (6 months period)
        // document.getElementById('monitoring-start-date').value = sixMonthsAgo.toISOString().split('T')[0];
        // document.getElementById('monitoring-end-date').value = today.toISOString().split('T')[0];

        // Candidate search dates
        document.getElementById('change-start-date').value = sixMonthsAgo.toISOString().split('T')[0];
        document.getElementById('change-end-date').value = today.toISOString().split('T')[0];
    }

    resetSearchState() {
        this.currentImages = [];
        this.s1Images = [];
        this.s2Images = [];
        this.selectedImageId = null;
        document.getElementById('search-btn').disabled = true;
        document.getElementById('search-candidates-btn').disabled = true;
        document.getElementById('process-image').disabled = true;
        this.hidePanel('results-panel');
        this.hidePanel('processing-panel');

        // Hide search result panels
        document.getElementById('inline-results-s1')?.classList.add('hidden');
        document.getElementById('inline-results-s2')?.classList.add('hidden');
        document.getElementById('compare-section')?.classList.add('hidden');

        // Reset analysis results to empty state
        this.showNoAnalysisResults();

        // Reset change monitoring results
        this.hideChangeMonitoringResults();
        
        // Exit compare mode if active
        this.exitCompareMode();
    }

    // Switch displayed results based on satellite selection
    switchSatelliteResults(satellite) {
        const s1Results = document.getElementById('inline-results-s1');
        const s2Results = document.getElementById('inline-results-s2');
        
        if (satellite === 's1') {
            s1Results?.classList.remove('hidden');
            s2Results?.classList.add('hidden');
            this.currentImages = this.s1Images;
        } else {
            s2Results?.classList.remove('hidden');
            s1Results?.classList.add('hidden');
            this.currentImages = this.s2Images;
        }
    }

    // Update compare select dropdowns with available images
    updateCompareSelects() {
        const s1Select = document.getElementById('compare-s1-select');
        const s2Select = document.getElementById('compare-s2-select');
        
        if (!s1Select || !s2Select) return;
        
        // Update S1 select with AOI coverage
        s1Select.innerHTML = '<option value="">-- Select S1 Image --</option>';
        this.s1Images.forEach(img => {
            const date = img.datetime ? img.datetime.split('T')[0] : 'Unknown';
            const orbit = img.orbit || 'N/A';
            const aoiCoverage = img.aoi_overlap ? `AOI:${(img.aoi_overlap * 100).toFixed(0)}%` : '';
            const option = document.createElement('option');
            option.value = img.id;
            option.textContent = `${date} | ${orbit} | ${aoiCoverage}`;
            s1Select.appendChild(option);
        });
        
        // Update S2 select with AOI coverage
        s2Select.innerHTML = '<option value="">-- Select S2 Image --</option>';
        this.s2Images.forEach(img => {
            const date = img.datetime ? img.datetime.split('T')[0] : 'Unknown';
            const cloud = img.cloud_cover ? `☁${img.cloud_cover.toFixed(0)}%` : '';
            const aoiCoverage = img.aoi_overlap ? `AOI:${(img.aoi_overlap * 100).toFixed(0)}%` : '';
            const option = document.createElement('option');
            option.value = img.id;
            option.textContent = `${date} | ${cloud} | ${aoiCoverage}`;
            s2Select.appendChild(option);
        });
        
        this.updateCompareButtonState();
    }

    // Show compare section if both satellites have results
    showCompareSection() {
        const section = document.getElementById('compare-section');
        if (!section) return;
        
        if (this.s1Images.length > 0 && this.s2Images.length > 0) {
            section.classList.remove('hidden');
        } else if (this.s1Images.length > 0 || this.s2Images.length > 0) {
            // Show but indicate one satellite needs searching
            section.classList.remove('hidden');
        }
    }

    // Update compare button state
    updateCompareButtonState() {
        const s1Select = document.getElementById('compare-s1-select');
        const s2Select = document.getElementById('compare-s2-select');
        const compareBtn = document.getElementById('compare-btn');
        
        if (!compareBtn) return;
        
        const s1Selected = s1Select?.value;
        const s2Selected = s2Select?.value;
        
        compareBtn.disabled = !(s1Selected && s2Selected);
    }

    // Start side-by-side comparison mode
    async startCompareMode() {
        const s1Id = document.getElementById('compare-s1-select')?.value;
        const s2Id = document.getElementById('compare-s2-select')?.value;
        
        if (!s1Id || !s2Id) {
            this.showNotification('Please select both S1 and S2 images', 'error');
            return;
        }
        
        this.showLoading('Loading comparison view...');
        
        try {
            const bounds = window.mapManager.getCurrentBounds();
            const geometry = window.mapManager.getCurrentGeoJSON()?.geometry;
            
            // Get S2 tile with current visualization params
            const s2VisParams = this.getS2VisualizationParams();
            const s2Response = await fetch('/api/get-s2-tile-custom', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    item_id: s2Id, 
                    bbox: bounds, 
                    geometry,
                    bands: s2VisParams.bands,
                    min: s2VisParams.min,
                    max: s2VisParams.max
                })
            });
            
            if (!s2Response.ok) throw new Error('Failed to get S2 tile');
            const s2Data = await s2Response.json();
            
            // Get S1 tile with current visualization params
            const s1VisParams = this.getS1VisualizationParams();
            const s1Response = await fetch('/api/get-s1-tile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: s1Id,
                    bbox: bounds,
                    geometry,
                    bands: s1VisParams.bands,
                    min: s1VisParams.min,
                    max: s1VisParams.max
                })
            });
            
            if (!s1Response.ok) throw new Error('Failed to get S1 tile');
            const s1Data = await s1Response.json();
            
            // Clear existing layers
            if (window.mapManager) {
                window.mapManager.clearImageLayers();
            }
            
            // Create tile layers for comparison
            const aoiBounds = window.mapManager.getCombinedAOIBounds();
            const leafletBounds = [
                [aoiBounds.getSouth(), aoiBounds.getWest()],
                [aoiBounds.getNorth(), aoiBounds.getEast()]
            ];
            
            // Create S2 layer (left side)
            this.compareS2Layer = L.tileLayer(s2Data.tile_template, {
                bounds: leafletBounds,
                maxZoom: 18,
                tileSize: 256
            });
            
            // Create S1 layer (right side)
            this.compareS1Layer = L.tileLayer(s1Data.tile_template, {
                bounds: leafletBounds,
                maxZoom: 18,
                tileSize: 256
            });
            
            // Add layers to map
            this.compareS2Layer.addTo(window.mapManager.map);
            this.compareS1Layer.addTo(window.mapManager.map);
            
            // Create side-by-side control
            try {
                if (L && L.control && L.control.sideBySide) {
                    this.sideBySideControl = L.control.sideBySide(this.compareS2Layer, this.compareS1Layer);
                    this.sideBySideControl.addTo(window.mapManager.map);
                    console.log('✅ Side-by-side control added successfully');
                } else {
                    throw new Error('Plugin not available');
                }
            } catch (pluginError) {
                console.error('Side-by-side plugin error:', pluginError);
                this.showNotification('Drag the slider in the center to compare images!', 'info');
            }
            
            // Update UI
            this.compareMode = true;
            document.getElementById('compare-btn')?.classList.add('hidden');
            document.getElementById('compare-exit-btn')?.classList.remove('hidden');
            
            // Add comparison mode indicator
            this.showCompareModeIndicator();
            
            // Fit to AOI bounds
            window.mapManager.map.fitBounds(aoiBounds);
            window.mapManager.outlineAOI();
            
            this.showNotification('Compare mode active. Drag the slider to compare S1 and S2!', 'success');
            
        } catch (error) {
            console.error('Error starting compare mode:', error);
            this.showNotification('Failed to start comparison mode', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // Exit comparison mode
    exitCompareMode() {
        if (!this.compareMode) return;
        
        // Remove side-by-side control
        if (this.sideBySideControl && window.mapManager?.map) {
            window.mapManager.map.removeControl(this.sideBySideControl);
            this.sideBySideControl = null;
        }
        
        // Remove comparison layers
        if (this.compareS1Layer && window.mapManager?.map) {
            window.mapManager.map.removeLayer(this.compareS1Layer);
            this.compareS1Layer = null;
        }
        if (this.compareS2Layer && window.mapManager?.map) {
            window.mapManager.map.removeLayer(this.compareS2Layer);
            this.compareS2Layer = null;
        }
        
        // Update UI
        this.compareMode = false;
        document.getElementById('compare-btn')?.classList.remove('hidden');
        document.getElementById('compare-exit-btn')?.classList.add('hidden');
        
        // Remove indicator
        this.hideCompareModeIndicator();
        
        this.showNotification('Exited compare mode', 'info');
    }

    // Show comparison mode indicator
    showCompareModeIndicator() {
        // Remove existing indicator
        this.hideCompareModeIndicator();
        
        const indicator = document.createElement('div');
        indicator.id = 'compare-mode-indicator';
        indicator.className = 'compare-mode-indicator';
        indicator.innerHTML = `
            <span class="side-label">◀ Sentinel-2 (Optical)</span>
            <span class="divider"></span>
            <span class="side-label">Sentinel-1 (SAR) ▶</span>
        `;
        document.body.appendChild(indicator);
    }

    // Hide comparison mode indicator
    hideCompareModeIndicator() {
        const indicator = document.getElementById('compare-mode-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    showPanel(panelId) {
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.remove('hidden');
            panel.classList.add('slide-in');
            setTimeout(() => panel.classList.remove('slide-in'), 300);
        }
    }

    hidePanel(panelId) {
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.add('slide-out');
            setTimeout(() => {
                panel.classList.add('hidden');
                panel.classList.remove('slide-out');
            }, 300);
        }
    }

    showNotification(message, type = 'info', duration = 5000) {
        // Clear existing notification
        if (this.currentNotification) {
            clearTimeout(this.currentNotification.timeout);
        }

        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type} show`;

        this.currentNotification = {
            timeout: setTimeout(() => {
                notification.classList.remove('show');
                this.currentNotification = null;
            }, duration)
        };
    }

    showLoading(message = 'Processing...') {
        console.log(`[LOADING] Showing overlay: ${message}`);
        const overlay = document.getElementById('loading-overlay');
        const messageElement = document.getElementById('loading-message');
        const bar = document.getElementById('loading-progress-bar');
        const text = document.getElementById('loading-progress-text');

        if (!overlay) {
            console.error('[LOADING] Overlay element not found!');
            return;
        }

        if (messageElement) messageElement.textContent = message;
        if (bar) {
            bar.style.width = '0%';
            console.log('[LOADING] Reset progress bar to 0%');
        }
        if (text) text.textContent = '0%';
        overlay.classList.remove('hidden');
        console.log('[LOADING] Overlay displayed');
    }



    updateLoadingProgress(percent, note = '') {
        const bar = document.getElementById('loading-progress-bar');
        const text = document.getElementById('loading-progress-text');
        const targetPercent = Math.max(0, Math.min(100, Math.round(percent)));

        console.log(`[PROGRESS] Updating to ${targetPercent}% - ${note}`);

        if (!bar || !text) {
            console.error('[PROGRESS] Elements not found!', { bar: !!bar, text: !!text });
            return;
        }

        // 즉시 업데이트 - CSS transition이 부드럽게 처리
        bar.style.width = `${targetPercent}%`;
        text.textContent = `${targetPercent}% ${note ? ' - ' + note : ''}`;

        console.log(`[PROGRESS] Updated bar width to: ${bar.style.width}`);
    }



    hideLoading() {
        console.log('[LOADING] Hiding overlay');
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
            console.log('[LOADING] Overlay hidden');
        } else {
            console.error('[LOADING] Overlay element not found!');
        }
        this._progressStop = true; // polling 중지
    }

    applyS1Preset(preset) {
        const rSelect = document.getElementById('s1-channel-r');
        const gSelect = document.getElementById('s1-channel-g');
        const bSelect = document.getElementById('s1-channel-b');
        const minInput = document.getElementById('s1-min');
        const maxInput = document.getElementById('s1-max');

        switch (preset) {
            case 'vv':
                rSelect.value = 'VV'; gSelect.value = 'VV'; bSelect.value = 'VV';
                minInput.value = -25; maxInput.value = 0;
                break;
            case 'vh':
                rSelect.value = 'VH'; gSelect.value = 'VH'; bSelect.value = 'VH';
                minInput.value = -30; maxInput.value = -5;
                break;
            case 'vvvh':
                rSelect.value = 'VV'; gSelect.value = 'VH'; bSelect.value = 'VV';
                minInput.value = -25; maxInput.value = 0;
                break;
            case 'ratio':
                rSelect.value = 'VH'; gSelect.value = 'VV'; bSelect.value = 'angle';
                minInput.value = -25; maxInput.value = 0;
                break;
        }

        this.s1Channels = { r: rSelect.value, g: gSelect.value, b: bSelect.value };
        this.s1MinMax = { min: parseFloat(minInput.value), max: parseFloat(maxInput.value) };
    }

    getS1VisualizationParams() {
        const r = document.getElementById('s1-channel-r')?.value || 'VV';
        const g = document.getElementById('s1-channel-g')?.value || 'VH';
        const b = document.getElementById('s1-channel-b')?.value || 'VV';
        const min = parseFloat(document.getElementById('s1-min')?.value) || -25;
        const max = parseFloat(document.getElementById('s1-max')?.value) || 0;
        
        return { bands: [r, g, b], min, max };
    }

    getS2VisualizationParams() {
        const r = document.getElementById('s2-channel-r')?.value || 'B4';
        const g = document.getElementById('s2-channel-g')?.value || 'B3';
        const b = document.getElementById('s2-channel-b')?.value || 'B2';
        const min = parseFloat(document.getElementById('s2-min')?.value) || 0;
        const max = parseFloat(document.getElementById('s2-max')?.value) || 3000;
        
        return { bands: [r, g, b], min, max };
    }

    applyS2Preset(preset) {
        const presets = {
            'truecolor': { r: 'B4', g: 'B3', b: 'B2', min: 0, max: 3000 },
            'falsecolor': { r: 'B8', g: 'B4', b: 'B3', min: 0, max: 5000 },
            'agriculture': { r: 'B11', g: 'B8', b: 'B2', min: 0, max: 5000 },
            'swir': { r: 'B12', g: 'B8A', b: 'B4', min: 0, max: 5000 }
        };
        
        const config = presets[preset] || presets['truecolor'];
        
        document.getElementById('s2-channel-r').value = config.r;
        document.getElementById('s2-channel-g').value = config.g;
        document.getElementById('s2-channel-b').value = config.b;
        document.getElementById('s2-min').value = config.min;
        document.getElementById('s2-max').value = config.max;
    }

    async applyS2VisualizationToCurrentImage() {
        // Check if there's a currently selected S2 image
        if (!this.selectedImageId) {
            this.showNotification('Please select a Sentinel-2 image first', 'error');
            return;
        }

        let image = this.currentImages?.find(img => img.id === this.selectedImageId);
        if (!image) {
            image = this.s2Images?.find(img => img.id === this.selectedImageId);
        }
        if (!image || image.collection === 'Sentinel-1') {
            this.showNotification('Selected image is not a Sentinel-2 image', 'error');
            return;
        }

        this.showNotification('Applying new visualization...', 'info');

        try {
            const visParams = this.getS2VisualizationParams();
            
            const response = await fetch(`${this.apiBaseUrl}/api/get-s2-tile-custom`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: this.selectedImageId,
                    bbox: window.mapManager.getCurrentBounds(),
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    bands: visParams.bands,
                    min: visParams.min,
                    max: visParams.max
                })
            });

            if (response.ok) {
                const tileData = await response.json();
                
                // Update the image layer with new visualization
                image.tile_template = tileData.tile_template;
                image.tile_bounds = tileData.bounds;
                image.display_type = 'tile';

                if (window.mapManager) {
                    // Remove old layer and add new one
                    window.mapManager.clearImageLayers();
                    window.mapManager.addImageLayer(this.selectedImageId, image);
                    window.mapManager.outlineAOI();
                    this.showNotification('Visualization updated!', 'success');
                }
            } else {
                throw new Error('Failed to apply visualization');
            }
        } catch (error) {
            console.error('Error applying S2 visualization:', error);
            this.showNotification('Error applying visualization', 'error');
        }
    }

    async applyS1VisualizationToCurrentImage() {
        // Check if there's a currently selected S1 image
        if (!this.selectedImageId) {
            this.showNotification('Please select a Sentinel-1 image first', 'error');
            return;
        }

        const image = this.currentImages?.find(img => img.id === this.selectedImageId);
        if (!image || image.collection !== 'Sentinel-1') {
            this.showNotification('Selected image is not a Sentinel-1 image', 'error');
            return;
        }

        this.showNotification('Applying new visualization...', 'info');

        try {
            const visParams = this.getS1VisualizationParams();
            
            const response = await fetch(`${this.apiBaseUrl}/api/get-s1-tile`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: this.selectedImageId,
                    bbox: window.mapManager.getCurrentBounds(),
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    bands: visParams.bands,
                    min: visParams.min,
                    max: visParams.max
                })
            });

            if (response.ok) {
                const tileData = await response.json();
                
                // Update the image layer with new visualization
                image.tile_template = tileData.tile_template;
                image.tile_bounds = tileData.bounds;
                image.display_type = 'tile';

                if (window.mapManager) {
                    // Remove old layer and add new one
                    window.mapManager.clearImageLayers();
                    window.mapManager.addImageLayer(this.selectedImageId, image);
                    window.mapManager.outlineAOI();
                    this.showNotification('Visualization updated!', 'success');
                }
            } else {
                throw new Error('Failed to apply visualization');
            }
        } catch (error) {
            console.error('Error applying S1 visualization:', error);
            this.showNotification('Error applying visualization', 'error');
        }
    }

    async searchImages() {
        // Delegate to ImageSearchController if available, otherwise use legacy implementation
        if (this.imageSearch && typeof this.imageSearch.searchImagesLegacy === 'function') {
            return await this.imageSearch.searchImagesLegacy(this);
        }
        
        // Legacy implementation for backward compatibility
        if (!window.mapManager || !window.mapManager.getCurrentBounds()) {
            this.showNotification('Please draw an AOI first', 'error');
            return;
        }

        const satellite = this.selectedSatellite || 's2';
        const satelliteName = satellite === 's1' ? 'Sentinel-1 (SAR)' : 'Sentinel-2 (Optical)';
        this.showLoading(`Searching for ${satelliteName} images...`);

        try {
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            const cloudCover = document.getElementById('cloud-cover').value;
            const bounds = window.mapManager.getCurrentBounds();
            const apiEndpoint = satellite === 's1' ? '/api/search-s1-images' : '/api/search-images';
            
            const requestBody = {
                bbox: bounds,
                geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                start_date: startDate,
                end_date: endDate,
                limit: 20
            };
            if (satellite === 's2') requestBody.cloud_cover_max = parseInt(cloudCover);

            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (response.ok) {
                const data = await response.json();
                const images = data.images;
                this.currentSatelliteType = satellite;
                if (satellite === 's1') { this.s1Images = images; } else { this.s2Images = images; }
                this.currentImages = images;

                this.showSearchResults(images, satellite);
                if (images.length === 0) {
                    this.showNotification(`No ${satelliteName} images found.`, 'info');
                } else {
                    this.showNotification(`Found ${images.length} ${satelliteName} images.`, 'success');
                }
                this.updateCompareSelects();
                this.showCompareSection();
            } else {
                throw new Error((await response.json()).detail || 'Search failed');
            }
        } catch (error) {
            console.error('Error searching images:', error);
            this.showNotification('Error occurred while searching images.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    showSearchResults(images, satellite = null) {
        // Determine which satellite's results to show
        const sat = satellite || this.selectedSatellite || 's2';
        const containerId = sat === 's1' ? 'inline-results-s1' : 'inline-results-s2';
        const countId = sat === 's1' ? 'results-count-s1' : 'results-count-s2';
        const resultsId = sat === 's1' ? 'search-results-s1' : 'search-results-s2';
        
        const inlineResults = document.getElementById(containerId);
        const resultsCount = document.getElementById(countId);
        const searchResults = document.getElementById(resultsId);

        // Update results count
        resultsCount.textContent = `${images.length} images found`;

        // Clear previous results
        searchResults.innerHTML = '';

        if (images.length === 0) {
            searchResults.innerHTML = '<p class="no-results">No images match your search criteria. Try adjusting the parameters.</p>';
        } else {
            // Create image items
            images.forEach(image => {
                const imageItem = this.createImageItem(image);
                searchResults.appendChild(imageItem);
            });
        }

        // Show inline results
        inlineResults.classList.remove('hidden');
    }

    createImageItem(image) {
        const item = document.createElement('div');
        item.className = 'image-item';
        item.dataset.imageId = image.id;

        const date = image.datetime ? new Date(image.datetime).toLocaleDateString() : 'Unknown';
        const aoiOverlap = image.aoi_overlap ? (image.aoi_overlap * 100).toFixed(1) : 'N/A';
        const isS1 = image.collection === 'Sentinel-1';
        
        // Sentinel-1: show orbit info, Sentinel-2: show cloud cover
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

        item.addEventListener('click', () => {
            this.selectImage(image.id);
        });

        // Add button event listeners
        const processBtn = item.querySelector('.process-btn');
        const downloadBtn = item.querySelector('.download-btn');
        const downloadOptions = item.querySelectorAll('.download-option');
        const downloadDropdown = item.querySelector('.download-dropdown');
        
        if (processBtn) {
            processBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.processImage(image.id);
            });
        }
        
        // Toggle download dropdown
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
                this.downloadSatelliteImage(image, type);
                // Hide dropdown
                const options = optBtn.closest('.download-options');
                if (options) options.classList.add('hidden');
            });
        });

        return item;
    }
    
    // Close dropdowns when clicking elsewhere
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
        this.showLoading(`Downloading ${satelliteName} ${typeLabel} image...`);
        
        try {
            const bounds = window.mapManager.getCurrentBounds();
            const geometry = window.mapManager.getCurrentGeoJSON()?.geometry;
            
            // Build request body
            const requestBody = {
                item_id: image.id,
                bbox: bounds,
                geometry: geometry,
                as_visualization: isVisualization
            };
            
            // Add visualization params if downloading with visualization
            if (isVisualization) {
                if (isS1) {
                    const visParams = this.getS1VisualizationParams();
                    requestBody.bands = visParams.bands;
                    requestBody.min = visParams.min;
                    requestBody.max = visParams.max;
                } else {
                    const visParams = this.getS2VisualizationParams();
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
                // Get filename from Content-Disposition header or generate one
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = `${image.id.split('/').pop()}_${typeLabel}.tif`;
                if (contentDisposition) {
                    const match = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
                    if (match && match[1]) {
                        filename = match[1].replace(/['"]/g, '');
                    }
                }
                
                // Download the blob
                const blob = await response.blob();
                
                // Create download link
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
                
                this.showNotification(`${satelliteName} ${typeLabel} downloaded: ${filename}`, 'success');
            } else {
                const errorData = await response.json().catch(() => ({ detail: 'Download failed' }));
                throw new Error(errorData.detail || 'Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showNotification(`Error downloading image: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async selectImage(imageId) {
        // Toggle logic: if clicking the same image, hide it
        if (this.selectedImageId === imageId) {
            this.selectedImageId = null;

            console.log(`User toggled off image: ${imageId}`);

            // Update UI - remove selection
            document.querySelectorAll('.image-item').forEach(item => {
                item.classList.remove('selected');
            });

            // Clear image layers
            if (window.mapManager) {
                window.mapManager.clearImageLayers();
            }

            this.showNotification('Image hidden from map', 'info');
            return;
        }
        this.selectedImageId = imageId;

        console.log(`User selected image: ${imageId}`);

        // Update UI - highlight selected image
        document.querySelectorAll('.image-item').forEach(item => {
            item.classList.remove('selected');
        });

        const selectedItem = document.querySelector(`[data-image-id="${imageId}"]`);
        if (selectedItem) {
            selectedItem.classList.add('selected');
        }

        // Clear existing image layers
        if (window.mapManager) {
            window.mapManager.clearImageLayers();
        }

        // Find selected image data (search in both S1 and S2 arrays)
        let image = this.currentImages.find(img => img.id === imageId);
        if (!image) {
            image = this.s1Images.find(img => img.id === imageId);
        }
        if (!image) {
            image = this.s2Images.find(img => img.id === imageId);
        }
        if (!image) {
            console.error(`Image ${imageId} not found in any image arrays`);
            return;
        }

        // Show loading
        this.showNotification('Preparing image for display...', 'info');

        try {
            // Determine API endpoint based on satellite type
            const isS1 = this.currentSatelliteType === 's1' || 
                         image.collection === 'Sentinel-1' ||
                         imageId.includes('S1_GRD');
            const apiEndpoint = isS1 ? '/api/get-s1-tile' : '/api/get-gee-tile';
            
            // Build request body
            const requestBody = {
                item_id: imageId,
                bbox: window.mapManager.getCurrentBounds(),
                geometry: window.mapManager.getCurrentGeoJSON()?.geometry
            };
            
            // Add S1 visualization params if applicable
            if (isS1) {
                const visParams = this.getS1VisualizationParams();
                requestBody.bands = visParams.bands;
                requestBody.min = visParams.min;
                requestBody.max = visParams.max;
            }
            
            // Generate GEE tile URLs for AOI (immediate map rendering)
            const response = await fetch(`${this.apiBaseUrl}${apiEndpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (response.ok) {
                const tileData = await response.json();

                // Use dynamic tile layer for base image
                image.display_url = tileData.static_url; // fallback only
                image.tile_template = tileData.tile_template;
                image.tile_bounds = tileData.bounds;
                image.display_type = 'tile';

                if (window.mapManager) {
                    window.mapManager.addImageLayer(imageId, image);
                    // Note: Removed fitToAOI to preserve user's current view
                    window.mapManager.outlineAOI();
                    this.showNotification('Image displayed on map.', 'success');
                }

            } else {
                console.warn(`Failed to generate AOI image URL for ${imageId}: ${response.status}`);
                // Fallback: use thumbnail if available
                if (image.assets && image.assets.thumbnail) {
                    image.display_url = image.assets.thumbnail;
                    image.display_type = 'image';
                    if (window.mapManager) {
                        window.mapManager.addImageLayer(imageId, image);
                        // Note: Removed fitToAOI to preserve user's current view
                    }
                    this.showNotification('Showing thumbnail image (AOI crop failed)', 'warning');
                } else {
                    this.showNotification('Unable to display image', 'error');
                }
            }

        } catch (error) {
            console.error(`Error generating AOI image URL for ${imageId}:`, error);
            this.showNotification('Error generating image URL', 'error');
        }

        console.log(`Image ${imageId} selection completed`);
    }

    async processImage(imageId) {
        if (!imageId) {
            this.showNotification('Please select an image first', 'error');
            return;
        }

        this.showLoading('Processing image on server (export + analysis)...');

        try {
            // 진행률 polling 시작
            const jobId = `${imageId}-${Date.now()}`;
            console.log(`[PROCESS] Starting process-image with job_id: ${jobId}`);
            this._progressStop = false;

            // 즉시 polling 시작
            console.log(`[PROCESS] Starting progress polling...`);
            this.pollProgress(jobId).catch((e) => {
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
                })
            });

            if (response.ok) {
                const data = await response.json();

                // Use single multi-band COG and proxy correctly
                const originalUri = data.original_cog_uri || data.original_10m_gcs_uri || data.original_20m_gcs_uri;
                if (!originalUri) throw new Error('No original COG URI returned');
                const makeProxy = (uri) => {
                    if (uri.startsWith('file://')) return `/api/proxy-file?path=${encodeURIComponent(uri)}`;
                    if (uri.startsWith('gs://')) return `/api/proxy-gcs?gcs_uri=${encodeURIComponent(uri)}`;
                    return uri;
                };
                this._cog = makeProxy(originalUri);

                // Store model-specific result URLs (e.g., binary mask for model1)
                this._modelMaskUrls = this._modelMaskUrls || {};
                if (data.model1_cog_uri) {
                    this._modelMaskUrls.model1 = makeProxy(data.model1_cog_uri);
                } else {
                    this._modelMaskUrls.model1 = null;
                }

                // Prefetch single COG and prepare model layers
                if (window.mapManager) {
                    try {
                        await window.mapManager.prefetchCOG(this._cog);
                        const prepTasks = [];
                        Object.keys(data.analysis_results || {}).forEach((modelId) => {
                            if (modelId === 'model1' && this._modelMaskUrls && this._modelMaskUrls.model1) {
                                prepTasks.push(window.mapManager.prepareBinaryMaskLayerFromCOG(modelId, this._modelMaskUrls.model1, { opacity: 1.0 }));
                            } else {
                                prepTasks.push(window.mapManager.prepareModelLayerFromCOG(modelId, this._cog, { opacity: 0.7 }));
                            }
                        });
                        await Promise.all(prepTasks);
                    } catch (e) {
                        console.warn('Prefetch/prepare failed:', e);
                    }
                }

                // Render final thumbnails immediately (server already analyzed)
                if (data.analysis_results) {
                    this.showAnalysisResults(data.analysis_results);
                }

                this.showNotification('Processing complete!', 'success');
                // Stop progress polling and hide loading on success
                this._progressStop = true;
                this.hideLoading();
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Processing failed');
            }

        } catch (error) {
            console.error('Error processing image:', error);
            this.showNotification('Error occurred during image processing', 'error');
            // Only stop progress polling on error
            this._progressStop = true;
            this.hideLoading();
        }
        // Note: progress polling will stop automatically when backend returns "done" status
    }

    async pollProgress(jobId) {
        console.log(`[POLLING] Starting progress polling for job: ${jobId}`);
        try {
            let pollCount = 0;
            while (!this._progressStop) {
                pollCount++;
                const url = `/api/progress?job_id=${encodeURIComponent(jobId)}`;
                console.log(`[POLLING] Poll #${pollCount} - Requesting: ${url}`);

                const r = await fetch(url);
                if (r.ok) {
                    const p = await r.json();
                    console.log(`[POLLING] Response:`, p);

                    // 백엔드에서 오는 실제 진행률로 즉시 업데이트
                    if (p.percent !== undefined && p.percent !== null) {
                        this.updateLoadingProgress(p.percent, p.message || '');
                    }

                    // 상태가 'completed' 또는 'done'이거나 진행률이 100% 이상이면 완료
                    if ((p.status && (p.status === 'done' || p.status === 'completed')) || (p.percent || 0) >= 100) {
                        console.log(`[POLLING] Progress completed for job: ${jobId}`);
                        this._progressStop = true;
                        setTimeout(() => this.hideLoading(), 500); // 잠깐 기다린 후 숨김
                        break;
                    }
                } else {
                    console.warn(`[POLLING] Failed for job: ${jobId}, Status: ${r.status}`);
                    const text = await r.text();
                    console.warn(`[POLLING] Response text:`, text);
                }

                // Stop after 300 polls to prevent infinite loop
                if (pollCount > 300) {
                    console.warn(`[POLLING] Stopping after ${pollCount} attempts`);
                    this._progressStop = true;
                    this.hideLoading();
                    break;
                }

                await new Promise(res => setTimeout(res, 300)); // 더 빠른 polling
            }
        } catch (e) {
            console.error('[POLLING] Error:', e);
            this._progressStop = true;
            this.hideLoading();
        }
    }

    showAnalysisResults(analysisResults) {
        // Store analysis results for later reference (e.g., threshold revert)
        this.lastAnalysisResults = analysisResults;

        // Switch to analysis tab
        this.switchToAnalysisTab();

        // Populate the analysis list
        const analysisList = document.querySelector('.analysis-list');
        analysisList.innerHTML = '';

        // Sort entries to put cloud_mask first
        const sortedEntries = Object.entries(analysisResults).sort(([keyA], [keyB]) => {
            if (keyA === 'overlay_meta') return 1;  // Put meta at end
            if (keyB === 'overlay_meta') return -1;
            if (keyA === 'cloud_mask') return -1;  // Cloud mask first
            if (keyB === 'cloud_mask') return 1;
            return 0;  // Keep other order
        });

        sortedEntries.forEach(([modelId, result]) => {
            if (modelId === 'overlay_meta') return; // skip meta entry

            const analysisItem = document.createElement('div');
            analysisItem.className = 'analysis-item';
            analysisItem.dataset.modelId = modelId;
            analysisItem.dataset.active = 'false';

            // Create base HTML structure
            let innerHTML = `
                <img class="analysis-thumbnail" src="${result.preview_url || result.thumbnail_url || ''}" alt="${result.name}" />
                <div class="analysis-info">
                    <h4 class="analysis-title">${result.name}</h4>
                    <p class="analysis-subtitle">${(result.bands || []).join(', ')}</p>
                    <div class="analysis-status inactive">Click to toggle overlay</div>
            `;

            // Add colorbar and controls for index-based results
            const isIndexResult = result.colormap || modelId.includes('model2') || modelId.includes('model3') || modelId.includes('model4');
            const isDeepLearningModel = modelId === 'model1'; // Segmentation model
            const isCloudMask = modelId === 'cloud_mask'; // Cloud mask - no special controls
            const isAlphaEarth = modelId === 'model5'; // AlphaEarth - only change monitoring

            if (isIndexResult && result.colormap) {
                innerHTML += this.createColorbarWithControls(result.colormap, modelId);
            } else if (result.colormap) {
                innerHTML += this.createColorbar(result.colormap);
            } else if (isDeepLearningModel) {
                // Add change monitoring button for deep learning models
                innerHTML += `
                    <div class="model-controls">
                        <button class="change-monitoring-btn" title="Click to start change monitoring">🌏</button>
                    </div>
                `;
            } else if (isAlphaEarth) {
                // AlphaEarth - only change monitoring button
                innerHTML += `
                    <div class="model-controls">
                        <button class="change-monitoring-btn" title="Click to start change monitoring">🌏</button>
                    </div>
                `;
            } else if (isCloudMask) {
                // Cloud mask - just simple toggle, no extra controls
                // innerHTML already contains the basic structure
            }

            innerHTML += `</div>`;
            analysisItem.innerHTML = innerHTML;

            // Store the original model ID and initialize current layer ID
            analysisItem.dataset.originalModelId = modelId;
            analysisItem.dataset.currentLayerId = modelId;
            analysisItem.dataset.currentOverlayUrl = result.overlay_url;
            analysisItem.dataset.originalOverlayUrl = result.overlay_url; // Store original for revert

            // Click to toggle overlay
            analysisItem.onclick = async () => {
                const isActive = analysisItem.dataset.active === 'true';
                const statusEl = analysisItem.querySelector('.analysis-status');
                const currentLayerId = analysisItem.dataset.currentLayerId;
                const currentOverlayUrl = analysisItem.dataset.currentOverlayUrl;

                if (!currentOverlayUrl) {
                    this.showNotification('Overlay not available yet', 'error');
                    return;
                }

                if (!isActive) {
                    // Show this overlay (no longer deactivating others for multi-layer support)
                    if (window.mapManager) {
                        // Special handling for model1 (segmentation) to use COG layer
                        if (currentLayerId === 'model1' && this._modelMaskUrls && this._modelMaskUrls.model1) {
                            await window.mapManager.showBinaryMaskCOG(currentLayerId, this._modelMaskUrls.model1);
                        } else {
                            await window.mapManager.showAnalysisLayer(currentLayerId, currentOverlayUrl, result.name);
                        }
                        // Apply AOI outline only (removed fitToAOI to preserve user's current view)
                        window.mapManager.outlineAOI();
                    }
                    analysisItem.dataset.active = 'true';
                    analysisItem.classList.add('active');
                    statusEl.textContent = 'Overlay active';
                    statusEl.classList.remove('inactive');
                    statusEl.classList.add('active');
                } else {
                    // Hide this specific overlay
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

            // Add control button event handlers for index results
            if (isIndexResult && result.colormap) {
                const pixelValueBtn = analysisItem.querySelector('.pixel-value-btn');
                if (pixelValueBtn) {
                    pixelValueBtn.onclick = (e) => {
                        e.stopPropagation(); // Prevent triggering the parent click
                        this.togglePixelValueInspection(modelId, result, pixelValueBtn);
                    };
                }

                const thresholdBtn = analysisItem.querySelector('.threshold-btn');
                if (thresholdBtn) {
                    thresholdBtn.onclick = async (e) => {
                        e.stopPropagation();
                        await this.toggleThresholdControl(modelId, result, thresholdBtn);
                    };
                }

                const changeMonitoringBtn = analysisItem.querySelector('.change-monitoring-btn');
                if (changeMonitoringBtn) {
                    changeMonitoringBtn.onclick = (e) => {
                        e.stopPropagation();
                        this.startChangeMonitoring(modelId, result, changeMonitoringBtn);
                    };
                }
            }

            // Add change monitoring handler for deep learning models
            if (isDeepLearningModel) {
                const changeMonitoringBtn = analysisItem.querySelector('.change-monitoring-btn');
                if (changeMonitoringBtn) {
                    changeMonitoringBtn.onclick = (e) => {
                        e.stopPropagation();
                        this.startChangeMonitoring(modelId, result, changeMonitoringBtn);
                    };
                }
            }

            // Add change monitoring handler for AlphaEarth
            if (isAlphaEarth) {
                const changeMonitoringBtn = analysisItem.querySelector('.change-monitoring-btn');
                if (changeMonitoringBtn) {
                    changeMonitoringBtn.onclick = (e) => {
                        e.stopPropagation();
                        this.startChangeMonitoring(modelId, result, changeMonitoringBtn);
                    };
                }
            }

            analysisList.appendChild(analysisItem);
        });

        // Add Custom Visualization option
        this.addCustomVisualizationOption(analysisList);

        // Add Target Detection option
        this.addTargetDetectionOption(analysisList);

        console.log('Analysis results displayed in list format');
    }

    /**
     * Add Target Detection option to analysis results
     */
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
            // Don't handle if clicking inside the UI section
            if (e.target.closest('.td-ui')) {
                return;
            }
            
            if (!this.targetDetection) {
                this.showNotification('Target Detection module not loaded', 'error');
                return;
            }

            // Delegate to controller
            this.targetDetection.handleItemClick();
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
            this.showCustomVisualizationModal();
        };

        analysisList.appendChild(customItem);
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

    createColorbarWithControls(colormap, modelId) {
        const minFormatted = colormap.min_val.toFixed(3);
        const maxFormatted = colormap.max_val.toFixed(3);

        return `
            <div class="colorbar-container">
                <div class="colorbar-header">
                    <div class="colorbar-label">${colormap.label}</div>
                    <div class="colorbar-controls">
                        <button class="pixel-value-btn" title="Click to enable pixel value inspection">🖱️</button>
                        <button class="threshold-btn" title="Click to set threshold">⚙️</button>
                        <button class="change-monitoring-btn" title="Click to start change monitoring">🌏</button>
                    </div>
                </div>
                <div class="colorbar">
                    <div class="colorbar-gradient ${colormap.name}"></div>
                    <div class="threshold-indicator" style="display: none;"></div>
                </div>
                <div class="colorbar-values">
                    <span class="colorbar-min">${minFormatted}</span>
                    <span class="colorbar-max">${maxFormatted}</span>
                </div>
                <div class="threshold-control-panel" style="display: none;">
                    <div class="threshold-slider-container">
                        <label>Threshold Range: <span class="threshold-range-value">${minFormatted} - ${maxFormatted}</span></label>
                        <div class="range-slider-container">
                            <div class="range-slider-track">
                                <div class="range-slider-range"></div>
                            </div>
                            <input type="range" class="range-slider range-min" 
                                   min="${colormap.min_val}" max="${colormap.max_val}" 
                                   step="0.01" value="${colormap.min_val}">
                            <input type="range" class="range-slider range-max" 
                                   min="${colormap.min_val}" max="${colormap.max_val}" 
                                   step="0.01" value="${colormap.max_val}">
                        </div>
                        <div class="range-values">
                            <span class="range-min-label">Min: <span class="threshold-min-value">${minFormatted}</span></span>
                            <span class="range-max-label">Max: <span class="threshold-max-value">${maxFormatted}</span></span>
                        </div>
                    </div>
                    <div class="threshold-buttons">
                        <button class="threshold-apply-btn">Apply Range</button>
                        <button class="threshold-cancel-btn">Cancel</button>
                    </div>
                </div>
            </div>
        `;
    }

    showCustomVisualizationModal() {
        // Delegate to CustomVisualizationController if available
        if (this.customViz) {
            return this.customViz.showModal();
        }
        // Legacy: Create modal if it doesn't exist
        if (!document.getElementById('custom-viz-modal')) {
            this.createCustomVisualizationModal();
        }
        const modal = document.getElementById('custom-viz-modal');
        modal.style.display = 'flex';
    }

    createCustomVisualizationModal() {
        const modalHTML = `
            <div id="custom-viz-modal" class="modal">
                <div class="modal-content custom-viz-modal-content">
                    <div class="modal-header">
                        <h3>🛠️ Custom Visualization</h3>
                        <button class="close-btn" onclick="this.closest('.modal').style.display='none'">&times;</button>
                    </div>
                    
                    <div class="custom-viz-name">
                        <label for="custom-viz-name-input">Visualization Name:</label>
                        <input type="text" id="custom-viz-name-input" placeholder="Enter custom name..." value="Custom Visualization">
                    </div>
                    
                    <div class="custom-viz-tabs">
                        <button class="custom-tab-btn active" data-tab="composite">Composite</button>
                        <button class="custom-tab-btn" data-tab="index">Index</button>
                        <button class="custom-tab-btn" data-tab="script">Custom Script</button>
                    </div>
                    
                    <!-- Composite Tab -->
                    <div id="composite-tab" class="custom-tab-content active">
                        <h4>RGB Composite</h4>
                        <p>Drag bands onto RGB fields.</p>
                        
                        <div class="band-selector">
                            <div class="available-bands">
                                <small style="color: #666; display: block; margin-bottom: 8px;">Available bands: B2, B3, B4, B8, B11, B12</small>
                                <div class="band-item" draggable="true" data-band="B2">B2</div>
                                <div class="band-item" draggable="true" data-band="B3">B3</div>
                                <div class="band-item" draggable="true" data-band="B4">B4</div>
                                <div class="band-item" draggable="true" data-band="B8">B8</div>
                                <div class="band-item" draggable="true" data-band="B11">B11</div>
                                <div class="band-item" draggable="true" data-band="B12">B12</div>
                            </div>
                            
                            <div class="rgb-fields">
                                <div class="rgb-field" data-channel="red">
                                    <label>R:</label>
                                    <div class="drop-zone" data-channel="red">Drop band here</div>
                                </div>
                                <div class="rgb-field" data-channel="green">
                                    <label>G:</label>
                                    <div class="drop-zone" data-channel="green">Drop band here</div>
                                </div>
                                <div class="rgb-field" data-channel="blue">
                                    <label>B:</label>
                                    <div class="drop-zone" data-channel="blue">Drop band here</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Index Tab -->
                    <div id="index-tab" class="custom-tab-content">
                        <h4>Create Index</h4>
                        <p>Drag bands into the index equation</p>
                        
                        <div class="index-builder">
                            <div class="available-bands">
                                <small style="color: #666; display: block; margin-bottom: 8px;">Available bands: B2, B3, B4, B8, B11, B12</small>
                                <div class="band-item" draggable="true" data-band="B2">B2</div>
                                <div class="band-item" draggable="true" data-band="B3">B3</div>
                                <div class="band-item" draggable="true" data-band="B4">B4</div>
                                <div class="band-item" draggable="true" data-band="B8">B8</div>
                                <div class="band-item" draggable="true" data-band="B11">B11</div>
                                <div class="band-item" draggable="true" data-band="B12">B12</div>
                            </div>
                            
                            <div class="index-formula">
                                <label>Index: (A-B)/(A+B)</label>
                                <div class="formula-builder">
                                    <span>(</span>
                                    <div class="formula-slot" data-slot="a">A</div>
                                    <span>-</span>
                                    <div class="formula-slot" data-slot="b">B</div>
                                    <span>) / (</span>
                                    <div class="formula-slot" data-slot="a2">A</div>
                                    <span>+</span>
                                    <div class="formula-slot" data-slot="b2">B</div>
                                    <span>)</span>
                                </div>
                                
                                <div class="colormap-control">
                                    <label>Colormap:</label>
                                    <select id="colormap-select">
                                        <option value="RdYlGn">RdYlGn (Red-Yellow-Green)</option>
                                        <option value="viridis">Viridis (Purple-Blue-Green-Yellow)</option>
                                        <option value="plasma">Plasma (Purple-Pink-Yellow)</option>
                                        <option value="coolwarm">Cool-Warm (Blue-Red)</option>
                                    </select>
                                    <div class="colormap-preview">
                                        <div class="colorbar">
                                            <div id="colormap-preview-gradient" class="colorbar-gradient RdYlGn"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Custom Script Tab -->
                    <div id="script-tab" class="custom-tab-content">
                        <h4>Custom Script</h4>
                        <p>Use custom script to create a custom visualization</p>
                        
                        <div class="script-editor">
                            <textarea id="custom-script" rows="15" placeholder="// Enter your Earth Engine JavaScript code here
function setup() {
  return {
    input: ['B01','B02','B03', 'dataMask'],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  return [2.5 * sample.B01, 2.5 * sample.B02, 2.5 * sample.B03, sample.dataMask];
}"></textarea>
                        </div>
                    </div>
                    
                    <div class="modal-footer">
                        <button class="cancel-btn" onclick="this.closest('.modal').style.display='none'">Cancel</button>
                        <button class="apply-btn" onclick="window.appInstance.applyCustomVisualization()">Apply</button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);
        this.setupCustomVisualizationControls();
    }

    setupCustomVisualizationControls() {
        // Tab switching
        const tabButtons = document.querySelectorAll('.custom-tab-btn');
        const tabContents = document.querySelectorAll('.custom-tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');

                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                button.classList.add('active');
                document.getElementById(`${targetTab}-tab`).classList.add('active');
            });
        });

        // Drag and Drop for bands
        this.setupDragAndDrop();

        // Colormap selection preview
        const colormapSelect = document.getElementById('colormap-select');
        const previewGradient = document.getElementById('colormap-preview-gradient');
        if (colormapSelect && previewGradient) {
            colormapSelect.addEventListener('change', () => {
                const selectedColormap = colormapSelect.value;
                // Remove all existing colormap classes
                previewGradient.className = 'colorbar-gradient';
                // Add the selected colormap class
                previewGradient.classList.add(selectedColormap);
            });
        }
    }

    setupDragAndDrop() {
        // Make band items draggable
        document.querySelectorAll('.band-item').forEach(item => {
            item.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', item.getAttribute('data-band'));
            });
        });

        // Make drop zones droppable
        document.querySelectorAll('.drop-zone, .formula-slot').forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('drag-over');
            });

            zone.addEventListener('dragleave', () => {
                zone.classList.remove('drag-over');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                const bandName = e.dataTransfer.getData('text/plain');
                zone.textContent = bandName;
                zone.setAttribute('data-band', bandName);
                zone.classList.remove('drag-over');
                zone.classList.add('filled');
            });
        });
    }

    async applyCustomVisualization() {
        const activeTab = document.querySelector('.custom-tab-btn.active').getAttribute('data-tab');

        let customVizData = {};

        if (activeTab === 'composite') {
            // Get RGB band assignments
            const redBand = document.querySelector('.drop-zone[data-channel="red"]').getAttribute('data-band');
            const greenBand = document.querySelector('.drop-zone[data-channel="green"]').getAttribute('data-band');
            const blueBand = document.querySelector('.drop-zone[data-channel="blue"]').getAttribute('data-band');

            if (!redBand || !greenBand || !blueBand) {
                this.showNotification('Please assign bands to all RGB channels', 'error');
                return;
            }

            customVizData = {
                type: 'composite',
                bands: [redBand, greenBand, blueBand]
            };
        } else if (activeTab === 'index') {
            // Get index formula bands
            const bandA = document.querySelector('.formula-slot[data-slot="a"]').getAttribute('data-band');
            const bandB = document.querySelector('.formula-slot[data-slot="b"]').getAttribute('data-band');
            const colormap = document.getElementById('colormap-select').value;

            if (!bandA || !bandB) {
                this.showNotification('Please assign bands to the formula slots', 'error');
                return;
            }

            customVizData = {
                type: 'index',
                bandA: bandA,
                bandB: bandB,
                colormap: colormap
            };
        } else if (activeTab === 'script') {
            // Get custom script
            const script = document.getElementById('custom-script').value;

            if (!script.trim()) {
                this.showNotification('Please enter a custom script', 'error');
                return;
            }

            customVizData = {
                type: 'script',
                script: script
            };
        }

        // Send to backend for processing
        try {
            this.showNotification('Applying custom visualization...', 'info');
            await this.processCustomVisualization(customVizData);
            document.getElementById('custom-viz-modal').style.display = 'none';
        } catch (error) {
            console.error('Custom visualization failed:', error);
            this.showNotification('Failed to apply custom visualization', 'error');
        }
    }

    getAOI() {
        // Get AOI bounds from map manager
        if (window.mapManager && window.mapManager.getCurrentBounds) {
            return window.mapManager.getCurrentBounds();
        }
        return null;
    }

    getSafeBounds() {
        try {
            const bounds = window.mapManager ? window.mapManager.getCurrentBounds() : null;
            if (!bounds) return null;

            // Convert Leaflet bounds to safe JSON serializable format
            if (bounds && typeof bounds === 'object') {
                // If it's a Leaflet bounds object, extract the coordinates
                if (bounds._southWest && bounds._northEast) {
                    return [
                        [Number(bounds._southWest.lat), Number(bounds._southWest.lng)],
                        [Number(bounds._northEast.lat), Number(bounds._northEast.lng)]
                    ];
                }
                // If it's already an array format, return as is but ensure it's safe
                if (Array.isArray(bounds) && bounds.length === 2) {
                    return [
                        [Number(bounds[0][0]), Number(bounds[0][1])],
                        [Number(bounds[1][0]), Number(bounds[1][1])]
                    ];
                }
            }
            return null;
        } catch (error) {
            console.error('Error getting safe bounds:', error);
            return null;
        }
    }

    async processCustomVisualization(customVizData) {
        const aoi = this.getAOI();
        if (!aoi) {
            this.showNotification('Please define an area of interest first', 'error');
            return;
        }

        // Get the current selected image ID
        if (!this.selectedImageId) {
            this.showNotification('Please select and process an image first', 'error');
            return;
        }

        const imageId = this.selectedImageId;

        // Get custom name from input
        const customName = document.getElementById('custom-viz-name-input')?.value || 'Custom Visualization';

        const payload = {
            image_id: imageId,
            bbox: aoi,
            custom_visualization: customVizData,
            custom_name: customName
        };

        const response = await fetch('/api/custom-visualization', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        // Add the custom visualization result to analysis results
        this.addCustomVisualizationResult(result);
    }

    addCustomVisualizationResult(result) {
        const analysisList = document.querySelector('.analysis-list');

        // Use the custom_id from backend if available, otherwise generate one
        const customId = result.custom_id || `custom-result-${Date.now()}`;

        const customResultItem = document.createElement('div');
        customResultItem.className = 'analysis-item';
        customResultItem.dataset.modelId = customId;
        customResultItem.dataset.active = 'false';

        // Create base HTML structure
        let innerHTML = `
            <img class="analysis-thumbnail" src="${result.preview_url}" alt="Custom Visualization" />
            <div class="analysis-info">
                <h4 class="analysis-title">🎨 ${result.name || 'Custom Visualization'}</h4>
                <p class="analysis-subtitle">${result.description || 'Custom visualization result'}</p>
                <div class="analysis-status inactive">Click to toggle overlay</div>
        `;

        // Add colorbar and controls for index visualizations if available
        if (result.colormap) {
            innerHTML += this.createColorbarWithControls(result.colormap, customId);
        }

        innerHTML += `</div>`;
        customResultItem.innerHTML = innerHTML;

        // Store the original custom ID and initialize current layer ID
        customResultItem.dataset.originalModelId = customId;
        customResultItem.dataset.currentLayerId = customId;
        customResultItem.dataset.currentOverlayUrl = result.overlay_url;
        customResultItem.dataset.originalOverlayUrl = result.overlay_url; // Store original for revert

        // Add click handler for overlay toggle
        customResultItem.onclick = async () => {
            const isActive = customResultItem.dataset.active === 'true';
            const statusEl = customResultItem.querySelector('.analysis-status');
            const currentLayerId = customResultItem.dataset.currentLayerId;
            const currentOverlayUrl = customResultItem.dataset.currentOverlayUrl;

            if (!isActive) {
                // Show this overlay (no longer deactivating others for multi-layer support)
                if (window.mapManager && currentOverlayUrl) {
                    await window.mapManager.showAnalysisLayer(currentLayerId, currentOverlayUrl, 'Custom Visualization');
                    // Apply AOI outline only (removed fitToAOI to preserve user's current view)
                    window.mapManager.outlineAOI();
                }
                customResultItem.dataset.active = 'true';
                customResultItem.classList.add('active');
                statusEl.textContent = 'Overlay active';
                statusEl.classList.remove('inactive');
                statusEl.classList.add('active');
            } else {
                // Hide this specific overlay
                if (window.mapManager) {
                    window.mapManager.hideAnalysisLayer(currentLayerId);
                }
                customResultItem.dataset.active = 'false';
                customResultItem.classList.remove('active');
                statusEl.textContent = 'Click to toggle overlay';
                statusEl.classList.remove('active');
                statusEl.classList.add('inactive');
            }
        };

        // Add control button event handlers for index results
        if (result.colormap) {
            // Pixel value inspection button
            const pixelValueBtn = customResultItem.querySelector('.pixel-value-btn');
            if (pixelValueBtn) {
                pixelValueBtn.onclick = (e) => {
                    e.stopPropagation();
                    this.togglePixelValueInspection(customId, result, pixelValueBtn);
                };
            }

            // Threshold control button
            const thresholdBtn = customResultItem.querySelector('.threshold-btn');
            if (thresholdBtn) {
                thresholdBtn.onclick = async (e) => {
                    e.stopPropagation();
                    await this.toggleThresholdControl(customId, result, thresholdBtn);
                };
            }

            // Change monitoring button
            const changeMonitoringBtn = customResultItem.querySelector('.change-monitoring-btn');
            if (changeMonitoringBtn) {
                changeMonitoringBtn.onclick = (e) => {
                    e.stopPropagation();
                    this.startChangeMonitoring(customId, result, changeMonitoringBtn);
                };
            }
        }

        // Add after the custom visualization option
        const customVizItem = analysisList.querySelector('.custom-visualization');
        if (customVizItem) {
            customVizItem.insertAdjacentElement('afterend', customResultItem);
        } else {
            analysisList.appendChild(customResultItem);
        }

        this.showNotification('Custom visualization applied successfully!', 'success');
    }

    setupTabControls() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');

                // Remove active class from all tabs and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab and corresponding content
                button.classList.add('active');
                document.getElementById(`${targetTab}-tab-content`).classList.add('active');

                // Update button states when switching to Change Monitoring tab
                if (targetTab === 'change-monitoring') {
                    const hasAOI = window.mapManager && window.mapManager.getCurrentBounds();
                    const searchBtn = document.getElementById('search-change-images-btn');
                    if (searchBtn) {
                        searchBtn.disabled = !hasAOI;
                    }
                }
            });
        });
    }



    showNoAnalysisResults() {
        const analysisList = document.querySelector('.analysis-list');
        if (analysisList) {
            analysisList.innerHTML = `
                <div class="no-analysis-results">
                    <div class="empty-state-icon">📊</div>
                    <h4 class="empty-state-title">No Analysis Results</h4>
                    <p class="empty-state-message">Select an image and click "Process Image" to see analysis results here.</p>
                </div>
            `;
        }
    }

    switchToAnalysisTab() {
        // Remove active from all tabs
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        // Activate analysis tab
        document.querySelector('[data-tab="analysis"]').classList.add('active');
        document.getElementById('analysis-tab-content').classList.add('active');
    }

    formatAnalysisData(analysisData) {
        if (!analysisData || analysisData.error) {
            return `<span style="color: #d93025;">Error: ${analysisData.error || 'Analysis failed'}</span>`;
        }

        let html = `<strong style="color: #34a853;">📊 ${analysisData.analysis_type}</strong><br>`;

        // Show resolution info prominently
        if (analysisData.resolution) {
            html += `<em style="color: #1976d2;">📏 ${analysisData.resolution}</em><br>`;
        }

        // Format different types of analysis results
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

    async searchChangeCandidates() {
        if (!window.mapManager || !window.mapManager.getCurrentBounds()) {
            this.showNotification('Please draw an AOI first', 'error');
            return;
        }

        this.showLoading('Searching for candidate images...');
        const candidateList = document.getElementById('candidate-list');
        candidateList.innerHTML = '';
        document.getElementById('candidate-results').classList.add('hidden');
        document.getElementById('run-change-btn').disabled = true;

        try {
            const startDate = document.getElementById('change-start-date').value;
            const endDate = document.getElementById('change-end-date').value;
            const bounds = window.mapManager.getCurrentBounds();

            const response = await fetch('/api/search-images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    bbox: bounds,
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    start_date: startDate,
                    end_date: endDate,
                    cloud_cover_max: 30, // Default for candidates
                    limit: 50 // Fetch more candidates
                })
            });

            if (response.ok) {
                const data = await response.json();
                const images = data.images;

                if (images.length === 0) {
                    this.showNotification('No candidate images found.', 'info');
                    candidateList.innerHTML = '<div class="no-results">No images found in this date range.</div>';
                } else {
                    this.renderCandidateList(images);
                    this.showNotification(`Found ${images.length} candidate images.`, 'success');
                }
                document.getElementById('candidate-results').classList.remove('hidden');
            } else {
                throw new Error('Search failed');
            }
        } catch (error) {
            console.error('Error searching candidates:', error);
            this.showNotification('Error searching for candidates.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    renderCandidateList(images) {
        const candidateList = document.getElementById('candidate-list');
        candidateList.innerHTML = '';

        images.forEach(image => {
            const item = document.createElement('div');
            item.className = 'candidate-item';

            const date = new Date(image.datetime).toLocaleDateString();
            const cloud = image.cloud_cover.toFixed(1);
            const overlap = (image.aoi_overlap * 100).toFixed(1);

            item.innerHTML = `
                <div class="candidate-checkbox">
                    <input type="checkbox" id="cb-${image.id}" value="${image.id}">
                </div>
                <div class="candidate-info">
                    <div class="candidate-date">${date}</div>
                    <div class="candidate-stats">
                        <span><i class="fas fa-cloud"></i> ${cloud}%</span>
                        <span><i class="fas fa-vector-square"></i> ${overlap}%</span>
                    </div>
                </div>
            `;

            // Row click toggles checkbox
            item.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox') {
                    const cb = item.querySelector('input[type="checkbox"]');
                    cb.checked = !cb.checked;
                    this.updateRunButtonState();
                }
            });

            item.querySelector('input[type="checkbox"]').addEventListener('change', () => {
                this.updateRunButtonState();
            });

            candidateList.appendChild(item);
        });
    }

    updateRunButtonState() {
        const checkedCount = document.querySelectorAll('#candidate-list input[type="checkbox"]:checked').length;
        const runBtn = document.getElementById('run-change-btn');
        runBtn.disabled = checkedCount < 2;
        runBtn.textContent = checkedCount < 2 ? 'Select at least 2 images' : `Run Analysis (${checkedCount} images)`;
    }

    async runChangeMonitoring() {
        const selectedCheckboxes = document.querySelectorAll('#candidate-list input[type="checkbox"]:checked');
        const imageIds = Array.from(selectedCheckboxes).map(cb => cb.value);

        if (imageIds.length < 2) {
            this.showNotification('Please select at least 2 images for change monitoring.', 'warning');
            return;
        }

        this.showLoading('Running change monitoring analysis...');

        try {
            const jobId = `change-${Date.now()}`;
            console.log(`[CHANGE] Starting change monitoring with job_id: ${jobId}`);
            this._progressStop = false;

            // Start polling
            this.pollProgress(jobId).catch(e => console.error('Polling failed:', e));

            const response = await fetch('/api/change-monitoring', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_ids: imageIds,
                    bbox: window.mapManager.getCurrentBounds(),
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    job_id: jobId
                })
            });

            if (response.ok) {
                const results = await response.json();
                this.displayChangeMonitoringResults(results);
                this.showNotification('Analysis complete!', 'success');
            } else {
                throw new Error('Analysis failed');
            }
        } catch (error) {
            console.error(error);
            this.showNotification('Error running analysis', 'error');
        } finally {
            this._progressStop = true;
            this.hideLoading();
        }
    }



    togglePixelValueInspection(modelId, result, btnElement) {
        // Delegate to ThresholdController if available
        if (this.thresholdController) {
            return this.thresholdController.togglePixelInspection(modelId, result, btnElement);
        }
        // Legacy implementation
        const analysisItem = btnElement.closest('.analysis-item');
        if (analysisItem.dataset.active !== 'true') {
            this.showNotification('Please activate the overlay first', 'warning');
            return;
        }
        const isActive = btnElement.classList.contains('active');
        if (isActive) {
            this.disablePixelValueInspection();
            btnElement.classList.remove('active');
            btnElement.textContent = '🖱️';
        } else {
            this.disablePixelValueInspection();
            btnElement.classList.add('active');
            btnElement.textContent = '🖱️✓';
            this.enablePixelValueInspection(modelId, result);
        }
    }

    enablePixelValueInspection(modelId, result) {
        if (!window.mapManager || !window.mapManager.map) return;

        this.currentInspectionModel = modelId;
        this.currentInspectionResult = result;

        // Change map cursor
        window.mapManager.map.getContainer().style.cursor = 'crosshair';

        // Add click event to map
        this.pixelInspectionHandler = (e) => {
            this.inspectPixelValue(e.latlng);
        };

        window.mapManager.map.on('click', this.pixelInspectionHandler);

        // Show instruction
        this.showNotification('Click on the map to inspect pixel values', 'info');
    }

    disablePixelValueInspection() {
        if (!window.mapManager || !window.mapManager.map) return;

        // Reset cursor
        window.mapManager.map.getContainer().style.cursor = '';

        // Remove click event
        if (this.pixelInspectionHandler) {
            window.mapManager.map.off('click', this.pixelInspectionHandler);
            this.pixelInspectionHandler = null;
        }

        // Remove active state from all pixel value buttons
        document.querySelectorAll('.pixel-value-btn.active').forEach(btn => {
            btn.classList.remove('active');
            btn.title = 'Click to enable pixel value inspection';
            btn.textContent = '🖱️';
        });

        // Hide any existing pixel value popup
        this.hidePixelValuePopup();

        this.currentInspectionModel = null;
        this.currentInspectionResult = null;
    }

    async inspectPixelValue(latlng) {
        if (!this.currentInspectionModel || !this.currentInspectionResult) return;

        try {
            // Show loading
            this.showPixelValuePopup(latlng, 'Loading...', null);

            // Get pixel value from backend
            const response = await fetch('/api/get-pixel-value', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: this.selectedImageId,
                    lat: latlng.lat,
                    lng: latlng.lng,
                    model_id: this.currentInspectionModel
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.showPixelValuePopup(latlng, data.value, this.currentInspectionResult.colormap);
            } else {
                this.showPixelValuePopup(latlng, 'Error loading value', null);
            }
        } catch (error) {
            console.error('Error inspecting pixel value:', error);
            this.showPixelValuePopup(latlng, 'Error', null);
        }
    }

    showPixelValuePopup(latlng, value, colormap) {
        // Remove existing popup
        this.hidePixelValuePopup();

        // Create popup
        this.pixelValuePopup = L.popup({
            closeButton: true,
            autoClose: false,
            closeOnClick: false,
            className: 'pixel-value-popup'
        })
            .setLatLng(latlng)
            .setContent(`
            <div class="pixel-value-content">
                <strong>Pixel Value:</strong><br>
                <span class="pixel-value">${value}</span>
                ${colormap ? `<br><small>${colormap.label || 'Index'}</small>` : ''}
            </div>
        `)
            .openOn(window.mapManager.map);
    }

    hidePixelValuePopup() {
        if (this.pixelValuePopup) {
            window.mapManager.map.closePopup(this.pixelValuePopup);
            this.pixelValuePopup = null;
        }
    }

    async toggleThresholdControl(modelId, result, btnElement) {
        // Check if the overlay is active
        const analysisItem = btnElement.closest('.analysis-item');
        const isOverlayActive = analysisItem.dataset.active === 'true';

        if (!isOverlayActive) {
            this.showNotification('Please activate the overlay first before setting threshold', 'warning');
            return;
        }

        // Toggle threshold control mode
        const isActive = btnElement.classList.contains('active');
        const thresholdPanel = analysisItem.querySelector('.threshold-control-panel');

        if (isActive) {
            // Disable threshold control
            await this.disableThresholdControl(btnElement);
        } else {
            // Enable threshold control (disable others first)
            await this.disableAllThresholdControls();
            btnElement.classList.add('active');
            btnElement.title = 'Click to disable threshold control';
            btnElement.textContent = '⚙️✓';

            if (thresholdPanel) {
                thresholdPanel.style.display = 'block';
                this.setupThresholdControls(modelId, result, analysisItem);
            }
        }
    }

    async disableThresholdControl(btnElement) {
        btnElement.classList.remove('active');
        btnElement.title = 'Click to set threshold';
        btnElement.textContent = '⚙️';

        const analysisItem = btnElement.closest('.analysis-item');
        const thresholdPanel = analysisItem.querySelector('.threshold-control-panel');
        const thresholdIndicator = analysisItem.querySelector('.threshold-indicator');

        if (thresholdPanel) thresholdPanel.style.display = 'none';
        if (thresholdIndicator) thresholdIndicator.style.display = 'none';

        // Revert to original layer if threshold was applied
        const originalModelId = analysisItem.dataset.originalModelId;
        const currentLayerId = analysisItem.dataset.currentLayerId;

        if (currentLayerId && currentLayerId.includes('-threshold')) {
            // Get original overlay URL - we need to find it from the original result
            const modelId = originalModelId;

            // Find original overlay URL from analysis results or custom viz cache
            let originalOverlayUrl = null;
            let originalName = '';

            // Get original overlay URL from stored dataset
            originalOverlayUrl = analysisItem.dataset.originalOverlayUrl;

            // Check if it's a custom visualization
            if (modelId.startsWith('custom-result-')) {
                originalName = 'Custom Visualization';
            } else {
                // For regular analysis results, get name from analysis results object
                if (this.lastAnalysisResults && this.lastAnalysisResults[modelId]) {
                    originalName = this.lastAnalysisResults[modelId].name || 'Analysis Result';

                    // If originalOverlayUrl is not stored, get it from lastAnalysisResults
                    if (!originalOverlayUrl && this.lastAnalysisResults[modelId].overlay_url) {
                        originalOverlayUrl = this.lastAnalysisResults[modelId].overlay_url;
                        // Store it for future reference
                        analysisItem.dataset.originalOverlayUrl = originalOverlayUrl;
                        console.log(`Recovered original overlay URL for ${modelId}: ${originalOverlayUrl}`);
                    }
                } else {
                    originalName = 'Analysis Result';
                }
            }

            if (originalOverlayUrl && window.mapManager) {
                console.log(`Reverting to original overlay: ${originalModelId} -> ${originalOverlayUrl}`);

                // Hide threshold layer
                window.mapManager.hideAnalysisLayer(currentLayerId);

                // Show original layer
                await window.mapManager.showAnalysisLayer(originalModelId, originalOverlayUrl, originalName);

                // Update analysis item state
                analysisItem.dataset.currentLayerId = originalModelId;
                analysisItem.dataset.currentOverlayUrl = originalOverlayUrl;

                const statusEl = analysisItem.querySelector('.analysis-status');
                if (statusEl) {
                    statusEl.textContent = 'Overlay active';
                    statusEl.classList.remove('inactive');
                    statusEl.classList.add('active');
                }
            } else {
                console.warn(`Cannot revert threshold overlay: originalOverlayUrl=${originalOverlayUrl}, modelId=${modelId}`);
                // Fallback: just hide the threshold layer and deactivate
                if (window.mapManager) {
                    window.mapManager.hideAnalysisLayer(currentLayerId);
                }

                const statusEl = analysisItem.querySelector('.analysis-status');
                if (statusEl) {
                    statusEl.textContent = 'Overlay inactive';
                    statusEl.classList.remove('active');
                    statusEl.classList.add('inactive');
                }

                analysisItem.dataset.active = 'false';
                analysisItem.classList.remove('active');
            }
        }
    }

    async disableAllThresholdControls() {
        const promises = Array.from(document.querySelectorAll('.threshold-btn.active')).map(btn =>
            this.disableThresholdControl(btn)
        );
        await Promise.all(promises);
    }

    setupThresholdControls(modelId, result, analysisItem) {
        const minSlider = analysisItem.querySelector('.range-min');
        const maxSlider = analysisItem.querySelector('.range-max');
        const minValueSpan = analysisItem.querySelector('.threshold-min-value');
        const maxValueSpan = analysisItem.querySelector('.threshold-max-value');
        const rangeValueSpan = analysisItem.querySelector('.threshold-range-value');
        const rangeTrack = analysisItem.querySelector('.range-slider-range');
        const applyBtn = analysisItem.querySelector('.threshold-apply-btn');
        const cancelBtn = analysisItem.querySelector('.threshold-cancel-btn');

        if (!minSlider || !maxSlider || !minValueSpan || !maxValueSpan || !applyBtn || !cancelBtn) return;

        // Helper function to update range display
        const updateRangeDisplay = () => {
            const minValue = parseFloat(minSlider.value);
            const maxValue = parseFloat(maxSlider.value);

            minValueSpan.textContent = minValue.toFixed(3);
            maxValueSpan.textContent = maxValue.toFixed(3);
            rangeValueSpan.textContent = `${minValue.toFixed(3)} - ${maxValue.toFixed(3)}`;

            // Update visual range indicator
            const colormap = result.colormap;
            const minPercent = ((minValue - colormap.min_val) / (colormap.max_val - colormap.min_val)) * 100;
            const maxPercent = ((maxValue - colormap.min_val) / (colormap.max_val - colormap.min_val)) * 100;

            if (rangeTrack) {
                rangeTrack.style.left = `${minPercent}%`;
                rangeTrack.style.width = `${maxPercent - minPercent}%`;
            }
        };

        // Update min threshold value display when slider changes
        minSlider.oninput = (e) => {
            e.stopPropagation();
            const minValue = parseFloat(minSlider.value);
            const maxValue = parseFloat(maxSlider.value);

            // Ensure min doesn't exceed max
            if (minValue >= maxValue) {
                minSlider.value = maxValue - 0.01;
            }
            updateRangeDisplay();
        };

        // Update max threshold value display when slider changes
        maxSlider.oninput = (e) => {
            e.stopPropagation();
            const minValue = parseFloat(minSlider.value);
            const maxValue = parseFloat(maxSlider.value);

            // Ensure max doesn't go below min
            if (maxValue <= minValue) {
                maxSlider.value = minValue + 0.01;
            }
            updateRangeDisplay();
        };

        // Prevent slider clicks from triggering parent
        minSlider.onclick = (e) => e.stopPropagation();
        maxSlider.onclick = (e) => e.stopPropagation();

        // Apply threshold range
        applyBtn.onclick = async (e) => {
            e.stopPropagation();
            const minThreshold = parseFloat(minSlider.value);
            const maxThreshold = parseFloat(maxSlider.value);
            await this.applyThresholdRange(modelId, result, minThreshold, maxThreshold);
        };

        // Cancel threshold
        cancelBtn.onclick = async (e) => {
            e.stopPropagation();
            const thresholdBtn = analysisItem.querySelector('.threshold-btn');
            await this.disableThresholdControl(thresholdBtn);
        };

        // Prevent threshold panel clicks from triggering parent
        const thresholdPanel = analysisItem.querySelector('.threshold-control-panel');
        if (thresholdPanel) {
            thresholdPanel.onclick = (e) => e.stopPropagation();
        }

        // Initialize threshold range display
        updateRangeDisplay();
    }

    updateThresholdIndicator(analysisItem, thresholdValue, colormap) {
        const indicator = analysisItem.querySelector('.threshold-indicator');
        if (!indicator) return;

        const minVal = colormap.min_val;
        const maxVal = colormap.max_val;
        const percentage = ((thresholdValue - minVal) / (maxVal - minVal)) * 100;

        indicator.style.display = 'block';
        indicator.style.left = `${percentage}%`;
    }

    updateThresholdRangeIndicator(analysisItem, minThreshold, maxThreshold, colormap) {
        const indicator = analysisItem.querySelector('.threshold-indicator');
        if (!indicator) return;

        const minVal = colormap.min_val;
        const maxVal = colormap.max_val;
        const minPercentage = ((minThreshold - minVal) / (maxVal - minVal)) * 100;
        const maxPercentage = ((maxThreshold - minVal) / (maxVal - minVal)) * 100;

        indicator.style.display = 'block';
        indicator.style.left = `${minPercentage}%`;
        indicator.style.width = `${maxPercentage - minPercentage}%`;
        indicator.style.background = 'rgba(255, 0, 0, 0.3)';
        indicator.style.border = '1px solid #ff0000';
    }

    async applyThreshold(modelId, result, thresholdValue) {
        try {
            this.showNotification('Applying threshold...', 'info');

            const aoi = this.getAOI();
            if (!aoi) {
                this.showNotification('Please define an area of interest first', 'error');
                return;
            }

            const response = await fetch('/api/apply-threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: this.selectedImageId,
                    model_id: modelId,
                    threshold: thresholdValue,
                    colormap: result.colormap
                })
            });

            if (response.ok) {
                const thresholdResult = await response.json();

                // Update the overlay with thresholded version
                if (window.mapManager && thresholdResult.overlay_url) {
                    const thresholdLayerId = `${modelId}-threshold`;

                    // Find the corresponding analysis item (works for both regular models and custom visualizations)
                    const analysisItem = document.querySelector(`[data-model-id="${modelId}"]`);
                    if (analysisItem) {
                        // Hide the current overlay first
                        const currentLayerId = analysisItem.dataset.currentLayerId;
                        window.mapManager.hideAnalysisLayer(currentLayerId);

                        // Update the analysis item's current layer info
                        analysisItem.dataset.currentLayerId = thresholdLayerId;
                        analysisItem.dataset.currentOverlayUrl = thresholdResult.overlay_url;

                        // Show threshold overlay
                        const displayName = result.name || 'Index';
                        await window.mapManager.showAnalysisLayer(
                            thresholdLayerId,
                            thresholdResult.overlay_url,
                            `${displayName} (Threshold: ${thresholdValue.toFixed(3)})`
                        );

                        // Update analysis item state
                        analysisItem.dataset.active = 'true';
                        analysisItem.classList.add('active');
                        const statusEl = analysisItem.querySelector('.analysis-status');
                        if (statusEl) {
                            statusEl.textContent = 'Overlay active (Threshold applied)';
                            statusEl.classList.remove('inactive');
                            statusEl.classList.add('active');
                        }
                    }
                }

                this.showNotification(`Threshold applied: ${thresholdValue.toFixed(3)}`, 'success');
            } else {
                this.showNotification('Failed to apply threshold', 'error');
            }
        } catch (error) {
            console.error('Error applying threshold:', error);
            this.showNotification('Error applying threshold', 'error');
        }
    }

    async applyThresholdRange(modelId, result, minThreshold, maxThreshold) {
        try {
            this.showNotification(`Applying threshold range: ${minThreshold.toFixed(3)} - ${maxThreshold.toFixed(3)}...`, 'info');

            const aoi = this.getAOI();
            if (!aoi) {
                this.showNotification('Please define an area of interest first', 'error');
                return;
            }

            const response = await fetch('/api/apply-threshold-range', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: this.selectedImageId,
                    model_id: modelId,
                    min_threshold: minThreshold,
                    max_threshold: maxThreshold,
                    colormap: result.colormap
                })
            });

            if (response.ok) {
                const thresholdResult = await response.json();

                // Update the overlay with thresholded version
                if (window.mapManager && thresholdResult.overlay_url) {
                    const thresholdLayerId = `${modelId}-threshold`;

                    // Find the corresponding analysis item (works for both regular models and custom visualizations)
                    const analysisItem = document.querySelector(`[data-model-id="${modelId}"]`);
                    if (analysisItem) {
                        // Hide the current overlay first
                        const currentLayerId = analysisItem.dataset.currentLayerId;
                        window.mapManager.hideAnalysisLayer(currentLayerId);

                        // Store original overlay URL if this is the first threshold application
                        if (!currentLayerId.includes('-threshold')) {
                            analysisItem.dataset.originalOverlayUrl = analysisItem.dataset.currentOverlayUrl;
                            analysisItem.dataset.originalModelId = modelId;
                            console.log(`Stored original overlay URL for ${modelId}: ${analysisItem.dataset.currentOverlayUrl}`);
                        } else {
                            console.log(`Threshold already applied for ${modelId}, not overwriting original URL`);
                        }

                        // Ensure we have a fallback from lastAnalysisResults if needed
                        if (!analysisItem.dataset.originalOverlayUrl && this.lastAnalysisResults && this.lastAnalysisResults[modelId]) {
                            analysisItem.dataset.originalOverlayUrl = this.lastAnalysisResults[modelId].overlay_url;
                            analysisItem.dataset.originalModelId = modelId;
                            console.log(`Fallback: set original overlay URL for ${modelId} from lastAnalysisResults: ${this.lastAnalysisResults[modelId].overlay_url}`);
                        }

                        // Update the analysis item's current layer info
                        analysisItem.dataset.currentLayerId = thresholdLayerId;
                        analysisItem.dataset.currentOverlayUrl = thresholdResult.overlay_url;

                        // Show threshold overlay
                        const displayName = result.name || 'Index';
                        await window.mapManager.showAnalysisLayer(
                            thresholdLayerId,
                            thresholdResult.overlay_url,
                            `${displayName} (Range: ${minThreshold.toFixed(3)}-${maxThreshold.toFixed(3)})`
                        );

                        // Update analysis item state
                        analysisItem.dataset.active = 'true';
                        analysisItem.classList.add('active');
                        const statusEl = analysisItem.querySelector('.analysis-status');
                        if (statusEl) {
                            statusEl.textContent = 'Overlay active (Range applied)';
                            statusEl.classList.remove('inactive');
                            statusEl.classList.add('active');
                        }
                    }
                }

                this.showNotification(`Range applied: ${minThreshold.toFixed(3)} - ${maxThreshold.toFixed(3)}`, 'success');
            } else {
                this.showNotification('Failed to apply threshold range', 'error');
            }
        } catch (error) {
            console.error('Error applying threshold range:', error);
            this.showNotification('Error applying threshold range', 'error');
        }
    }

    startChangeMonitoring(modelId, result, btnElement) {
        console.log('[startChangeMonitoring] Called with:', { modelId, result, btnElement });

        // Check if the overlay is active only if launched from a specific item button
        if (btnElement) {
            const analysisItem = btnElement.closest('.analysis-item');
            if (analysisItem) {
                const isOverlayActive = analysisItem.dataset.active === 'true';
                if (!isOverlayActive) {
                    this.showNotification('Please activate the overlay first before starting change monitoring', 'warning');
                    return;
                }
            }
        }

        // Store the selected model for change monitoring
        this.changeMonitoringModel = {
            modelId: modelId,
            result: result
        };

        // Switch to Change Monitoring tab
        this.switchToChangeMonitoringTab();

        this.showNotification(`Change monitoring started for ${result.name || modelId}`, 'success');
    }




    switchToChangeMonitoringTab() {
        // Show and activate the change monitoring tab
        const changeMonitoringTab = document.getElementById('change-monitoring-tab');
        if (changeMonitoringTab) {
            // Remove hidden class to show the tab
            changeMonitoringTab.classList.remove('hidden');
            // Click to activate it
            changeMonitoringTab.click();
        }
    }

    // ===== New Change Monitoring Functions =====

    async searchChangeImages() {
        if (!window.mapManager || !window.mapManager.getCurrentBounds()) {
            this.showNotification('Please draw an AOI first', 'error');
            return;
        }

        this.showLoading('Searching for images...');

        try {
            const startDate = document.getElementById('change-start-date').value;
            const endDate = document.getElementById('change-end-date').value;
            const cloudCover = document.getElementById('change-cloud-cover').value;
            const minAoiCoverage = parseInt(document.getElementById('change-aoi-coverage')?.value || '0');
            const bounds = window.mapManager.getCurrentBounds();

            console.log('🔍 Change monitoring search params:', {
                startDate,
                endDate,
                cloudCover,
                minAoiCoverage,
                bounds,
                geometry: window.mapManager.getCurrentGeoJSON()?.geometry ? 'present' : 'null'
            });

            // Validate dates
            if (!startDate || !endDate) {
                throw new Error('Please select start and end dates');
            }

            const requestBody = {
                bbox: bounds,
                geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                start_date: startDate,
                end_date: endDate,
                cloud_cover_max: parseInt(cloudCover) || 30,
                limit: 100  // Reduced from 200 to avoid Earth Engine timeout
            };
            
            console.log('🔍 Request body:', requestBody);

            const response = await fetch('/api/search-images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (response.ok) {
                const data = await response.json();
                
                console.log('📊 Change monitoring search results:', data.images.length, 'images');
                if (data.images.length > 0) {
                    console.log('📊 Sample image data:', data.images[0]);
                }
                
                // Filter by AOI coverage
                // API returns aoi_overlap as fraction (0-1), convert to percentage for comparison
                const filteredImages = data.images.filter(img => {
                    // Try aoi_overlap (fraction) first, then aoi_coverage (percentage)
                    // IMPORTANT: Check for null AND undefined since null * 100 = 0
                    let aoiCoverage = 100;  // default
                    if (img.aoi_overlap != null && img.aoi_overlap !== undefined) {
                        aoiCoverage = img.aoi_overlap * 100;
                    } else if (img.properties?.aoi_overlap != null) {
                        aoiCoverage = img.properties.aoi_overlap * 100;
                    } else if (img.aoi_coverage != null) {
                        aoiCoverage = img.aoi_coverage;
                    } else if (img.properties?.aoi_coverage != null) {
                        aoiCoverage = img.properties.aoi_coverage;
                    }
                    // else aoiCoverage stays 100 (default - assume full coverage if not specified)
                    
                    return aoiCoverage >= minAoiCoverage;
                });
                
                console.log(`📊 Filtered: ${filteredImages.length}/${data.images.length} images with AOI coverage >= ${minAoiCoverage}%`);
                
                this.changeMonitoringImages = filteredImages;
                this.renderChangeCalendar(filteredImages);
                this.showNotification(`Found ${filteredImages.length} images (filtered from ${data.images.length})`, 'success');
            } else {
                // Get actual error message from response
                const errorData = await response.json().catch(() => ({}));
                console.error('Search API error:', response.status, errorData);
                throw new Error(errorData.detail || `Search failed (${response.status})`);
            }
        } catch (error) {
            console.error('Error searching images:', error);
            this.showNotification('Error searching images', 'error');
        } finally {
            this.hideLoading();
        }
    }

    renderChangeCalendar(images) {
        const calendarGrid = document.getElementById('change-calendar-grid');
        const resultsDiv = document.getElementById('change-calendar-results');

        resultsDiv.classList.remove('hidden');
        calendarGrid.innerHTML = '';

        // Initialize change monitoring layers map if not exists
        if (!this.changeMonitoringLayers) {
            this.changeMonitoringLayers = new Map();
        }
        
        // Clear image data cache for new search (avoid stale data)
        if (this.changeImageDataCache) {
            this.changeImageDataCache.clear();
            console.log('Cleared change monitoring image cache for new search');
        }
        
        // Reset analysis completed flag for new search
        this.changeAnalysisCompleted = false;

        if (images.length === 0) {
            calendarGrid.innerHTML = '<p class="no-results">No images found for the selected period.</p>';
            return;
        }

        // Group images by date
        const imagesByDate = {};
        images.forEach(img => {
            const date = img.datetime ? img.datetime.split('T')[0] : null;
            if (date) {
                if (!imagesByDate[date]) {
                    imagesByDate[date] = [];
                }
                imagesByDate[date].push(img);
            }
        });

        // Store for later use
        this.imagesByDate = imagesByDate;

        // Get date range from images
        const dates = Object.keys(imagesByDate).sort();
        if (dates.length === 0) return;

        const firstDate = new Date(dates[0]);
        const lastDate = new Date(dates[dates.length - 1]);

        // Initialize current calendar month to first available month
        this.calendarCurrentYear = firstDate.getFullYear();
        this.calendarCurrentMonth = firstDate.getMonth();
        this.calendarMinDate = firstDate;
        this.calendarMaxDate = lastDate;

        // Render the calendar view
        this.renderCalendarView(calendarGrid, imagesByDate);

        this.selectedChangeDates = new Set();
        this.updateChangeSelectedCount();
    }

    renderCalendarView(container, imagesByDate) {
        const year = this.calendarCurrentYear;
        const month = this.calendarCurrentMonth;
        const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December'];
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

        // Check if there's a next/prev month with data
        const hasPrevMonth = this.calendarMinDate && 
            (year > this.calendarMinDate.getFullYear() || 
             (year === this.calendarMinDate.getFullYear() && month > this.calendarMinDate.getMonth()));
        const hasNextMonth = this.calendarMaxDate && 
            (year < this.calendarMaxDate.getFullYear() || 
             (year === this.calendarMaxDate.getFullYear() && month < this.calendarMaxDate.getMonth()));

        container.innerHTML = `
            <div class="calendar-widget">
                <!-- Calendar Header with Month Navigation -->
                <div class="calendar-header">
                    <button class="calendar-nav-btn ${!hasPrevMonth ? 'disabled' : ''}" id="prev-month-btn" ${!hasPrevMonth ? 'disabled' : ''}>
                        ◀
                    </button>
                    <div class="calendar-month-year">
                        <span class="calendar-icon">📅</span>
                        <span class="month-name">${monthNames[month]} ${year}</span>
                    </div>
                    <button class="calendar-nav-btn ${!hasNextMonth ? 'disabled' : ''}" id="next-month-btn" ${!hasNextMonth ? 'disabled' : ''}>
                        ▶
                    </button>
                </div>

                <!-- Legend -->
                <div class="calendar-legend">
                    <div class="legend-item"><span class="legend-dot available"></span> Available</div>
                    <div class="legend-item"><span class="legend-dot selected"></span> Selected</div>
                    <div class="legend-item"><span class="legend-dot low-cloud"></span> Low Cloud (&lt;20%)</div>
                </div>

                <!-- Day Names Header -->
                <div class="calendar-day-names">
                    ${dayNames.map(day => `<div class="day-name">${day}</div>`).join('')}
                </div>

                <!-- Calendar Days Grid -->
                <div class="calendar-days">
                    ${this.generateCalendarDays(year, month, imagesByDate)}
                </div>
            </div>
        `;

        // Add event listeners
        const prevBtn = container.querySelector('#prev-month-btn');
        const nextBtn = container.querySelector('#next-month-btn');

        if (prevBtn && hasPrevMonth) {
            prevBtn.addEventListener('click', () => {
                this.calendarCurrentMonth--;
                if (this.calendarCurrentMonth < 0) {
                    this.calendarCurrentMonth = 11;
                    this.calendarCurrentYear--;
                }
                this.renderCalendarView(container, imagesByDate);
            });
        }

        if (nextBtn && hasNextMonth) {
            nextBtn.addEventListener('click', () => {
                this.calendarCurrentMonth++;
                if (this.calendarCurrentMonth > 11) {
                    this.calendarCurrentMonth = 0;
                    this.calendarCurrentYear++;
                }
                this.renderCalendarView(container, imagesByDate);
            });
        }

        // Add click handlers for calendar days with data
        container.querySelectorAll('.calendar-day.has-data').forEach(dayEl => {
            // Click on day to toggle selection
            dayEl.addEventListener('click', (e) => {
                // Don't toggle if clicking on buttons
                if (e.target.closest('.day-btn')) return;
                this.handleCalendarDayClick(dayEl);
            });

            // View button handler
            const viewBtn = dayEl.querySelector('.view-btn');
            if (viewBtn) {
                viewBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const date = dayEl.dataset.date;
                    const imageId = dayEl.dataset.imageId;
                    const image = this.findImageById(imageId);
                    if (image) {
                        await this.toggleChangeRawTile(date, image, viewBtn, dayEl);
                    }
                });
            }

            // Analyze button handler
            const analyzeBtn = dayEl.querySelector('.analyze-btn');
            if (analyzeBtn) {
                analyzeBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const date = dayEl.dataset.date;
                    const imageId = dayEl.dataset.imageId;
                    const image = this.findImageById(imageId);
                    if (image) {
                        await this.toggleChangeSpectralTile(date, image, analyzeBtn, dayEl);
                    }
                });
            }
        });
    }

    generateCalendarDays(year, month, imagesByDate) {
        const firstDay = new Date(year, month, 1);
        const lastDay = new Date(year, month + 1, 0);
        const startPadding = firstDay.getDay(); // 0 = Sunday
        const daysInMonth = lastDay.getDate();

        let html = '';

        // Empty cells for padding before first day
        for (let i = 0; i < startPadding; i++) {
            html += '<div class="calendar-day empty"></div>';
        }

        // Days of the month
        for (let day = 1; day <= daysInMonth; day++) {
            const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
            const hasData = imagesByDate[dateStr];
            const isSelected = this.selectedChangeDates && this.selectedChangeDates.has(dateStr);

            if (hasData) {
                const imgs = imagesByDate[dateStr];
                const bestImage = imgs.reduce((best, img) =>
                    img.cloud_cover < best.cloud_cover ? img : best
                );
                const cloudCover = bestImage.cloud_cover ? bestImage.cloud_cover.toFixed(0) : '?';
                const aoiCoverage = bestImage.aoi_overlap ? (bestImage.aoi_overlap * 100).toFixed(0) : '?';
                const isLowCloud = bestImage.cloud_cover && bestImage.cloud_cover < 20;

                // Check if analysis has been run (analyze button only shown after analysis)
                const showAnalyzeBtn = this.changeAnalysisCompleted === true;
                
                html += `
                    <div class="calendar-day has-data ${isSelected ? 'selected' : ''} ${isLowCloud ? 'low-cloud' : ''}" 
                         data-date="${dateStr}" 
                         data-image-id="${bestImage.id}"
                         title="Cloud: ${cloudCover}% | AOI: ${aoiCoverage}%">
                        <div class="day-number">${day}</div>
                        <div class="day-stats">
                            <span class="stat">☁${cloudCover}</span>
                            <span class="stat">◎${aoiCoverage}</span>
                        </div>
                        <div class="day-actions">
                            <button class="day-btn view-btn" title="View Satellite Tile" data-date="${dateStr}">🛰️</button>
                        </div>
                    </div>
                `;
            } else {
                html += `<div class="calendar-day no-data"><div class="day-number">${day}</div></div>`;
            }
        }

        return html;
    }

    handleCalendarDayClick(dayEl) {
        const date = dayEl.dataset.date;
        const isSelected = dayEl.classList.contains('selected');

        if (isSelected) {
            // Deselect
            this.selectedChangeDates.delete(date);
            dayEl.classList.remove('selected');
        } else {
            // Select
            this.selectedChangeDates.add(date);
            dayEl.classList.add('selected');
        }

        this.updateChangeSelectedCount();
    }

    findImageById(imageId) {
        if (!this.changeMonitoringImages) return null;
        return this.changeMonitoringImages.find(img => img.id === imageId);
    }

    createChangeDateItem(date, image, count) {
        const item = document.createElement('div');
        item.className = 'change-date-item';
        item.dataset.date = date;
        item.dataset.imageId = image.id;

        const dateObj = new Date(date);
        const cloudCover = image.cloud_cover ? image.cloud_cover.toFixed(1) : 'N/A';
        const aoiCoverage = image.aoi_overlap ? (image.aoi_overlap * 100).toFixed(1) : 'N/A';

        item.innerHTML = `
            <div class="date-header">
                <div class="date-label">${dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</div>
                ${count > 1 ? `<div class="image-count">${count} imgs</div>` : ''}
            </div>
            <div class="date-info">
                <div class="info-item">☁️ ${cloudCover}%</div>
                <div class="info-item">📍 ${aoiCoverage}%</div>
            </div>
            <div class="date-actions">
                <button class="view-tile-btn" title="View satellite tile">🛰️</button>
                <input type="checkbox" id="date-${date}" title="Select for analysis" />
            </div>
        `;

        const checkbox = item.querySelector('input[type="checkbox"]');
        const viewBtn = item.querySelector('.view-tile-btn');

        // Initialize change monitoring layers map if not exists
        if (!this.changeMonitoringLayers) {
            this.changeMonitoringLayers = new Map();
        }

        // Checkbox for selection
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                this.selectedChangeDates.add(date);
                item.classList.add('selected');
            } else {
                this.selectedChangeDates.delete(date);
                item.classList.remove('selected');
            }
            this.updateChangeSelectedCount();
        });

        // View satellite tile button
        viewBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await this.toggleChangeRawTile(date, image, viewBtn, item);
        });

        // Click on date item to select checkbox
        item.addEventListener('click', (e) => {
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') {
                checkbox.checked = !checkbox.checked;
                checkbox.dispatchEvent(new Event('change'));
            }
        });

        return item;
    }

    async toggleChangeRawTile(date, image, viewBtn, itemElement) {
        const layerKey = `change-raw-${date}`;

        // Initialize image data cache if not exists
        if (!this.changeImageDataCache) {
            this.changeImageDataCache = new Map();
        }

        // Check if layer is currently visible on map
        const isVisible = this.changeMonitoringLayers.has(layerKey);

        if (isVisible) {
            // Just hide the layer (don't delete from cache)
            if (window.mapManager && window.mapManager.imageLayers && layerKey in window.mapManager.imageLayers) {
                window.mapManager.removeImageLayer(layerKey);
            }
            this.changeMonitoringLayers.delete(layerKey);
            viewBtn.textContent = '👁';
            viewBtn.classList.remove('active');
            itemElement.classList.remove('raw-visible');
            return;
        }

        // Check cache first - if we already have the data, just show it
        if (this.changeImageDataCache.has(layerKey)) {
            const cachedData = this.changeImageDataCache.get(layerKey);
            window.mapManager.addImageLayer(layerKey, cachedData);
            this.changeMonitoringLayers.set(layerKey, { type: 'raw', data: cachedData });
            viewBtn.textContent = '👁';
            viewBtn.classList.add('active');
            itemElement.classList.add('raw-visible');
            console.log(`Using cached raw tile for ${date}`);
            return;
        }

        // Show loading - need to fetch from backend
        viewBtn.textContent = '⏳';
        viewBtn.disabled = true;

        try {
            // Get GEE tile URL (same as Satellite Search)
            const response = await fetch(`${this.apiBaseUrl}/api/get-gee-tile`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    item_id: image.id,
                    bbox: window.mapManager.getCurrentBounds()
                })
            });

            if (response.ok) {
                const tileData = await response.json();

                // Create image data object
                const imageData = {
                    display_url: tileData.static_url,
                    tile_template: tileData.tile_template,
                    tile_bounds: tileData.bounds,
                    display_type: 'tile'
                };

                // Cache the image data for future use
                this.changeImageDataCache.set(layerKey, imageData);

                // Add to map using the same method as Satellite Search
                window.mapManager.addImageLayer(layerKey, imageData);
                
                this.changeMonitoringLayers.set(layerKey, { type: 'raw', data: imageData });
                viewBtn.textContent = '👁';
                viewBtn.classList.add('active');
                itemElement.classList.add('raw-visible');
                
                this.showNotification(`Showing raw image for ${date}`, 'success');
            } else {
                throw new Error('Failed to get tile URL');
            }
        } catch (error) {
            console.error('Error loading tile:', error);
            this.showNotification('Error loading tile: ' + error.message, 'error');
            viewBtn.textContent = '👁';
        } finally {
            viewBtn.disabled = false;
        }
    }

    async toggleChangeSpectralTile(date, image, spectralBtn, itemElement) {
        const spectralMethod = document.getElementById('spectral-method')?.value || 'ndvi';
        const layerKey = `change-spectral-${date}-${spectralMethod}`;

        // Initialize image data cache if not exists
        if (!this.changeImageDataCache) {
            this.changeImageDataCache = new Map();
        }

        // Check if layer is currently visible on map
        const isVisible = this.changeMonitoringLayers.has(layerKey);

        if (isVisible) {
            // Just hide the layer (don't delete from cache)
            if (window.mapManager && window.mapManager.imageLayers && layerKey in window.mapManager.imageLayers) {
                window.mapManager.removeImageLayer(layerKey);
            }
            this.changeMonitoringLayers.delete(layerKey);
            spectralBtn.textContent = '📊';
            spectralBtn.classList.remove('active');
            itemElement.classList.remove('spectral-visible');
            return;
        }

        // Check cache first - if we already have the data, just show it
        if (this.changeImageDataCache.has(layerKey)) {
            const cachedData = this.changeImageDataCache.get(layerKey);
            window.mapManager.addImageLayer(layerKey, cachedData);
            this.changeMonitoringLayers.set(layerKey, { type: 'spectral', data: cachedData });
            spectralBtn.textContent = '📊';
            spectralBtn.classList.add('active');
            itemElement.classList.add('spectral-visible');
            console.log(`Using cached spectral tile for ${date} (${spectralMethod})`);
            return;
        }

        // Show loading - need to fetch from backend
        spectralBtn.textContent = '⏳';
        spectralBtn.disabled = true;

        try {
            // Request processed spectral image from backend
            const response = await fetch(`${this.apiBaseUrl}/api/process-spectral-image`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    item_id: image.id,
                    bbox: window.mapManager.getCurrentBounds(),
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    spectral_method: spectralMethod
                })
            });

            if (response.ok) {
                const result = await response.json();

                if (result.tile_url || result.url) {
                    // Create image data object for spectral result
                    // This is a static PNG overlay, NOT a tile service
                    const imageData = {
                        display_url: result.static_url || result.url || result.tile_url,
                        tile_bounds: result.bounds,
                        display_type: 'static'  // Static image, not tile service
                    };

                    // Cache the image data for future use
                    this.changeImageDataCache.set(layerKey, imageData);

                    // Add to map
                    window.mapManager.addImageLayer(layerKey, imageData);
                    
                    this.changeMonitoringLayers.set(layerKey, { type: 'spectral', data: imageData });
                    spectralBtn.textContent = '📊';
                    spectralBtn.classList.add('active');
                    itemElement.classList.add('spectral-visible');
                    
                    this.showNotification(`Showing ${spectralMethod.toUpperCase()} for ${date}`, 'success');
                } else {
                    throw new Error('No tile URL in response');
                }
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to process spectral image');
            }
        } catch (error) {
            console.error('Error loading spectral tile:', error);
            this.showNotification('Error loading spectral image: ' + error.message, 'error');
            spectralBtn.textContent = '📊';
        } finally {
            spectralBtn.disabled = false;
        }
    }

    updateChangeSelectedCount() {
        const count = this.selectedChangeDates ? this.selectedChangeDates.size : 0;
        document.getElementById('change-selected-count').textContent = count;
        document.getElementById('run-change-analysis-btn').disabled = count < 2;
        
        // Update Select All button text
        const selectAllBtn = document.getElementById('select-all-dates-btn');
        if (selectAllBtn && this.changeMonitoringImages) {
            const totalImages = this.changeMonitoringImages.length;
            if (count === totalImages && totalImages > 0) {
                selectAllBtn.textContent = '✗ Deselect All';
            } else {
                selectAllBtn.textContent = '✓ Select All';
            }
        }
    }

    toggleSelectAllDates() {
        if (!this.changeMonitoringImages || this.changeMonitoringImages.length === 0) {
            this.showNotification('No images to select', 'error');
            return;
        }

        // Check if all are already selected
        const allSelected = this.selectedChangeDates && 
                           this.selectedChangeDates.size === this.changeMonitoringImages.length;

        if (allSelected) {
            // Deselect all
            this.selectedChangeDates.clear();
            
            // Update UI - uncheck all checkboxes and remove selected class
            document.querySelectorAll('.calendar-day.has-data').forEach(dayEl => {
                dayEl.classList.remove('selected');
            });
            document.querySelectorAll('.date-item').forEach(item => {
                item.classList.remove('selected');
                const checkbox = item.querySelector('input[type="checkbox"]');
                if (checkbox) checkbox.checked = false;
            });
        } else {
            // Select all
            if (!this.selectedChangeDates) {
                this.selectedChangeDates = new Set();
            }
            
            this.changeMonitoringImages.forEach(img => {
                const date = img.datetime?.split('T')[0];
                if (date) {
                    this.selectedChangeDates.add(date);
                }
            });
            
            // Update UI - check all checkboxes and add selected class
            document.querySelectorAll('.calendar-day.has-data').forEach(dayEl => {
                dayEl.classList.add('selected');
            });
            document.querySelectorAll('.date-item').forEach(item => {
                item.classList.add('selected');
                const checkbox = item.querySelector('input[type="checkbox"]');
                if (checkbox) checkbox.checked = true;
            });
        }

        this.updateChangeSelectedCount();
    }

    async runChangeAnalysis() {
        if (!this.selectedChangeDates || this.selectedChangeDates.size < 2) {
            this.showNotification('Please select at least 2 dates', 'error');
            return;
        }

        // Get selected spectral method
        const spectralMethod = document.getElementById('spectral-method')?.value || 'ndvi';

        this.showLoading(`Running ${spectralMethod.toUpperCase()} time series analysis...`);

        try {
            const imageIds = Array.from(this.selectedChangeDates).map(date => {
                const img = this.changeMonitoringImages.find(i =>
                    i.datetime && i.datetime.startsWith(date)
                );
                return img ? img.id : null;
            }).filter(id => id !== null);

            console.log('Running analysis for image IDs:', imageIds);
            console.log('Spectral method:', spectralMethod);

            const response = await fetch('/api/change-monitoring', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_ids: imageIds,
                    bbox: window.mapManager.getCurrentBounds(),
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    spectral_method: spectralMethod,
                    model_id: this.changeMonitoringModel?.modelId || 'model2' // Use selected model or default to NDVI
                })
            });

            if (response.ok) {
                const results = await response.json();
                console.log('Analysis results:', results);
                this.displayChangeAnalysisResults(results, spectralMethod);
                this.showNotification('Analysis complete!', 'success');
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error running analysis:', error);
            this.showNotification('Error running analysis: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayChangeAnalysisResults(results, spectralMethod = 'ndvi') {
        const resultsDiv = document.getElementById('change-analysis-results');
        resultsDiv.classList.remove('hidden');

        // Store analysis data for CSV export
        this.changeAnalysisData = {
            results: results.results || [],
            spectral_method: spectralMethod,
            statistics: results.statistics
        };
        
        // Mark analysis as completed and re-render calendar to show analyze buttons
        this.changeAnalysisCompleted = true;
        if (this.imagesByDate) {
            const calendarGrid = document.getElementById('change-calendar-grid');
            if (calendarGrid) {
                this.renderCalendarView(calendarGrid, this.imagesByDate);
            }
        }

        // Prepare data for chart
        // Backend returns { results: [...], statistics: {...} }
        // Each result item has { date, mean_spectral, area_km2, image_url, ... }
        const periods = results.results || [];
        const dates = periods.map(p => p.date);
        const values = periods.map(p => p.mean_spectral !== undefined ? p.mean_spectral : p.area_km2);

        console.log('Chart data:', { dates, values, spectralMethod });

        // Use Chart.js to display time series
        const canvas = document.getElementById('change-time-series-chart');
        const ctx = canvas.getContext('2d');

        // Destroy existing chart if any
        if (this.changeChart) {
            this.changeChart.destroy();
        }

        // Get label and color for spectral method
        const methodInfo = this.getSpectralMethodInfo(spectralMethod);

        this.changeChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: methodInfo.label,
                    data: values,
                    borderColor: methodInfo.color,
                    backgroundColor: methodInfo.bgColor,
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Time Series Analysis - ${spectralMethod.toUpperCase()}`
                    },
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: methodInfo.yAxisLabel
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                }
            }
        });

        // Render analyzed images list
        this.renderAnalyzedImagesList(periods, spectralMethod);

        // Scroll to results
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    renderAnalyzedImagesList(periods, spectralMethod) {
        const listContainer = document.getElementById('analyzed-images-list');
        if (!listContainer) return;

        listContainer.innerHTML = '';
        
        // Store the analyzed periods for toggle functions
        this.analyzedPeriods = periods;

        periods.forEach((period, index) => {
            const date = period.date;
            const imageId = period.image_id;
            const areaKm2 = period.area_km2?.toFixed(4) || '0';
            const meanSpectral = period.mean_spectral?.toFixed(3) || '0';
            
            // Find corresponding image from changeMonitoringImages for cloud info
            const imageInfo = this.changeMonitoringImages?.find(img => 
                img.datetime?.startsWith(date) || img.id === imageId
            );
            const cloudPercent = imageInfo?.cloud_cover?.toFixed(0) || 
                                 imageInfo?.properties?.cloud_cover?.toFixed(0) || '—';
            // aoi_overlap is returned as fraction (0-1), convert to percentage
            let aoiCoverage = '—';
            if (imageInfo?.aoi_overlap !== undefined) {
                aoiCoverage = (imageInfo.aoi_overlap * 100).toFixed(0);
            } else if (imageInfo?.properties?.aoi_overlap !== undefined) {
                aoiCoverage = (imageInfo.properties.aoi_overlap * 100).toFixed(0);
            } else if (imageInfo?.aoi_coverage !== undefined) {
                aoiCoverage = imageInfo.aoi_coverage.toFixed(0);
            }

            const item = document.createElement('div');
            item.className = 'analyzed-image-item';
            item.dataset.date = date;
            item.dataset.imageId = imageId;
            item.dataset.index = index;

            item.innerHTML = `
                <div class="analyzed-image-info">
                    <span class="analyzed-image-date">📅 ${date}</span>
                    <div class="analyzed-image-stats">
                        <span class="analyzed-image-stat" title="AOI Coverage">📐 ${aoiCoverage}%</span>
                        <span class="analyzed-image-stat" title="Cloud Cover">☁️ ${cloudPercent}%</span>
                        <span class="analyzed-image-stat" title="Mean ${spectralMethod.toUpperCase()}">📈 ${meanSpectral}</span>
                    </div>
                </div>
                <div class="analyzed-image-actions">
                    <button class="analyzed-image-btn tile-btn" title="Toggle satellite tile">🛰️</button>
                    <button class="analyzed-image-btn spectral-btn" title="Toggle ${spectralMethod.toUpperCase()} analysis">📊</button>
                </div>
            `;

            // Add event listeners for toggle buttons
            const tileBtn = item.querySelector('.tile-btn');
            const spectralBtn = item.querySelector('.spectral-btn');

            tileBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                await this.toggleAnalyzedTile(date, imageId, tileBtn, item);
            });

            spectralBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                await this.toggleAnalyzedSpectral(date, imageId, spectralBtn, item, spectralMethod);
            });

            listContainer.appendChild(item);
        });
    }

    async toggleAnalyzedTile(date, imageId, btn, itemElement) {
        const layerKey = `analyzed-tile-${date}`;

        // Check if layer is currently visible
        const isVisible = this.changeMonitoringLayers?.has(layerKey);

        if (isVisible) {
            // Hide the layer
            if (window.mapManager?.imageLayers?.[layerKey]) {
                window.mapManager.removeImageLayer(layerKey);
            }
            this.changeMonitoringLayers.delete(layerKey);
            btn.classList.remove('active');
            itemElement.classList.remove('tile-visible');
            return;
        }

        // Show loading
        btn.textContent = '⏳';
        btn.disabled = true;

        try {
            // Get tile URL from backend
            const response = await fetch('/api/get-gee-tile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: imageId,
                    bbox: window.mapManager.getCurrentBounds()
                })
            });

            if (response.ok) {
                const tileData = await response.json();
                const imageData = {
                    tile_template: tileData.tile_template,
                    tile_bounds: tileData.bounds,
                    display_type: 'tile'
                };

                window.mapManager.addImageLayer(layerKey, imageData);
                this.changeMonitoringLayers.set(layerKey, { type: 'tile', data: imageData });
                btn.classList.add('active');
                itemElement.classList.add('tile-visible');
            } else {
                throw new Error('Failed to get tile');
            }
        } catch (error) {
            console.error('Error loading tile:', error);
            this.showNotification('Error loading tile', 'error');
        } finally {
            btn.textContent = '🛰️';
            btn.disabled = false;
        }
    }

    async toggleAnalyzedSpectral(date, imageId, btn, itemElement, spectralMethod) {
        const layerKey = `analyzed-spectral-${date}-${spectralMethod}`;

        // Initialize cache if not exists
        if (!this.changeImageDataCache) {
            this.changeImageDataCache = new Map();
        }

        // Check if layer is currently visible
        const isVisible = this.changeMonitoringLayers?.has(layerKey);

        if (isVisible) {
            // Hide the layer
            if (window.mapManager?.imageLayers?.[layerKey]) {
                window.mapManager.removeImageLayer(layerKey);
            }
            this.changeMonitoringLayers.delete(layerKey);
            btn.classList.remove('active');
            itemElement.classList.remove('spectral-visible');
            return;
        }

        // Check cache first
        if (this.changeImageDataCache.has(layerKey)) {
            const cachedData = this.changeImageDataCache.get(layerKey);
            window.mapManager.addImageLayer(layerKey, cachedData);
            this.changeMonitoringLayers.set(layerKey, { type: 'spectral', data: cachedData });
            btn.classList.add('active');
            itemElement.classList.add('spectral-visible');
            return;
        }

        // Show loading
        btn.textContent = '⏳';
        btn.disabled = true;

        try {
            // Request spectral image - TIF should already be cached from analysis
            const response = await fetch('/api/process-spectral-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: imageId,
                    bbox: window.mapManager.getCurrentBounds(),
                    geometry: window.mapManager.getCurrentGeoJSON()?.geometry,
                    spectral_method: spectralMethod
                })
            });

            if (response.ok) {
                const result = await response.json();
                
                const imageData = {
                    display_url: result.static_url || result.url || result.tile_url,
                    tile_bounds: result.bounds,
                    display_type: 'static'
                };

                // Cache for future use
                this.changeImageDataCache.set(layerKey, imageData);

                window.mapManager.addImageLayer(layerKey, imageData);
                this.changeMonitoringLayers.set(layerKey, { type: 'spectral', data: imageData });
                btn.classList.add('active');
                itemElement.classList.add('spectral-visible');
            } else {
                throw new Error('Failed to get spectral image');
            }
        } catch (error) {
            console.error('Error loading spectral image:', error);
            this.showNotification('Error loading spectral image: ' + error.message, 'error');
        } finally {
            btn.textContent = '📊';
            btn.disabled = false;
        }
    }

    getSpectralMethodInfo(method) {
        const infoMap = {
            'ndvi': {
                label: 'NDVI Mean Value',
                color: '#1a73e8',
                bgColor: 'rgba(26, 115, 232, 0.1)',
                yAxisLabel: 'NDVI Value'
            },
            'ndmi': {
                label: 'NDMI Mean Value',
                color: '#34a853',
                bgColor: 'rgba(52, 168, 83, 0.1)',
                yAxisLabel: 'NDMI Value'
            },
            'mvi': {
                label: 'MVI Mean Value',
                color: '#f9ab00',
                bgColor: 'rgba(249, 171, 0, 0.1)',
                yAxisLabel: 'MVI Value'
            }
        };
        return infoMap[method] || infoMap['ndvi'];
    }

    downloadChangeAnalysisCSV() {
        if (!this.changeAnalysisData || !this.changeAnalysisData.results) {
            this.showNotification('No analysis data to download', 'error');
            return;
        }

        const results = this.changeAnalysisData.results;
        const spectralMethod = this.changeAnalysisData.spectral_method || 'ndvi';

        // Create CSV header
        let csv = 'Date,Mean Value,Area (km²)\n';

        // Add data rows
        results.forEach(result => {
            const date = result.date || '';
            const meanValue = result.mean_spectral !== undefined ? result.mean_spectral : '';
            const area = result.area_km2 !== undefined ? result.area_km2 : '';
            csv += `${date},${meanValue},${area}\n`;
        });

        // Create blob and download
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        const timestamp = new Date().toISOString().split('T')[0];
        link.setAttribute('href', url);
        link.setAttribute('download', `change_monitoring_${spectralMethod}_${timestamp}.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        this.showNotification('CSV downloaded successfully', 'success');
    }
}

// PlatformController class ends here - initialization is now handled in index.html 