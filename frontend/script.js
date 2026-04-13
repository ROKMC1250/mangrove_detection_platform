/**
 * ===== MANGROVE DETECTION PLATFORM =====
 * Google Maps Style Interface Controller
 * Slim facade that coordinates modular components
 */

class PlatformController {
    constructor() {
        // API configuration
        this.apiBaseUrl = '';
        console.log(`🔗 API Base URL: ${this.apiBaseUrl || 'Same origin (relative)'}`);

        // Core state
        this.currentImages = [];
        this.s1Images = [];
        this.s2Images = [];
        this.selectedImageId = null;
        this.currentSatelliteType = 's2';
        this.selectedSatellite = 's2';
        this.currentNotification = null;
        this._progressStop = false;

        // Processed data
        this.processedImageData = null;
        this.currentProcessedImageId = null;
        this.lastAnalysisResults = null;
        this._modelMaskUrls = null;

        // Change monitoring state
        this.changeImages = [];
        this.selectedChangeDates = new Set();
        this.imagesByDate = {};
        this.calendarCurrentYear = new Date().getFullYear();
        this.calendarCurrentMonth = new Date().getMonth();

        // S1 visualization state
        this.s1Channels = { r: 'VV', g: 'VH', b: 'VV' };
        this.s1MinMax = { min: -25, max: 0 };

        // Keyboard shortcut mode state
        this.keyboardShortcutMode = false;
        this.keyboardShortcutMap = {
            '1': 'B2', '2': 'B3', '3': 'B4', '4': 'B5', '5': 'B6',
            '6': 'B7', '7': 'B8', '8': 'B8A', '9': 'B11'
        };
        this._originalTileUrls = {};  // layerId -> { url, bounds } for RGB restore

        // Initialize modules
        this.initModules();
        this.init();
    }

    /**
     * Initialize modular controllers
     */
    initModules() {
        // Initialize UI manager
        if (typeof UIManager !== 'undefined') {
            this.uiManager = new UIManager(this);
        }

        // Initialize search results controller
        if (typeof SearchResultsController !== 'undefined') {
            this.searchResultsController = new SearchResultsController(this);
        }

        // Initialize image processor controller
        if (typeof ImageProcessorController !== 'undefined') {
            this.imageProcessorController = new ImageProcessorController(this);
        }

        // Initialize API client
        if (typeof ApiClient !== 'undefined') {
            this.apiClient = new ApiClient(this.apiBaseUrl);
        }

        // Initialize image search controller
        if (typeof ImageSearchController !== 'undefined') {
            this.imageSearch = new ImageSearchController(this);
        }

        // Initialize analysis controller
        if (typeof AnalysisController !== 'undefined') {
            this.analysisController = new AnalysisController(this);
        }

        // Initialize change monitoring controller (disabled)
        // if (typeof ChangeMonitoringController !== 'undefined') {
        //     this.changeMonitoring = new ChangeMonitoringController(this);
        // }

        // Initialize custom visualization controller
        if (typeof CustomVisualizationController !== 'undefined') {
            this.customViz = new CustomVisualizationController(this);
        }

        // Initialize threshold controller
        if (typeof ThresholdController !== 'undefined') {
            this.thresholdController = new ThresholdController(this);
        }

        // Initialize target detection controller
        if (typeof TargetDetectionController !== 'undefined') {
            this.targetDetection = new TargetDetectionController(this);
        }

        // Initialize SAM2 controller
        if (typeof SAM2Controller !== 'undefined') {
            this.sam2Controller = new SAM2Controller(this);
        }

        // Initialize spectral analysis controller
        if (typeof SpectralAnalysisController !== 'undefined') {
            this.spectralAnalysis = new SpectralAnalysisController(this);
        }

        // Initialize local image controller
        if (typeof LocalImageController !== 'undefined') {
            this.localImage = new LocalImageController(this);
        }

        // Initialize spectral inspector
        if (typeof SpectralInspectorController !== 'undefined') {
            this.spectralInspector = new SpectralInspectorController(this);
        }

        // Initialize layer control panel
        if (typeof LayerControlPanel !== 'undefined' && window.mapManager) {
            this.layerControlPanel = new LayerControlPanel(window.mapManager);
        }

        // Initialize dual map controller
        if (typeof DualMapController !== 'undefined') {
            this.dualMapController = new DualMapController(this);
        }

        console.log('📦 Modular controllers initialized');
    }

    async init() {
        console.log('🚀 Initializing Platform Controller...');

        await this.loadConfig();
        this.setupEventListeners();
        
        if (this.searchResultsController) {
            this.searchResultsController.setupDownloadDropdownClose();
        }
        
        this.initializeDateInputs();
        
        if (this.uiManager) {
            this.uiManager.setupTabControls();
            this.uiManager.setupRightPanelTabControls();
        }

        console.log('✅ Platform Controller Ready!');
    }

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            if (response.ok) {
                const config = await response.json();
                console.log('📋 Configuration loaded:', config);
            }
        } catch (error) {
            console.warn('Config load failed:', error);
        }
    }

    setupEventListeners() {
        // Coordinate search handler
        this.setupCoordinateSearch();

        // Draw AOI button
        const drawBtn = document.getElementById('draw-rectangle');
        if (drawBtn) {
            drawBtn.addEventListener('click', () => {
                if (window.mapManager) {
                    window.mapManager.startDrawing();
                }
            });
        }

        // Clear button
        const clearBtn = document.getElementById('clear-drawings');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                if (window.mapManager) {
                    window.mapManager.clearDrawings();
                }
                this.currentImages = [];
                this.selectedImageId = null;
            });
        }

        // Panel toggle
        const panelToggle = document.getElementById('panel-toggle-btn');
        const panel = document.getElementById('control-panel');
        if (panelToggle && panel) {
            panelToggle.addEventListener('click', () => {
                const collapsed = panel.classList.toggle('collapsed');
                panelToggle.textContent = collapsed ? '▶' : '◀';
                panelToggle.title = collapsed ? 'Show panel' : 'Hide panel';
                // Resize map
                setTimeout(() => {
                    if (window.mapManager?.map) window.mapManager.map.invalidateSize();
                }, 350);
            });
        }

        // Right panel toggle (collapse/expand, not hide)
        const rightPanelToggle = document.getElementById('right-panel-toggle-btn');
        if (rightPanelToggle) {
            rightPanelToggle.addEventListener('click', () => {
                const rp = document.getElementById('right-panel');
                if (rp) {
                    const collapsed = rp.classList.toggle('collapsed');
                    rightPanelToggle.textContent = collapsed ? '◀' : '▶';
                    setTimeout(() => {
                        if (window.mapManager?.map) window.mapManager.map.invalidateSize();
                    }, 350);
                }
            });
        }

        // "Analyze →" buttons
        const openAnalysisBtn = document.getElementById('open-analysis-btn');
        if (openAnalysisBtn) {
            openAnalysisBtn.addEventListener('click', () => {
                // Check if any image is displayed on the map
                const tileLayers = window.mapManager?.tileLayers || {};
                const hasImage = Object.keys(tileLayers).some(id => id.startsWith('preview-'));
                if (!hasImage) {
                    this.showNotification('Display an image first by clicking it in search results', 'warning');
                    return;
                }
                // Trigger processing on the selected/displayed image
                const imageId = this.selectedImageId || this.currentProcessedImageId;
                if (!imageId) {
                    this.showNotification('Select an image first', 'warning');
                    return;
                }
                this.processImage(imageId);
            });
        }
        const localOpenAnalysisBtn = document.getElementById('local-open-analysis-btn');
        if (localOpenAnalysisBtn) {
            localOpenAnalysisBtn.addEventListener('click', () => {
                if (!this.localImage) return;
                // Delegate to LocalImageController which handles mode checks
                this.localImage.runLocalAnalysis();
            });
        }

        // Compare buttons (dual map)
        const compareBtn = document.getElementById('compare-btn');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => {
                if (this.dualMapController) this.dualMapController.toggle();
            });
        }
        const localCompareBtn = document.getElementById('local-compare-btn');
        if (localCompareBtn) {
            localCompareBtn.addEventListener('click', () => {
                if (this.dualMapController) this.dualMapController.toggle();
            });
        }
        const exitCompareBtn = document.getElementById('exit-compare-btn');
        if (exitCompareBtn) {
            exitCompareBtn.addEventListener('click', () => {
                if (this.dualMapController) this.dualMapController.deactivate();
            });
        }

        // File upload handlers
        this.setupFileUploadHandlers();

        // Search buttons
        this.setupSearchButtons();

        // Satellite tab handlers
        this.setupSatelliteTabs();

        // Visualization preset handlers
        this.setupVisualizationHandlers();

        // Map callbacks
        this.setupMapCallbacks();
    }

    setupFileUploadHandlers() {
        const uploadBtn = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        if (uploadBtn && fileInput) {
            uploadBtn.addEventListener('click', () => fileInput.click());
            
            fileInput.addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;

                const ext = file.name.split('.').pop().toLowerCase();
                
                try {
                    if (ext === 'kml' || ext === 'kmz') {
                        await window.mapManager.loadKmlFile(file);
                        this.showNotification('KML file loaded successfully', 'success');
                    } else if (ext === 'zip' || ext === 'shp') {
                        await window.mapManager.loadShpFile(file);
                        this.showNotification('SHP file loaded successfully', 'success');
                    } else {
                        this.showNotification('Unsupported file format', 'error');
                    }
                } catch (error) {
                    console.error('File load error:', error);
                    this.showNotification(`Error loading file: ${error.message}`, 'error');
                }
                
                fileInput.value = '';
            });

            // Drag and drop
            uploadBtn.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadBtn.classList.add('dragover');
            });
            
            uploadBtn.addEventListener('dragleave', () => {
                uploadBtn.classList.remove('dragover');
            });
            
            uploadBtn.addEventListener('drop', async (e) => {
                e.preventDefault();
                uploadBtn.classList.remove('dragover');
                
                const file = e.dataTransfer.files[0];
                if (file) {
                    fileInput.files = e.dataTransfer.files;
                    fileInput.dispatchEvent(new Event('change'));
                }
            });
        }
    }

    setupSearchButtons() {
        // Main Search button (single button for current satellite)
        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) {
            searchBtn.addEventListener('click', () => this.searchImages());
        }

        // Change monitoring search (disabled)
        // const changeSearchBtn = document.getElementById('search-change-images-btn');
        // if (changeSearchBtn) {
        //     changeSearchBtn.addEventListener('click', () => {
        //         if (this.changeMonitoring) {
        //             this.changeMonitoring.searchChangeImages();
        //         }
        //     });
        // }
    }

    setupSatelliteTabs() {
        const satelliteBtns = document.querySelectorAll('.satellite-btn');
        satelliteBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                satelliteBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const satellite = btn.dataset.satellite;
                this.switchSatelliteContent(satellite);
            });
        });
    }

    switchSatelliteContent(satellite) {
        this.currentSatelliteType = satellite;
        this.selectedSatellite = satellite;
        
        // Hide all channel options
        const s2Options = document.getElementById('s2-channel-options');
        const s1Options = document.getElementById('s1-channel-options');
        
        if (s2Options) s2Options.classList.add('hidden');
        if (s1Options) s1Options.classList.add('hidden');
        
        // Show selected channel options
        if (satellite === 's2' && s2Options) {
            s2Options.classList.remove('hidden');
        } else if (satellite === 's1' && s1Options) {
            s1Options.classList.remove('hidden');
        }
        
        // Disable analyze button for S1 (no spectral analysis available)
        const analyzeBtn = document.getElementById('open-analysis-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = satellite === 's1';
            analyzeBtn.title = satellite === 's1' ? 'Analysis not available for Sentinel-1' : '';
        }

        console.log(`Switched to satellite: ${satellite}`);
    }

    setupVisualizationHandlers() {
        // S2 presets
        const s2Presets = document.querySelectorAll('.s2-preset');
        s2Presets.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active from all S2 presets
                s2Presets.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const preset = btn.dataset.preset;
                this.applyS2Preset(preset);
            });
        });

        // S1 presets (inside .s1-presets container)
        const s1PresetsContainer = document.querySelector('.s1-presets');
        if (s1PresetsContainer) {
            const s1Presets = s1PresetsContainer.querySelectorAll('.preset-btn');
            s1Presets.forEach(btn => {
                btn.addEventListener('click', () => {
                    // Remove active from all S1 presets
                    s1Presets.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    
                    const preset = btn.dataset.preset;
                    this.applyS1Preset(preset);
                });
            });
        }

        // Apply S2 visualization button
        const applyS2VisBtn = document.getElementById('apply-s2-vis');
        if (applyS2VisBtn) {
            applyS2VisBtn.addEventListener('click', () => this.applyVisualizationToAllImages('s2'));
        }

        // Apply S1 visualization button
        const applyS1VisBtn = document.getElementById('apply-s1-vis');
        if (applyS1VisBtn) {
            applyS1VisBtn.addEventListener('click', () => this.applyVisualizationToAllImages('s1'));
        }

        // Setup drag-and-drop for band selection
        this.setupBandDragDrop();

        // Percentile sliders
        this.setupPercentileSliders();

        // Keyboard shortcut mode
        this.setupKeyboardShortcutMode();
    }

    setupBandDragDrop() {
        // Setup for S2
        this.initDragDropForSatellite('s2');
        // Setup for S1
        this.initDragDropForSatellite('s1');
    }

    initDragDropForSatellite(prefix) {
        const bandPool = document.getElementById(`${prefix}-band-pool`);
        const slots = document.querySelectorAll(`#${prefix}-slot-r, #${prefix}-slot-g, #${prefix}-slot-b`);
        
        if (!bandPool) return;

        // Make chips draggable
        const chips = bandPool.querySelectorAll('.band-chip');
        chips.forEach(chip => {
            chip.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', chip.dataset.band);
                chip.classList.add('dragging');
            });
            chip.addEventListener('dragend', () => {
                chip.classList.remove('dragging');
            });
        });

        // Setup slots as drop targets
        slots.forEach(slot => {
            slot.addEventListener('dragover', (e) => {
                e.preventDefault();
                slot.classList.add('drag-over');
            });
            slot.addEventListener('dragleave', () => {
                slot.classList.remove('drag-over');
            });
            slot.addEventListener('drop', (e) => {
                e.preventDefault();
                slot.classList.remove('drag-over');
                const band = e.dataTransfer.getData('text/plain');
                const slotBand = slot.querySelector('.slot-band');
                if (slotBand) {
                    slotBand.textContent = band;
                    slotBand.dataset.band = band;
                }
            });
        });
    }

    setupPercentileSliders() {
        this._initDualRange('s2', 0, 10000);
        this._initDualRange('s1', -30, 5);
    }

    _initDualRange(prefix, dataMin, dataMax) {
        const lowSlider = document.getElementById(`${prefix}-pct-low`);
        const highSlider = document.getElementById(`${prefix}-pct-high`);
        const lowInput = document.getElementById(`${prefix}-pct-low-val`);
        const highInput = document.getElementById(`${prefix}-pct-high-val`);
        const minHidden = document.getElementById(`${prefix}-min`);
        const maxHidden = document.getElementById(`${prefix}-max`);
        if (!lowSlider || !highSlider) return;

        const sync = (source) => {
            let lo, hi;
            if (source === 'slider') {
                lo = parseFloat(lowSlider.value);
                hi = parseFloat(highSlider.value);
                if (lo > hi) { lowSlider.value = hi; lo = hi; }
                if (hi < lo) { highSlider.value = lo; hi = lo; }
                lowInput.value = lo.toFixed(1);
                highInput.value = hi.toFixed(1);
            } else {
                lo = Math.max(0, Math.min(100, parseFloat(lowInput.value) || 0));
                hi = Math.max(0, Math.min(100, parseFloat(highInput.value) || 100));
                if (lo > hi) lo = hi;
                lowSlider.value = lo;
                highSlider.value = hi;
            }
            const range = dataMax - dataMin;
            minHidden.value = Math.round(dataMin + range * lo / 100);
            maxHidden.value = Math.round(dataMin + range * hi / 100);
        };

        lowSlider.addEventListener('input', () => sync('slider'));
        highSlider.addEventListener('input', () => sync('slider'));
        lowInput.addEventListener('change', () => sync('input'));
        highInput.addEventListener('change', () => sync('input'));
    }

    setupKeyboardShortcutMode() {
        const toggle = document.getElementById('keyboard-shortcut-toggle');
        const slotsContainer = document.getElementById('keyboard-shortcut-slots');
        if (!toggle || !slotsContainer) return;

        // Toggle handler
        toggle.addEventListener('change', () => {
            this.keyboardShortcutMode = toggle.checked;
            slotsContainer.style.display = toggle.checked ? 'grid' : 'none';
        });

        // Drag-and-drop for shortcut slots
        const slots = slotsContainer.querySelectorAll('.shortcut-slot');
        slots.forEach(slot => {
            slot.addEventListener('dragover', (e) => {
                e.preventDefault();
                slot.classList.add('drag-over');
            });
            slot.addEventListener('dragleave', () => {
                slot.classList.remove('drag-over');
            });
            slot.addEventListener('drop', (e) => {
                e.preventDefault();
                slot.classList.remove('drag-over');
                const band = e.dataTransfer.getData('text/plain');
                if (!band) return;
                const label = slot.querySelector('.slot-band-label');
                if (label) {
                    label.textContent = band;
                    label.dataset.band = band;
                }
                slot.dataset.band = band;
                this.keyboardShortcutMap[slot.dataset.key] = band;
            });
        });

        // Global keydown listener
        document.addEventListener('keydown', (e) => {
            if (!this.keyboardShortcutMode) return;
            // Only when search tab is active
            const searchTab = document.getElementById('search-tab-content');
            if (!searchTab || !searchTab.classList.contains('active')) return;
            const tag = document.activeElement?.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            if (document.activeElement?.isContentEditable) return;
            if (e.key >= '1' && e.key <= '9') {
                e.preventDefault();
                this.applyGrayscaleBand(e.key);
            }
            if (e.key === '`' || e.key === '~') {
                e.preventDefault();
                this.restoreRGBTiles();
            }
        });
    }

    async applyGrayscaleBand(key) {
        const band = this.keyboardShortcutMap[key];
        if (!band) return;

        // Visual feedback on the slot
        const slot = document.querySelector(`.shortcut-slot[data-key="${key}"]`);
        if (slot) {
            slot.classList.add('active-key');
            setTimeout(() => slot.classList.remove('active-key'), 300);
        }

        // Get S2 tile layers
        const tileLayers = window.mapManager?.tileLayers || {};
        const layerIds = Object.keys(tileLayers).filter(id => {
            if (id.startsWith('change-preview-')) return false;
            if (!id.startsWith('preview-')) return false;
            const imageId = id.replace('preview-', '');
            return !(imageId.startsWith('S1A_') || imageId.startsWith('S1B_') || imageId.includes('S1_GRD'));
        });

        if (layerIds.length === 0) {
            this.showNotification('No S2 images displayed', 'warning');
            return;
        }

        const min = parseFloat(document.getElementById('s2-min')?.value) || 0;
        const max = parseFloat(document.getElementById('s2-max')?.value) || 3000;
        const aoi = this.getAOI();
        if (!aoi?.bbox) {
            this.showNotification('No AOI defined', 'warning');
            return;
        }

        this.showNotification(`Band ${band} (key ${key})`, 'info', 2000);

        // Capture original tile URLs before first grayscale switch
        for (const layerId of layerIds) {
            if (!this._originalTileUrls[layerId]) {
                const existingLayer = tileLayers[layerId];
                if (existingLayer && existingLayer._url) {
                    this._originalTileUrls[layerId] = {
                        url: existingLayer._url,
                        bounds: existingLayer.options.bounds
                    };
                }
            }
        }

        for (const layerId of layerIds) {
            const imageId = layerId.replace('preview-', '');
            try {
                const response = await fetch('/api/get-s2-tile-custom', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        item_id: imageId,
                        bbox: aoi.bbox,
                        geometry: aoi.geometry,
                        bands: [band, band, band],
                        min: min,
                        max: max
                    })
                });
                if (response.ok) {
                    const data = await response.json();
                    if (data.tile_template && window.mapManager) {
                        window.mapManager.hideTileLayer(layerId);
                        window.mapManager.showTileLayer(
                            layerId, data.tile_template, data.bounds,
                            `Grayscale ${band}: ${imageId}`
                        );
                    }
                }
            } catch (error) {
                console.error(`Grayscale band error for ${layerId}:`, error);
            }
        }
    }

    restoreRGBTiles() {
        if (Object.keys(this._originalTileUrls).length === 0) {
            this.showNotification('No original RGB to restore', 'warning', 2000);
            return;
        }
        for (const [layerId, original] of Object.entries(this._originalTileUrls)) {
            if (window.mapManager) {
                window.mapManager.hideTileLayer(layerId);
                window.mapManager.showTileLayer(
                    layerId, original.url, original.bounds,
                    `RGB: ${layerId.replace('preview-', '')}`
                );
            }
        }
        this.showNotification('Restored RGB (~)', 'info', 2000);
    }

    async applyVisualizationToAllImages(satellite) {
        const aoi = this.getAOI();
        if (!aoi || !aoi.bbox) {
            this.showNotification('No AOI defined', 'warning');
            return;
        }

        // Get all currently displayed tile layers for this satellite type
        const tileLayers = window.mapManager?.tileLayers || {};
        console.log('[VIS] Available tile layers:', Object.keys(tileLayers));
        
        const layerIds = Object.keys(tileLayers).filter(id => {
            // Skip change monitoring layers
            if (id.startsWith('change-preview-')) return false;
            
            if (!id.startsWith('preview-')) return false;
            
            // S1 images have IDs starting with S1A_ or S1B_ or contain S1
            const imageId = id.replace('preview-', '');
            const isS1Image = imageId.startsWith('S1A_') || imageId.startsWith('S1B_') || imageId.includes('S1_GRD');
            
            if (satellite === 's2') {
                return !isS1Image;
            } else if (satellite === 's1') {
                return isS1Image;
            }
            return false;
        });

        console.log(`[VIS] Found ${layerIds.length} ${satellite} layers:`, layerIds);

        if (layerIds.length === 0) {
            this.showNotification('No images displayed to update. Display images first by clicking them in the search results.', 'warning');
            return;
        }

        this.showLoading('Applying visualization...');

        const visParams = satellite === 's2' ? this.getS2VisualizationParams() : this.getS1VisualizationParams();

        try {
            let successCount = 0;
            for (const layerId of layerIds) {
                const imageId = layerId.replace('preview-', '');
                
                try {
                    const endpoint = satellite === 's1' ? '/api/get-s1-tile' : '/api/get-s2-tile-custom';
                    const requestBody = {
                        item_id: imageId,
                        bbox: aoi.bbox,
                        geometry: aoi.geometry,
                        bands: visParams.bands,
                        min: visParams.min,
                        max: visParams.max
                    };

                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.tile_template && window.mapManager) {
                            // Remove old layer and add new one with updated visualization
                            window.mapManager.hideTileLayer(layerId);
                            window.mapManager.showTileLayer(
                                layerId,
                                data.tile_template,
                                data.bounds,
                                `${satellite.toUpperCase()}: ${imageId}`
                            );
                            successCount++;
                        }
                    }
                } catch (error) {
                    console.error(`Failed to update ${layerId}:`, error);
                }
            }

            this.hideLoading();
            this.showNotification(`Visualization applied to ${successCount}/${layerIds.length} images`, 'success');

        } catch (error) {
            console.error('Apply visualization error:', error);
            this.hideLoading();
            this.showNotification('Failed to apply visualization', 'error');
        }
    }

    setupMapCallbacks() {
        if (window.mapManager) {
            window.mapManager.onAOIChange = (hasAOI) => {
                this.onAOIChange(hasAOI);
            };
        }
    }

    setupCoordinateSearch() {
        const coordInput = document.getElementById('coord-input');
        const goBtn = document.getElementById('go-to-coords');

        if (goBtn && coordInput) {
            goBtn.addEventListener('click', () => this.goToCoordinates());
            
            // Also trigger on Enter key
            coordInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.goToCoordinates();
                }
            });
        }
    }

    goToCoordinates() {
        const coordInput = document.getElementById('coord-input');
        if (!coordInput) return;

        const input = coordInput.value.trim();
        if (!input) {
            this.showNotification('Please enter coordinates', 'warning');
            return;
        }

        // Parse coordinates - support formats:
        // "26.761, 103.794" or "26.761 103.794" or "26.761,103.794"
        const parts = input.split(/[\s,]+/).filter(p => p.length > 0);
        
        if (parts.length !== 2) {
            this.showNotification('Invalid format. Use: lat, lng (e.g., 26.761, 103.794)', 'error');
            return;
        }

        const lat = parseFloat(parts[0]);
        const lng = parseFloat(parts[1]);

        if (isNaN(lat) || isNaN(lng)) {
            this.showNotification('Invalid coordinates. Use numbers only.', 'error');
            return;
        }

        // Validate coordinate ranges
        if (lat < -90 || lat > 90) {
            this.showNotification('Latitude must be between -90 and 90', 'error');
            return;
        }
        if (lng < -180 || lng > 180) {
            this.showNotification('Longitude must be between -180 and 180', 'error');
            return;
        }

        // Move map to coordinates
        if (window.mapManager && window.mapManager.map) {
            window.mapManager.map.setView([lat, lng], 14);
            
            // Add a temporary marker to show the location
            const marker = L.marker([lat, lng], {
                icon: L.divIcon({
                    className: 'coord-marker',
                    html: '<div class="coord-marker-inner">📍</div>',
                    iconSize: [30, 30],
                    iconAnchor: [15, 30]
                })
            }).addTo(window.mapManager.map);

            // Create popup
            marker.bindPopup(`<b>Location</b><br>Lat: ${lat.toFixed(6)}<br>Lng: ${lng.toFixed(6)}`).openPopup();

            // Remove marker after 5 seconds
            setTimeout(() => {
                window.mapManager.map.removeLayer(marker);
            }, 5000);

            this.showNotification(`Moved to ${lat.toFixed(6)}, ${lng.toFixed(6)}`, 'success');
        }
    }

    onAOIChange(hasAOI) {
        // Enable/disable main search button
        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) {
            searchBtn.disabled = !hasAOI;
        }

        // Enable/disable change monitoring search (disabled)
        // const changeSearchBtn = document.getElementById('search-change-images-btn');
        // if (changeSearchBtn) {
        //     changeSearchBtn.disabled = !hasAOI;
        // }
        
        console.log(`AOI changed: hasAOI=${hasAOI}`);
    }

    initializeDateInputs() {
        const today = new Date();
        const oneMonthAgo = new Date(today);
        oneMonthAgo.setMonth(today.getMonth() - 1);

        const formatDate = (date) => date.toISOString().split('T')[0];

        // Main search dates (single date inputs for all satellites)
        const startDate = document.getElementById('start-date');
        const endDate = document.getElementById('end-date');
        if (startDate) startDate.value = formatDate(oneMonthAgo);
        if (endDate) endDate.value = formatDate(today);

        // Change monitoring dates
        const changeStartDate = document.getElementById('change-start-date');
        const changeEndDate = document.getElementById('change-end-date');
        if (changeStartDate) changeStartDate.value = formatDate(oneMonthAgo);
        if (changeEndDate) changeEndDate.value = formatDate(today);
        
        // Cloud cover slider value display
        const cloudCoverSlider = document.getElementById('cloud-cover');
        const cloudCoverValue = document.getElementById('cloud-cover-value');
        if (cloudCoverSlider && cloudCoverValue) {
            cloudCoverSlider.addEventListener('input', () => {
                cloudCoverValue.textContent = `${cloudCoverSlider.value}%`;
            });
        }
    }

    // ===== UI Delegation Methods =====

    showPanel(panelId) {
        if (this.uiManager) {
            this.uiManager.showPanel(panelId);
        }
    }

    hidePanel(panelId) {
        if (this.uiManager) {
            this.uiManager.hidePanel(panelId);
        }
    }

    showNotification(message, type = 'info', duration = 5000) {
        if (this.uiManager) {
            this.uiManager.showNotification(message, type, duration);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }

    showLoading(message = 'Processing...') {
        if (this.uiManager) {
            this.uiManager.showLoading(message);
        }
    }

    updateLoadingProgress(percent, note = '') {
        if (this.uiManager) {
            this.uiManager.updateLoadingProgress(percent, note);
        }
    }

    hideLoading() {
        if (this.uiManager) {
            this.uiManager.hideLoading();
        }
    }

    switchToAnalysisTab() {
        if (this.uiManager) {
            this.uiManager.switchToAnalysisTab();
        }
    }

    switchToChangeMonitoringTab() {
        if (this.uiManager) {
            this.uiManager.switchToChangeMonitoringTab();
        }
    }

    // ===== Search Delegation =====

    async searchImages() {
        if (this.imageSearch) {
            // Set the current satellite in the image search controller
            this.imageSearch.currentSatellite = this.currentSatelliteType;
            await this.imageSearch.searchImages();
            
            // Sync image data back to platform controller
            this.s1Images = this.imageSearch.s1Images || [];
            this.s2Images = this.imageSearch.s2Images || [];
            this.currentImages = this.currentSatelliteType === 's1' ? this.s1Images : this.s2Images;
            return;
        }
        console.warn('ImageSearchController not available');
    }

    showSearchResults(images, satellite = null) {
        if (this.searchResultsController) {
            this.searchResultsController.showSearchResults(images, satellite);
        }
    }

    // ===== Image Processing Delegation =====

    async processImage(imageId) {
        if (this.imageProcessorController) {
            return this.imageProcessorController.processImage(imageId);
        }
        console.warn('ImageProcessorController not available');
    }

    showAnalysisResults(results) {
        if (this.imageProcessorController) {
            this.imageProcessorController.showAnalysisResults(results);
        }
    }

    // ===== Image Selection =====

    async selectImage(imageId) {
        // Toggle logic
        if (this.selectedImageId === imageId) {
            this.selectedImageId = null;
            document.querySelectorAll('.image-item').forEach(item => {
                item.classList.remove('selected');
            });
            if (window.mapManager) {
                window.mapManager.clearImageLayers();
            }
            this.showNotification('Image hidden from map', 'info');
            return;
        }

        this.selectedImageId = imageId;

        // Update UI
        document.querySelectorAll('.image-item').forEach(item => {
            item.classList.remove('selected');
        });
        const selectedItem = document.querySelector(`[data-image-id="${imageId}"]`);
        if (selectedItem) {
            selectedItem.classList.add('selected');
        }

        // Clear existing layers
        if (window.mapManager) {
            window.mapManager.clearImageLayers();
        }

        // Find image data
        let image = this.currentImages.find(img => img.id === imageId) ||
                    this.s1Images.find(img => img.id === imageId) ||
                    this.s2Images.find(img => img.id === imageId);

        if (!image) {
            console.error(`Image ${imageId} not found`);
            return;
        }

        this.showNotification('Preparing image for display...', 'info');

        try {
            const isS1 = this.currentSatelliteType === 's1' || 
                         image.collection === 'Sentinel-1' ||
                         imageId.includes('S1_GRD');
            
            const apiEndpoint = isS1 ? '/api/get-s1-tile' : '/api/get-gee-tile';
            
            const requestBody = {
                item_id: imageId,
                bbox: window.mapManager.getCurrentBounds(),
                geometry: window.mapManager.getCurrentGeoJSON()?.geometry
            };
            
            if (isS1) {
                const visParams = this.getS1VisualizationParams();
                requestBody.bands = visParams.bands;
                requestBody.min = visParams.min;
                requestBody.max = visParams.max;
            }

            const response = await fetch(`${this.apiBaseUrl}${apiEndpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (response.ok) {
                const tileData = await response.json();
                image.tile_template = tileData.tile_template;
                image.tile_bounds = tileData.bounds;
                image.display_type = 'tile';

                if (window.mapManager) {
                    window.mapManager.addImageLayer(imageId, image);
                    window.mapManager.outlineAOI();
                    this.showNotification('Image displayed on map.', 'success');
                }
            } else {
                this.showNotification('Unable to display image', 'error');
            }

        } catch (error) {
            console.error('Error displaying image:', error);
            this.showNotification('Error displaying image', 'error');
        }
    }

    // ===== Progress Polling =====

    async pollProgress(jobId) {
        console.log(`[POLLING] Starting progress polling for job: ${jobId}`);
        try {
            let pollCount = 0;
            while (!this._progressStop) {
                pollCount++;
                const url = `/api/progress?job_id=${encodeURIComponent(jobId)}`;

                const r = await fetch(url);
                if (r.ok) {
                    const p = await r.json();
                    if (p.percent !== undefined) {
                        this.updateLoadingProgress(p.percent, p.message || '');
                    }

                    if ((p.status && (p.status === 'done' || p.status === 'completed')) || (p.percent || 0) >= 100) {
                        this._progressStop = true;
                        setTimeout(() => this.hideLoading(), 500);
                        break;
                    }
                }

                if (pollCount > 300) {
                    this._progressStop = true;
                    this.hideLoading();
                    break;
                }

                await new Promise(res => setTimeout(res, 300));
            }
        } catch (e) {
            console.error('[POLLING] Error:', e);
            this._progressStop = true;
            this.hideLoading();
        }
    }

    // ===== Visualization Parameters =====

    applyS1Preset(preset) {
        const rSelect = document.getElementById('s1-channel-r');
        const gSelect = document.getElementById('s1-channel-g');
        const bSelect = document.getElementById('s1-channel-b');
        const minInput = document.getElementById('s1-min');
        const maxInput = document.getElementById('s1-max');

        switch (preset) {
            case 'vv':
                if (rSelect) rSelect.value = 'VV';
                if (gSelect) gSelect.value = 'VV';
                if (bSelect) bSelect.value = 'VV';
                if (minInput) minInput.value = -25;
                if (maxInput) maxInput.value = 0;
                break;
            case 'vh':
                if (rSelect) rSelect.value = 'VH';
                if (gSelect) gSelect.value = 'VH';
                if (bSelect) bSelect.value = 'VH';
                if (minInput) minInput.value = -30;
                if (maxInput) maxInput.value = -5;
                break;
            case 'vvvh':
                if (rSelect) rSelect.value = 'VV';
                if (gSelect) gSelect.value = 'VH';
                if (bSelect) bSelect.value = 'VV';
                if (minInput) minInput.value = -25;
                if (maxInput) maxInput.value = 0;
                break;
            case 'ratio':
                if (rSelect) rSelect.value = 'VH';
                if (gSelect) gSelect.value = 'VV';
                if (bSelect) bSelect.value = 'angle';
                if (minInput) minInput.value = -25;
                if (maxInput) maxInput.value = 0;
                break;
        }

        this.s1Channels = { 
            r: rSelect?.value || 'VV', 
            g: gSelect?.value || 'VH', 
            b: bSelect?.value || 'VV' 
        };
        this.s1MinMax = { 
            min: parseFloat(minInput?.value) || -25, 
            max: parseFloat(maxInput?.value) || 0 
        };
    }

    applyS2Preset(preset) {
        const presets = {
            'truecolor': { r: 'B4', g: 'B3', b: 'B2', min: 0, max: 3000 },
            'falsecolor': { r: 'B8', g: 'B4', b: 'B3', min: 0, max: 5000 },
            'agriculture': { r: 'B11', g: 'B8', b: 'B2', min: 0, max: 5000 },
            'geology': { r: 'B12', g: 'B11', b: 'B2', min: 0, max: 5000 }
        };

        const settings = presets[preset];
        if (!settings) return;

        const rSelect = document.getElementById('s2-channel-r');
        const gSelect = document.getElementById('s2-channel-g');
        const bSelect = document.getElementById('s2-channel-b');
        const minInput = document.getElementById('s2-min');
        const maxInput = document.getElementById('s2-max');

        if (rSelect) rSelect.value = settings.r;
        if (gSelect) gSelect.value = settings.g;
        if (bSelect) bSelect.value = settings.b;
        if (minInput) minInput.value = settings.min;
        if (maxInput) maxInput.value = settings.max;
    }

    getS2VisualizationParams() {
        const rSlot = document.querySelector('#s2-slot-r .slot-band');
        const gSlot = document.querySelector('#s2-slot-g .slot-band');
        const bSlot = document.querySelector('#s2-slot-b .slot-band');
        const r = rSlot?.dataset.band || 'B4';
        const g = gSlot?.dataset.band || 'B3';
        const b = bSlot?.dataset.band || 'B2';
        const min = parseFloat(document.getElementById('s2-min')?.value) || 0;
        const max = parseFloat(document.getElementById('s2-max')?.value) || 3000;
        return { bands: [r, g, b], min, max };
    }

    getS1VisualizationParams() {
        const rSlot = document.querySelector('#s1-slot-r .slot-band');
        const gSlot = document.querySelector('#s1-slot-g .slot-band');
        const bSlot = document.querySelector('#s1-slot-b .slot-band');
        const r = rSlot?.dataset.band || 'VV';
        const g = gSlot?.dataset.band || 'VH';
        const b = bSlot?.dataset.band || 'VV';
        const min = parseFloat(document.getElementById('s1-min')?.value) || -25;
        const max = parseFloat(document.getElementById('s1-max')?.value) || 0;
        return { bands: [r, g, b], min, max };
    }

    // ===== Utility Methods =====

    getAOI() {
        if (!window.mapManager) return null;
        
        const bbox = window.mapManager.getCurrentBounds();
        const geometry = window.mapManager.getCurrentGeoJSON()?.geometry;
        
        if (!bbox) return null;
        
        return { bbox, geometry };
    }

    getSafeBounds() {
        if (window.mapManager && window.mapManager.getCurrentBounds()) {
            return window.mapManager.getCurrentBounds();
        }
        return null;
    }

    // ===== Custom Visualization Modal =====

    showCustomVisualizationModal() {
        if (this.customViz) {
            this.customViz.showModal();
        }
    }

    // ===== Threshold Control =====

    async toggleThresholdControl(modelId, result, btn) {
        if (this.thresholdController) {
            return this.thresholdController.toggleThresholdControl(modelId, result, btn);
        }
    }

    // ===== Pixel Value Inspection =====

    togglePixelValueInspection(modelId, result, btn) {
        if (this.thresholdController) {
            return this.thresholdController.togglePixelInspection(modelId, result, btn);
        }
    }

    // ===== Change Monitoring =====

    startChangeMonitoring(modelId, result, btnElement) {
        // Change monitoring disabled
        // if (this.changeMonitoring) {
        //     this.switchToChangeMonitoringTab();
        //     this.showNotification('Switch to Change Monitoring tab to continue', 'info');
        // }
    }

    // ===== Download Methods =====

    setupDownloadDropdownClose() {
        if (this.searchResultsController) {
            this.searchResultsController.setupDownloadDropdownClose();
        } else {
            // Fallback: close dropdowns when clicking elsewhere
            document.addEventListener('click', () => {
                document.querySelectorAll('.download-options').forEach(opt => {
                    opt.classList.add('hidden');
                });
            });
        }
    }

    async downloadSatelliteImage(image, downloadType = 'original') {
        if (this.searchResultsController) {
            return this.searchResultsController.downloadSatelliteImage(image, downloadType);
        }
        
        // Fallback implementation
        const isS1 = image.collection === 'Sentinel-1';
        const satelliteName = isS1 ? 'Sentinel-1' : 'Sentinel-2';
        const isVisualization = downloadType === 'visualization';
        const typeLabel = isVisualization ? 'visualization' : 'raw';
        
        this.showLoading(`Downloading ${satelliteName} ${typeLabel} image...`);
        
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
}

// Global instance will be created from index.html
