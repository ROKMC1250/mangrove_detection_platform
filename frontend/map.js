/**
 * ===== MAP MANAGER =====
 * Google Maps Style Map Controller
 */

class MapManager {
    constructor(mapId) {
        this.mapId = mapId;
        this.map = null;
        this.drawnItems = null;
        this.currentAOI = null;
        this.imageLayers = {};
        this.selectedImageLayer = null;
        this.onAOIChange = null; // Callback for AOI changes
        this.baseLayer = null; // OSM base layer reference
        this.processedLayers = {}; // Track processed image layers
        this.showProcessedImages = true; // Toggle for processed images
        this.analysisLayers = {}; // Track analysis model layers
        this.cogCache = {}; // url -> georaster
        this.modelLayerCache = {}; // modelId -> GeoRasterLayer instance
        this.currentBaseLayerType = 'osm'; // Track current base layer: 'osm' or 'satellite'
        
        this.initializeMap();
    }

    initializeMap() {
        console.log('🗺️ Initializing Map Manager...');
        
        // Check if Leaflet is loaded
        if (typeof L === 'undefined') {
            console.error('❌ Leaflet is not loaded!');
            this.showInitializationError('Leaflet library failed to load');
            return;
        }
        
        // Check if Leaflet.draw is loaded with retry mechanism
        if (typeof L.Draw === 'undefined') {
            console.warn('⚠️ Leaflet.draw not loaded yet, retrying...');
            
            // Retry after a short delay
            setTimeout(() => {
                if (typeof L.Draw === 'undefined') {
                    console.error('❌ Leaflet.draw is not loaded after retry!');
                    this.showInitializationError('Leaflet.draw library failed to load');
                    return;
                } else {
                    console.log('✅ Leaflet.draw loaded after retry');
                    this.continueInitialization();
                }
            }, 1000);
            return;
        }
        
        console.log('✅ Leaflet and Leaflet.draw are loaded');
        this.continueInitialization();
    }

    continueInitialization() {
        // Check if map container already has a map instance
        const mapContainer = document.getElementById(this.mapId);
        if (!mapContainer) {
            console.error(`❌ Map container with ID '${this.mapId}' not found!`);
            this.showInitializationError('Map container not found');
            return;
        }

        // Remove existing map instance if it exists
        if (mapContainer._leaflet_id) {
            console.log('🧹 Cleaning up existing map instance...');
            // Remove the Leaflet instance from the container
            mapContainer._leaflet_id = null;
            mapContainer.innerHTML = '';
        }
        
        // Initialize map centered on Philippines (mangrove region)
        this.map = L.map(this.mapId, {
            center: [13.0, 122.0], // Philippines coordinates
            zoom: 8,
            zoomControl: true,
            attributionControl: true
        });

        // Create custom panes for proper layer ordering
        // Pane z-index order: basemapPane (100) < tilePane (200) < dataPane (450) < aoiPane (500) < labelsPane (650)
        
        // Basemap pane - LOWEST z-index for base maps (OSM/Satellite)
        this.map.createPane('basemapPane');
        this.map.getPane('basemapPane').style.zIndex = 100;  // Below everything
        
        // Data pane for analysis results (satellite tiles, NDVI overlays, etc.)
        this.map.createPane('dataPane');
        this.map.getPane('dataPane').style.zIndex = 450;  // Above tilePane (200)
        
        // AOI pane for drawn items and KML polygons - ALWAYS on top of data
        this.map.createPane('aoiPane');
        this.map.getPane('aoiPane').style.zIndex = 500;  // Above dataPane (450)
        
        // Labels pane for map labels
        this.map.createPane('labelsPane');
        this.map.getPane('labelsPane').style.zIndex = 650;  // Above everything except popups
        
        console.log('✅ Custom panes created: basemapPane (100), dataPane (450), aoiPane (500), labelsPane (650)');

        // Create base layers for layer control
        // IMPORTANT: Base layers must have the LOWEST z-index (negative values) to always stay below data layers
        
        // Map base layer with English labels (using CARTO Voyager)
        this.osmLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap contributors, © CARTO',
            maxZoom: 19,
            subdomains: 'abcd',
            pane: 'basemapPane'  // Use custom basemap pane (z-index 100) - always at bottom
        });
        
        // Satellite base layer
        this.satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '© Esri, Maxar, Earthstar Geographics',
            maxZoom: 19,
            pane: 'basemapPane'  // Use custom basemap pane (z-index 100) - always at bottom
        });

        // Combined labels and borders overlay for Satellite
        // Using CARTO Positron labels which includes both place names and roads/borders
        // This is transparent and shows only annotations (no background map)
        this.labelsLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap contributors, © CARTO',
            maxZoom: 19,
            subdomains: 'abcd',
            pane: 'labelsPane'  // Use custom labels pane (z-index 650)
        });

        // Add default layer (OSM) - ensure only OSM is active initially
        this.baseLayer = this.osmLayer;
        this.osmLayer.addTo(this.map);
        this.currentBaseLayerType = 'osm'; // Set initial state
        console.log('🗺️ Default layer (OSM) added to map');
        
        // Ensure satellite and labels are NOT added initially
        if (this.map.hasLayer(this.satelliteLayer)) {
            this.map.removeLayer(this.satelliteLayer);
            console.log('🧹 Removed satellite layer (should not be active initially)');
        }
        if (this.map.hasLayer(this.labelsLayer)) {
            this.map.removeLayer(this.labelsLayer);
            console.log('🧹 Removed labels layer (should not be active initially)');
        }

        // Add custom layer control
        this.addCustomLayerControl();
        
        // Debug: Log initial layer state
        this.debugLayerState();

        // Add coordinate display control
        this.addCoordinateDisplay();

        // Initialize drawing layer
        this.drawnItems = new L.FeatureGroup();
        this.map.addLayer(this.drawnItems);

        // Setup drawing controls
        this.setupDrawingControls();
        
        // Setup map event listeners
        this.setupMapEvents();

        console.log('✅ Map Manager Ready!');
    }

    addCoordinateDisplay() {
        // Create coordinate display control
        const coordControl = L.control({position: 'bottomleft'});
        
        coordControl.onAdd = () => {
            const div = L.DomUtil.create('div', 'coordinate-display');
            div.innerHTML = '<div class="coordinate-content">Lat: <span id="coord-lat">--</span>, Lng: <span id="coord-lng">--</span></div>';
            div.style.cssText = `
                background: rgba(255, 255, 255, 0.95);
                padding: 8px 12px;
                border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 13px;
                font-weight: 500;
                color: #333;
                border: 2px solid #4CAF50;
                min-width: 220px;
            `;
            return div;
        };
        
        coordControl.addTo(this.map);
        
        // Update coordinates on mouse move
        this.map.on('mousemove', (e) => {
            const lat = e.latlng.lat.toFixed(6);
            const lng = e.latlng.lng.toFixed(6);
            document.getElementById('coord-lat').textContent = lat;
            document.getElementById('coord-lng').textContent = lng;
        });
        
        // Clear coordinates when mouse leaves map
        this.map.on('mouseout', () => {
            document.getElementById('coord-lat').textContent = '--';
            document.getElementById('coord-lng').textContent = '--';
        });
        
        console.log('✅ Coordinate display added');
    }

    addCustomLayerControl() {
        // Prevent duplicate controls
        if (this.layerControl) {
            console.log('⚠️ Layer control already exists, skipping');
            return;
        }
        
        console.log('🎛️ Creating custom layer control...');
        
        // Create custom layer control
        this.layerControl = L.control({position: 'bottomright'});
        
        this.layerControl.onAdd = (map) => {
            const div = L.DomUtil.create('div', 'custom-layer-control');
            div.innerHTML = `
                <div class="layer-control-container">
                    <div class="layer-option" data-layer="osm">
                        <input type="radio" id="layer-osm" name="base-layer" checked>
                        <label for="layer-osm">
                            <span class="layer-icon">🗺️</span>
                            <span class="layer-name">Map</span>
                        </label>
                    </div>
                    <div class="layer-option" data-layer="satellite">
                        <input type="radio" id="layer-satellite" name="base-layer">
                        <label for="layer-satellite">
                            <span class="layer-icon">🛰️</span>
                            <span class="layer-name">Satellite</span>
                        </label>
                    </div>
                </div>
            `;
            
            // Prevent map interactions when clicking on control
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            
            // Add event listeners to layer options (divs) instead of radio buttons
            const osmOption = div.querySelector('.layer-option[data-layer="osm"]');
            const satelliteOption = div.querySelector('.layer-option[data-layer="satellite"]');
            const osmRadio = div.querySelector('#layer-osm');
            const satelliteRadio = div.querySelector('#layer-satellite');
            
            if (osmOption && osmRadio) {
                L.DomEvent.on(osmOption, 'click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🗺️ Map option clicked');
                    
                    // Manually set radio state
                    osmRadio.checked = true;
                    satelliteRadio.checked = false;
                    
                    // Switch layer
                    this.switchToOSM();
                }, this);
                console.log('  ✓ OSM click event listener attached');
            }
            
            if (satelliteOption && satelliteRadio) {
                L.DomEvent.on(satelliteOption, 'click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🛰️ Satellite option clicked');
                    
                    // Manually set radio state
                    satelliteRadio.checked = true;
                    osmRadio.checked = false;
                    
                    // Switch layer
                    this.switchToSatellite();
                }, this);
                console.log('  ✓ Satellite click event listener attached');
            }
            
            return div;
        };
        
        this.layerControl.addTo(this.map);
        console.log('✅ Custom layer control added to map');
    }

    debugLayerState() {
        console.log('🔍 Current layer state:');
        console.log('  Current base layer type:', this.currentBaseLayerType);
        console.log('  OSM layer:', this.map.hasLayer(this.osmLayer) ? '✅ Active' : '❌ Inactive');
        console.log('  Satellite layer:', this.map.hasLayer(this.satelliteLayer) ? '✅ Active' : '❌ Inactive');
        console.log('  Labels layer:', this.map.hasLayer(this.labelsLayer) ? '✅ Active' : '❌ Inactive');
    }

    switchToOSM() {
        // Check if already on OSM
        if (this.currentBaseLayerType === 'osm') {
            console.log('ℹ️ Already on OSM layer, skipping switch');
            return;
        }
        
        console.log('🔄 Switching to OSM layer...');
        
        // Remove satellite and labels overlay
        if (this.map.hasLayer(this.satelliteLayer)) {
            this.map.removeLayer(this.satelliteLayer);
            console.log('  ✓ Removed satellite layer');
        }
        if (this.map.hasLayer(this.labelsLayer)) {
            this.map.removeLayer(this.labelsLayer);
            console.log('  ✓ Removed labels layer');
        }
        
        // Add OSM (already includes English labels and roads)
        if (!this.map.hasLayer(this.osmLayer)) {
            this.osmLayer.addTo(this.map);
            console.log('  ✓ Added OSM layer');
        } else {
            console.log('  ℹ OSM layer already present');
        }
        
        this.baseLayer = this.osmLayer;
        this.currentBaseLayerType = 'osm';
        console.log('✅ Switched to OSM layer');
        this.debugLayerState();
        
        // Bring data layers to front after base map switch
        this.bringDataLayersToFront();
    }

    switchToSatellite() {
        // Check if already on Satellite
        if (this.currentBaseLayerType === 'satellite') {
            console.log('ℹ️ Already on Satellite layer, skipping switch');
            return;
        }
        
        console.log('🔄 Switching to Satellite layer...');
        console.log('  Before switch - checking layers:');
        console.log('    OSM:', this.map.hasLayer(this.osmLayer));
        console.log('    Satellite:', this.map.hasLayer(this.satelliteLayer));
        console.log('    Labels:', this.map.hasLayer(this.labelsLayer));
        
        // CRITICAL: Remove OSM FIRST before adding satellite
        if (this.map.hasLayer(this.osmLayer)) {
            console.log('  🧹 Removing OSM layer...');
            this.map.removeLayer(this.osmLayer);
            console.log('  ✓ OSM layer removed');
            
            // Double check it was removed
            if (this.map.hasLayer(this.osmLayer)) {
                console.error('  ❌ ERROR: OSM layer is STILL on map after removal!');
            } else {
                console.log('  ✅ Verified: OSM layer removed from map');
            }
        } else {
            console.log('  ℹ OSM layer was not active');
        }
        
        // Add satellite base
        if (!this.map.hasLayer(this.satelliteLayer)) {
            console.log('  🛰️ Adding satellite layer...');
            this.satelliteLayer.addTo(this.map);
            console.log('  ✓ Satellite layer added');
            
            // Verify it was added
            if (this.map.hasLayer(this.satelliteLayer)) {
                console.log('  ✅ Verified: Satellite layer is now on map');
            } else {
                console.error('  ❌ ERROR: Satellite layer was NOT added to map!');
            }
        } else {
            console.log('  ℹ Satellite layer already present');
        }
        
        // Add labels overlay (includes place names, roads, and borders - transparent background)
        if (!this.map.hasLayer(this.labelsLayer)) {
            console.log('  🏷️ Adding labels overlay (includes borders and place names)...');
            this.labelsLayer.addTo(this.map);
            console.log('  ✓ Labels layer added');
            
            // Verify it was added
            if (this.map.hasLayer(this.labelsLayer)) {
                console.log('  ✅ Verified: Labels layer is now on map');
            } else {
                console.error('  ❌ ERROR: Labels layer was NOT added to map!');
            }
        } else {
            console.log('  ℹ Labels layer already present');
        }
        
        this.baseLayer = this.satelliteLayer;
        this.currentBaseLayerType = 'satellite';
        console.log('✅ Switched to Satellite layer with English labels and borders');
        
        // Bring data layers to front after base map switch
        this.bringDataLayersToFront();
        
        // Final verification with delay to ensure rendering
        setTimeout(() => {
            console.log('  🔍 Final verification (after 500ms):');
            this.debugLayerState();
            
            // List all layers on map
            console.log('  📋 All layers currently on map:');
            this.map.eachLayer((layer) => {
                if (layer === this.osmLayer) console.log('    - OSM Layer (SHOULD NOT BE HERE!)');
                else if (layer === this.satelliteLayer) console.log('    - Satellite Layer ✅');
                else if (layer === this.labelsLayer) console.log('    - Labels Layer (with borders) ✅');
                else if (layer === this.drawnItems) console.log('    - Drawn Items Layer');
                else console.log('    - Other layer:', layer);
            });
        }, 500);
    }
    
    // Bring all data layers to front after base map switch
    bringDataLayersToFront() {
        console.log('🔝 Bringing data layers to front...');
        
        // First, ensure base layer is at the bottom by sending it to back
        if (this.baseLayer && this.map.hasLayer(this.baseLayer)) {
            this.baseLayer.bringToBack();
        }
        
        // Bring image layers to front (in dataPane)
        Object.keys(this.imageLayers).forEach(id => {
            const layer = this.imageLayers[id];
            if (this.map.hasLayer(layer)) {
                layer.bringToFront();
            }
        });
        
        // Bring processed layers to front
        Object.keys(this.processedLayers).forEach(id => {
            const layer = this.processedLayers[id];
            if (this.map.hasLayer(layer)) {
                layer.bringToFront();
            }
        });
        
        // Bring analysis layers to front
        Object.keys(this.analysisLayers).forEach(id => {
            const layer = this.analysisLayers[id];
            if (this.map.hasLayer(layer)) {
                layer.bringToFront();
            }
        });
        
        // Bring tile layers to front
        if (this.tileLayers) {
            Object.keys(this.tileLayers).forEach(id => {
                const layer = this.tileLayers[id];
                if (this.map.hasLayer(layer)) {
                    layer.bringToFront();
                }
            });
        }
        
        // Bring drawn items (AOI polygons) to front last - they should always be on top
        if (this.drawnItems && this.map.hasLayer(this.drawnItems)) {
            this.drawnItems.bringToFront();
        }
        
        // Also bring individual AOI polygons to front
        if (this.aoiPolygons && this.aoiPolygons.length > 0) {
            this.aoiPolygons.forEach(polygon => {
                if (this.map.hasLayer(polygon)) {
                    polygon.bringToFront();
                }
            });
        }
        
        console.log('✅ Data layers brought to front');
    }

    showInitializationError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            background: #ea4335;
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            z-index: 1001;
            font-weight: 500;
        `;
        errorDiv.textContent = `Map Initialization Error: ${message}. Please refresh the page.`;
        document.body.appendChild(errorDiv);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 10000);
    }

    setupDrawingControls() {
        // Create drawing control with proper configuration
        this.drawControl = new L.Control.Draw({
            position: 'topright',
            draw: {
                polygon: false,
                polyline: false,
                circle: false,
                marker: false,
                circlemarker: false,
                rectangle: {
                    showArea: true,
                    showLength: false,
                    shapeOptions: {
                        color: '#ff7800',
                        weight: 3,
                        opacity: 0.8,
                        fillColor: '#ff7800',
                        fillOpacity: 0.1,
                        pane: 'aoiPane'  // Always on top of data layers
                    },
                    metric: true, // Use metric units
                    feet: false,
                    nautic: false
                }
            },
            edit: {
                featureGroup: this.drawnItems,
                remove: true,
                edit: true
            }
        });

        // Don't add the control to map initially - we'll trigger it programmatically
        // this.map.addControl(this.drawControl);
        
        console.log('✅ Drawing controls configured');
    }

    setupMapEvents() {
        // AOI creation event
        this.map.on(L.Draw.Event.CREATED, (event) => {
            const layer = event.layer;
            
            console.log('🎯 Draw event triggered:', event.layerType);
            
            // IMPORTANT: Disable the drawer immediately after creation to prevent continuous drawing
            if (this.currentDrawer) {
                this.currentDrawer.disable();
                this.currentDrawer = null;
                console.log('✅ Drawer disabled after AOI creation');
            }
            
            // Clear previous AOI first
            this.drawnItems.clearLayers();
            
            // Set the layer to use aoiPane for proper z-ordering
            layer.options.pane = 'aoiPane';
            
            // Add new AOI
            this.drawnItems.addLayer(layer);
            this.currentAOI = layer;
            
            // Clear and reset aoiPolygons array (single rectangle = single polygon)
            this.aoiPolygons = [layer];
            
            // Get AOI bounds
            const bounds = layer.getBounds();
            console.log('✅ AOI Created:', bounds);
            
            // Trigger callback
            if (this.onAOIChange) {
                this.onAOIChange(true);
            }
            
            // Show legend
            const legend = document.getElementById('map-legend');
            if (legend) {
                legend.classList.remove('hidden');
            }
            
            // Re-enable drawing button (it gets disabled during drawing)
            const drawBtn = document.getElementById('draw-rectangle');
            if (drawBtn) {
                drawBtn.disabled = false;
                drawBtn.textContent = '📍 Draw AOI';
            }
        });

        // AOI editing events
        this.map.on(L.Draw.Event.EDITED, (event) => {
            if (this.currentAOI) {
                const bounds = this.currentAOI.getBounds();
                console.log('✏️ AOI Edited:', bounds);
                
                if (this.onAOIChange) {
                    this.onAOIChange(true);
                }
            }
        });

        // AOI deletion event
        this.map.on(L.Draw.Event.DELETED, (event) => {
            this.currentAOI = null;
            console.log('🗑️ AOI Deleted');
            
            if (this.onAOIChange) {
                this.onAOIChange(false);
            }
            
            // Hide legend
            const legend = document.getElementById('map-legend');
            if (legend) {
                legend.classList.add('hidden');
            }
        });

        // Drawing start event
        this.map.on(L.Draw.Event.DRAWSTART, (event) => {
            console.log('🎨 Drawing started');
            const drawBtn = document.getElementById('draw-rectangle');
            if (drawBtn) {
                drawBtn.disabled = true;
                drawBtn.textContent = '✏️ Drawing...';
            }
        });

        // Drawing stop event  
        this.map.on(L.Draw.Event.DRAWSTOP, (event) => {
            console.log('🛑 Drawing stopped');
            
            // Clean up the drawer reference
            if (this.currentDrawer) {
                this.currentDrawer.disable();
                this.currentDrawer = null;
            }
            
            const drawBtn = document.getElementById('draw-rectangle');
            if (drawBtn) {
                drawBtn.disabled = false;
                drawBtn.textContent = '📍 Draw AOI';
            }
        });
    }

    startDrawing() {
        console.log('🎯 Starting AOI drawing mode...');
        
        try {
            // Clear any existing drawings
            this.drawnItems.clearLayers();
            this.currentAOI = null;
            
            // Disable previous drawer if exists
            if (this.currentDrawer) {
                this.currentDrawer.disable();
                this.currentDrawer = null;
            }
            
            // Create a new rectangle drawing handler manually
            const rectangleDrawer = new L.Draw.Rectangle(this.map, {
                shapeOptions: {
                    color: '#ff7800',
                    weight: 3,
                    opacity: 0.8,
                    fillColor: '#ff7800',
                    fillOpacity: 0.1,
                    pane: 'aoiPane'  // Always on top of data layers
                },
                showArea: true,
                metric: true,
                showLength: false,
                allowIntersection: false,
                repeatMode: false,  // IMPORTANT: Disable repeat mode to stop after one draw
                drawError: {
                    color: '#b00b00',
                    timeout: 1000
                }
            });
            
            // Store reference to current drawer for cleanup
            this.currentDrawer = rectangleDrawer;
            
            // Enable the drawing handler
            rectangleDrawer.enable();
            
            console.log('✅ AOI drawing mode enabled - click and drag on the map');
            
            // Update button state
            const drawBtn = document.getElementById('draw-rectangle');
            if (drawBtn) {
                drawBtn.disabled = true;
                drawBtn.textContent = '✏️ Drawing...';
            }
            
            return rectangleDrawer;
            
        } catch (error) {
            console.error('❌ Error starting AOI drawing:', error);
            
            // Reset button state on error
            const drawBtn = document.getElementById('draw-rectangle');
            if (drawBtn) {
                drawBtn.disabled = false;
                drawBtn.textContent = '📍 Draw AOI';
            }
            
            throw error;
        }
    }

    clearDrawings() {
        this.drawnItems.clearLayers();
        this.currentAOI = null;
        
        // Clear all image layers
        this.clearImageLayers();
        
        if (this.onAOIChange) {
            this.onAOIChange(false);
        }
        
        // Hide legend
        const mapLegend = document.getElementById('map-legend');
        if (mapLegend) {
            mapLegend.classList.add('hidden');
        }
        
        console.log('🧹 All drawings cleared');
    }

    getCurrentBounds() {
        // If we have multiple polygons, calculate combined bounds
        if (this.aoiPolygons && this.aoiPolygons.length > 0) {
            const combinedBounds = L.latLngBounds();
            this.aoiPolygons.forEach(layer => {
                combinedBounds.extend(layer.getBounds());
            });
            return [
                combinedBounds.getWest(),  // min_lon
                combinedBounds.getSouth(), // min_lat
                combinedBounds.getEast(),  // max_lon
                combinedBounds.getNorth()  // max_lat
            ];
        }
        
        if (this.currentAOI) {
            const bounds = this.currentAOI.getBounds();
            return [
                bounds.getWest(),  // min_lon
                bounds.getSouth(), // min_lat
                bounds.getEast(),  // max_lon
                bounds.getNorth()  // max_lat
            ];
        }
        return null;
    }

    // Get combined bounds of all AOI geometries (polygons, lines, points) as L.LatLngBounds
    getCombinedAOIBounds() {
        if (this.aoiPolygons && this.aoiPolygons.length > 0) {
            const combinedBounds = L.latLngBounds();
            this.aoiPolygons.forEach(layer => {
                if (layer.getBounds) {
                    // Polygon, Rectangle, Polyline
                    combinedBounds.extend(layer.getBounds());
                } else if (layer.getLatLng) {
                    // Point (CircleMarker, Marker)
                    combinedBounds.extend(layer.getLatLng());
                }
            });
            return combinedBounds;
        }
        
        if (this.currentAOI) {
            if (this.currentAOI.getBounds) {
                return this.currentAOI.getBounds();
            } else if (this.currentAOI.getLatLng) {
                // For single point, create a small bounds around it
                const latlng = this.currentAOI.getLatLng();
                const offset = 0.001; // ~100m
                return L.latLngBounds(
                    [latlng.lat - offset, latlng.lng - offset],
                    [latlng.lat + offset, latlng.lng + offset]
                );
            }
        }
        return null;
    }

    getCurrentGeoJSON() {
        // Lines are now converted to Polygons, so we mainly deal with Polygons and Points
        if (this.aoiPolygons && this.aoiPolygons.length > 0) {
            // Categorize geometries
            const polygonCoords = [];
            const pointCoords = [];
            
            this.aoiPolygons.forEach(layer => {
                const geojson = layer.toGeoJSON();
                const type = geojson.geometry?.type;
                
                if (type === 'Polygon') {
                    polygonCoords.push(geojson.geometry.coordinates);
                } else if (type === 'MultiPolygon') {
                    polygonCoords.push(...geojson.geometry.coordinates);
                } else if (type === 'Point') {
                    pointCoords.push(geojson.geometry.coordinates);
                } else if (type === 'MultiPoint') {
                    pointCoords.push(...geojson.geometry.coordinates);
                }
            });
            
            const hasPolygons = polygonCoords.length > 0;
            const hasPoints = pointCoords.length > 0;
            
            // If only polygons exist, return MultiPolygon
            if (hasPolygons && !hasPoints) {
                console.log(`Creating MultiPolygon with ${polygonCoords.length} polygon(s)`);
                return {
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'MultiPolygon',
                        coordinates: polygonCoords
                    }
                };
            }
            
            // If only points exist, return MultiPoint
            if (hasPoints && !hasPolygons) {
                console.log(`Creating MultiPoint with ${pointCoords.length} point(s)`);
                return {
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'MultiPoint',
                        coordinates: pointCoords
                    }
                };
            }
            
            // Mixed types (polygons + points): return GeometryCollection
            if (hasPolygons && hasPoints) {
                console.log(`Creating GeometryCollection with ${polygonCoords.length} polygon(s), ${pointCoords.length} point(s)`);
                return {
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'GeometryCollection',
                        geometries: [
                            { type: 'MultiPolygon', coordinates: polygonCoords },
                            { type: 'MultiPoint', coordinates: pointCoords }
                        ]
                    }
                };
            }
        }
        
        if (this.currentAOI) {
            return this.currentAOI.toGeoJSON();
        }
        return null;
    }

    async loadShpFile(file) {
        console.log('📂 Loading SHP file:', file.name);
        
        try {
            const arrayBuffer = await file.arrayBuffer();
            let geojson = await shp(arrayBuffer);
            
            console.log('✅ SHP parsed:', geojson);
            
            // Handle FeatureCollection or single Feature
            if (Array.isArray(geojson)) {
                // shpjs returns array if multiple files in zip
                geojson = geojson[0];
            }
            
            // Clear existing drawings
            this.drawnItems.clearLayers();
            
            // Style for AOI polygons - use aoiPane to always be on top
            const aoiStyle = {
                color: '#ff7800',
                weight: 3,
                opacity: 0.8,
                fillColor: '#ff7800',
                fillOpacity: 0.1,
                pane: 'aoiPane'  // Always on top of data layers
            };
            
            // Style for points
            const pointStyle = {
                radius: 8,
                color: '#ff7800',
                weight: 2,
                opacity: 0.8,
                fillColor: '#ff7800',
                fillOpacity: 0.5,
                pane: 'aoiPane'
            };
            
            // Style for lines
            const lineStyle = {
                color: '#ff7800',
                weight: 3,
                opacity: 0.8,
                pane: 'aoiPane'
            };
            
            // Collect ALL layers (polygons, lines converted to polygons, points)
            const createdLayers = [];
            const polygonLayers = [];
            const lineLayers = [];
            const pointLayers = [];
            
            // Process each feature and convert lines to polygons
            if (geojson.features) {
                geojson.features.forEach(feature => {
                    const type = feature.geometry?.type;
                    
                    if (type === 'Polygon' || type === 'MultiPolygon') {
                        const layer = L.geoJSON(feature, { style: aoiStyle });
                        layer.eachLayer(l => {
                            l.options.pane = 'aoiPane';
                            createdLayers.push(l);
                            polygonLayers.push(l);
                            this.drawnItems.addLayer(l);
                        });
                    } else if (type === 'LineString') {
                        // Convert LineString to Polygon (fill the area)
                        console.log('Converting LineString to Polygon');
                        const coords = feature.geometry.coordinates.map(c => [c[1], c[0]]); // [lng,lat] -> [lat,lng]
                        const polygon = L.polygon(coords, aoiStyle);
                        polygon.options.pane = 'aoiPane';
                        createdLayers.push(polygon);
                        polygonLayers.push(polygon);
                        this.drawnItems.addLayer(polygon);
                    } else if (type === 'MultiLineString') {
                        // Convert each line in MultiLineString to Polygon
                        console.log('Converting MultiLineString to Polygons');
                        feature.geometry.coordinates.forEach(lineCoords => {
                            const coords = lineCoords.map(c => [c[1], c[0]]); // [lng,lat] -> [lat,lng]
                            const polygon = L.polygon(coords, aoiStyle);
                            polygon.options.pane = 'aoiPane';
                            createdLayers.push(polygon);
                            polygonLayers.push(polygon);
                            this.drawnItems.addLayer(polygon);
                        });
                    } else if (type === 'Point') {
                        const latlng = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
                        const marker = L.circleMarker(latlng, pointStyle);
                        marker.options.pane = 'aoiPane';
                        createdLayers.push(marker);
                        pointLayers.push(marker);
                        this.drawnItems.addLayer(marker);
                    } else if (type === 'MultiPoint') {
                        feature.geometry.coordinates.forEach(coord => {
                            const latlng = [coord[1], coord[0]];
                            const marker = L.circleMarker(latlng, pointStyle);
                            marker.options.pane = 'aoiPane';
                            createdLayers.push(marker);
                            pointLayers.push(marker);
                            this.drawnItems.addLayer(marker);
                        });
                    }
                });
            } else {
                // Single geometry (not FeatureCollection)
                const geoJsonLayer = L.geoJSON(geojson, { 
                    style: aoiStyle,
                    pointToLayer: (feature, latlng) => {
                        return L.circleMarker(latlng, pointStyle);
                    }
                });
                
                geoJsonLayer.eachLayer(l => {
                    l.options.pane = 'aoiPane';
                    createdLayers.push(l);
                    this.drawnItems.addLayer(l);
                    
                    if (l instanceof L.Polygon || l instanceof L.Rectangle) {
                        polygonLayers.push(l);
                    } else if (l instanceof L.Polyline) {
                        // Convert polyline to polygon
                        const coords = l.getLatLngs();
                        const polygon = L.polygon(coords, aoiStyle);
                        polygon.options.pane = 'aoiPane';
                        polygonLayers.push(polygon);
                    } else if (l instanceof L.CircleMarker || l instanceof L.Marker) {
                        pointLayers.push(l);
                    }
                });
            }
            
            if (createdLayers.length > 0) {
                // Store all layers for multi-geometry support
                this.aoiPolygons = createdLayers;
                this.currentAOI = createdLayers[0];
                
                // Calculate combined bounds
                const combinedBounds = L.latLngBounds();
                createdLayers.forEach(layer => {
                    if (layer.getBounds) {
                        combinedBounds.extend(layer.getBounds());
                    } else if (layer.getLatLng) {
                        combinedBounds.extend(layer.getLatLng());
                    }
                });
                
                this.map.fitBounds(combinedBounds);
                
                console.log(`✅ AOI set from SHP: ${polygonLayers.length} polygon(s), ${lineLayers.length} line(s), ${pointLayers.length} point(s)`);
                
                // Trigger callback
                if (this.onAOIChange) {
                    this.onAOIChange(true);
                }
                
                // Show legend
                const legend = document.getElementById('map-legend');
                if (legend) {
                    legend.classList.remove('hidden');
                }
                
                return true;
            } else {
                console.warn('⚠️ No geometry found in SHP file');
                throw new Error('No valid geometry found in the file');
            }
            
        } catch (error) {
            console.error('❌ Error loading SHP:', error);
            throw error;
        }
    }

    async loadKmlFile(file) {
        console.log('📂 Loading KML file:', file.name);
        
        try {
            // Read file as text
            const text = await file.text();
            
            // Parse KML using DOMParser
            const parser = new DOMParser();
            const kml = parser.parseFromString(text, 'text/xml');
            
            // Check for parsing errors
            const parseError = kml.querySelector('parsererror');
            if (parseError) {
                throw new Error('Invalid KML file format');
            }
            
            console.log('KML parsed, searching for coordinates...');
            
            // Method 1: Try to directly extract coordinates from KML
            // KML structure: Placemark > Polygon/LineString/Point > coordinates
            let polygonCoords = [];
            let lineCoords = [];
            let pointCoords = [];
            
            // Look for Polygon elements
            const polygons = kml.getElementsByTagName('Polygon');
            console.log(`Found ${polygons.length} Polygon elements`);
            
            for (const polygon of polygons) {
                const coordsElements = polygon.getElementsByTagName('coordinates');
                if (coordsElements.length > 0) {
                    const coordsText = coordsElements[0].textContent.trim();
                    const coords = this._parseKmlCoordinates(coordsText);
                    if (coords.length >= 3) {
                        polygonCoords.push(coords);
                        console.log(`Extracted ${coords.length} coordinates from Polygon`);
                    }
                }
            }
            
            // Look for LinearRing (part of Polygon, but process separately if no Polygon found)
            if (polygonCoords.length === 0) {
                const linearRings = kml.getElementsByTagName('LinearRing');
                console.log(`Found ${linearRings.length} LinearRing elements`);
                
                for (const ring of linearRings) {
                    const coordsElements = ring.getElementsByTagName('coordinates');
                    if (coordsElements.length > 0) {
                        const coordsText = coordsElements[0].textContent.trim();
                        const coords = this._parseKmlCoordinates(coordsText);
                        if (coords.length >= 3) {
                            polygonCoords.push(coords);
                            console.log(`Extracted ${coords.length} coordinates from LinearRing`);
                        }
                    }
                }
            }
            
            // Look for LineString elements
            const lineStrings = kml.getElementsByTagName('LineString');
            console.log(`Found ${lineStrings.length} LineString elements`);
            
            for (const line of lineStrings) {
                const coordsElements = line.getElementsByTagName('coordinates');
                if (coordsElements.length > 0) {
                    const coordsText = coordsElements[0].textContent.trim();
                    const coords = this._parseKmlCoordinates(coordsText);
                    if (coords.length >= 2) {
                        lineCoords.push(coords);
                        console.log(`Extracted ${coords.length} coordinates from LineString`);
                    }
                }
            }
            
            // Look for Point elements
            const points = kml.getElementsByTagName('Point');
            console.log(`Found ${points.length} Point elements`);
            
            for (const point of points) {
                const coordsElements = point.getElementsByTagName('coordinates');
                if (coordsElements.length > 0) {
                    const coordsText = coordsElements[0].textContent.trim();
                    const coords = this._parseKmlCoordinates(coordsText);
                    if (coords.length >= 1) {
                        pointCoords.push(coords[0]); // Single point
                        console.log(`Extracted Point coordinates`);
                    }
                }
            }
            
            // If still no polygons, look for generic coordinates elements with 3+ points
            if (polygonCoords.length === 0 && lineCoords.length === 0 && pointCoords.length === 0) {
                const allCoords = kml.getElementsByTagName('coordinates');
                console.log(`Found ${allCoords.length} coordinates elements`);
                
                for (const coordEl of allCoords) {
                    const coordsText = coordEl.textContent.trim();
                    const coords = this._parseKmlCoordinates(coordsText);
                    if (coords.length >= 3) {
                        polygonCoords.push(coords);
                        console.log(`Extracted ${coords.length} coordinates from generic coordinates element`);
                    } else if (coords.length === 2) {
                        lineCoords.push(coords);
                    } else if (coords.length === 1) {
                        pointCoords.push(coords[0]);
                    }
                }
            }
            
            // Clear existing drawings
            this.drawnItems.clearLayers();
            
            // Style for AOI polygons - use aoiPane to always be on top
            const aoiStyle = {
                color: '#ff7800',
                weight: 3,
                opacity: 0.8,
                fillColor: '#ff7800',
                fillOpacity: 0.1,
                pane: 'aoiPane'  // Always on top of data layers
            };
            
            // Style for lines
            const lineStyle = {
                color: '#ff7800',
                weight: 3,
                opacity: 0.8,
                pane: 'aoiPane'
            };
            
            // Style for points
            const pointStyle = {
                radius: 8,
                color: '#ff7800',
                weight: 2,
                opacity: 0.8,
                fillColor: '#ff7800',
                fillOpacity: 0.5,
                pane: 'aoiPane'
            };
            
            const createdLayers = [];
            const polygonLayers = [];
            const lineLayers = [];
            const pointLayers = [];
            
            // Create polygons from KML coordinates
            if (polygonCoords.length > 0) {
                console.log(`Creating ${polygonCoords.length} polygon(s) from KML coordinates`);
                
                for (let i = 0; i < polygonCoords.length; i++) {
                    const latlngs = polygonCoords[i].map(coord => [coord[1], coord[0]]); // [lng, lat] -> [lat, lng]
                    console.log(`Polygon ${i + 1}: ${latlngs.length} vertices`);
                    
                    const polygon = L.polygon(latlngs, aoiStyle);
                    createdLayers.push(polygon);
                    polygonLayers.push(polygon);
                    this.drawnItems.addLayer(polygon);
                }
            }
            
            // Create polygons from lines (convert LineString to filled Polygon)
            if (lineCoords.length > 0) {
                console.log(`Converting ${lineCoords.length} line(s) to polygon(s) from KML coordinates`);
                
                for (let i = 0; i < lineCoords.length; i++) {
                    const latlngs = lineCoords[i].map(coord => [coord[1], coord[0]]); // [lng, lat] -> [lat, lng]
                    console.log(`Line ${i + 1}: ${latlngs.length} vertices -> converting to polygon`);
                    
                    // Convert line to filled polygon
                    const polygon = L.polygon(latlngs, aoiStyle);
                    createdLayers.push(polygon);
                    polygonLayers.push(polygon);  // Add to polygon layers, not line layers
                    this.drawnItems.addLayer(polygon);
                }
            }
            
            // Create points from KML coordinates
            if (pointCoords.length > 0) {
                console.log(`Creating ${pointCoords.length} point(s) from KML coordinates`);
                
                for (let i = 0; i < pointCoords.length; i++) {
                    const latlng = [pointCoords[i][1], pointCoords[i][0]]; // [lng, lat] -> [lat, lng]
                    console.log(`Point ${i + 1}: [${latlng[0]}, ${latlng[1]}]`);
                    
                    const marker = L.circleMarker(latlng, pointStyle);
                    createdLayers.push(marker);
                    pointLayers.push(marker);
                    this.drawnItems.addLayer(marker);
                }
            }
            
            if (createdLayers.length > 0) {
                console.log(`✅ Created ${polygonLayers.length} polygon(s), ${lineLayers.length} line(s), ${pointLayers.length} point(s) from KML`);
            } else {
                // Fallback: Try toGeoJSON library
                console.log('No coordinates found directly, trying toGeoJSON library...');
                
                if (typeof toGeoJSON !== 'undefined' && toGeoJSON.kml) {
                    const geojson = toGeoJSON.kml(kml);
                    console.log('toGeoJSON result:', geojson);
                    
                    if (geojson && geojson.features && geojson.features.length > 0) {
                        // Log all features for debugging
                        geojson.features.forEach((f, i) => {
                            console.log(`Feature ${i}: type=${f.geometry?.type}`, f.geometry?.coordinates?.length);
                        });
                        
                        // Process each feature - convert lines to polygons
                        geojson.features.forEach(feature => {
                            const type = feature.geometry?.type;
                            
                            if (type === 'Polygon' || type === 'MultiPolygon') {
                                const layer = L.geoJSON(feature, { style: aoiStyle });
                                layer.eachLayer(l => {
                                    l.options.pane = 'aoiPane';
                                    createdLayers.push(l);
                                    polygonLayers.push(l);
                                    this.drawnItems.addLayer(l);
                                });
                            } else if (type === 'LineString') {
                                // Convert LineString to Polygon
                                console.log('Converting LineString to Polygon (toGeoJSON fallback)');
                                const coords = feature.geometry.coordinates.map(c => [c[1], c[0]]);
                                const polygon = L.polygon(coords, aoiStyle);
                                polygon.options.pane = 'aoiPane';
                                createdLayers.push(polygon);
                                polygonLayers.push(polygon);
                                this.drawnItems.addLayer(polygon);
                            } else if (type === 'MultiLineString') {
                                // Convert each line to Polygon
                                console.log('Converting MultiLineString to Polygons (toGeoJSON fallback)');
                                feature.geometry.coordinates.forEach(lineCoords => {
                                    const coords = lineCoords.map(c => [c[1], c[0]]);
                                    const polygon = L.polygon(coords, aoiStyle);
                                    polygon.options.pane = 'aoiPane';
                                    createdLayers.push(polygon);
                                    polygonLayers.push(polygon);
                                    this.drawnItems.addLayer(polygon);
                                });
                            } else if (type === 'Point') {
                                const latlng = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
                                const marker = L.circleMarker(latlng, pointStyle);
                                marker.options.pane = 'aoiPane';
                                createdLayers.push(marker);
                                pointLayers.push(marker);
                                this.drawnItems.addLayer(marker);
                            } else if (type === 'MultiPoint') {
                                feature.geometry.coordinates.forEach(coord => {
                                    const latlng = [coord[1], coord[0]];
                                    const marker = L.circleMarker(latlng, pointStyle);
                                    marker.options.pane = 'aoiPane';
                                    createdLayers.push(marker);
                                    pointLayers.push(marker);
                                    this.drawnItems.addLayer(marker);
                                });
                            }
                        });
                        
                        // If still nothing, get bounds
                        if (createdLayers.length === 0) {
                            const tempLayer = L.geoJSON(geojson);
                            const bounds = tempLayer.getBounds();
                            if (bounds.isValid()) {
                                console.warn('⚠️ Using bounds rectangle as fallback');
                                const rect = L.rectangle(bounds, aoiStyle);
                                createdLayers.push(rect);
                                polygonLayers.push(rect);
                                this.drawnItems.addLayer(rect);
                            }
                        }
                    }
                }
            }
            
            if (createdLayers.length > 0) {
                // Store all layers for multi-geometry support
                this.aoiPolygons = createdLayers;
                
                // Set currentAOI to the first layer (for backward compatibility)
                this.currentAOI = createdLayers[0];
                
                // Calculate combined bounds of all geometries
                const combinedBounds = L.latLngBounds();
                createdLayers.forEach(layer => {
                    if (layer.getBounds) {
                        combinedBounds.extend(layer.getBounds());
                    } else if (layer.getLatLng) {
                        // For point markers
                        combinedBounds.extend(layer.getLatLng());
                    }
                });
                
                if (combinedBounds.isValid()) {
                    this.map.fitBounds(combinedBounds);
                }
                
                console.log(`✅ AOI set from KML: ${polygonLayers.length} polygon(s), ${lineLayers.length} line(s), ${pointLayers.length} point(s)`);
                console.log('Combined bounds:', combinedBounds);
                
                if (this.onAOIChange) {
                    this.onAOIChange(true);
                }
                
                const legend = document.getElementById('map-legend');
                if (legend) {
                    legend.classList.remove('hidden');
                }
                
                return true;
            } else {
                throw new Error('No valid geometry coordinates found in KML file');
            }
            
        } catch (error) {
            console.error('❌ Error loading KML:', error);
            throw error;
        }
    }
    
    // Helper function to parse KML coordinate string
    _parseKmlCoordinates(coordString) {
        const coords = [];
        // KML coordinates format: "lng,lat,alt lng,lat,alt ..." or "lng,lat lng,lat ..."
        // Split by whitespace (space, newline, tab)
        const parts = coordString.split(/\s+/).filter(p => p.length > 0);
        
        for (const part of parts) {
            const values = part.split(',').map(v => parseFloat(v.trim()));
            if (values.length >= 2 && !isNaN(values[0]) && !isNaN(values[1])) {
                coords.push([values[0], values[1]]); // [lng, lat]
            }
        }
        
        return coords;
    }

    addImageLayer(imageId, imageData) {
        try {
            console.log(`Adding AOI-cropped image layer ${imageId} to map`, imageData);
            
            // Use AOI bounds for image display
            if (!this.currentAOI) {
                console.warn('No AOI defined for image display');
                return;
            }

            const aoiBounds = this.getCombinedAOIBounds();
            const defaultBounds = [
                [aoiBounds.getSouth(), aoiBounds.getWest()], // Southwest
                [aoiBounds.getNorth(), aoiBounds.getEast()]  // Northeast
            ];

            // Normalize tile_bounds to Leaflet format [[south, west], [north, east]]
            let bounds = defaultBounds;
            if (imageData.tile_bounds) {
                if (Array.isArray(imageData.tile_bounds) && imageData.tile_bounds.length === 4) {
                    // Backend returns [west, south, east, north] (bbox format)
                    // Convert to Leaflet format [[south, west], [north, east]]
                    const [west, south, east, north] = imageData.tile_bounds;
                    bounds = [[south, west], [north, east]];
                    console.log(`Converted bbox [${west}, ${south}, ${east}, ${north}] to Leaflet bounds:`, bounds);
                } else if (Array.isArray(imageData.tile_bounds) && imageData.tile_bounds.length === 2) {
                    // Already in Leaflet format [[south, west], [north, east]]
                    bounds = imageData.tile_bounds;
                }
            }

            let imageLayer = null;
            
            // Check if we should use dynamic tiles or static image
            if (imageData.display_type === 'tile' && imageData.tile_template) {
                // Use dynamic tile layer for high-resolution zooming
                console.log(`Creating dynamic tile layer for ${imageId}`);
                console.log(`Tile template: ${imageData.tile_template}`);
                
                // Create tile layer with AOI bounds restriction
                // Use custom dataPane to ensure data layers are above base map but below labels
                imageLayer = L.tileLayer(imageData.tile_template, {
                    opacity: 1.0,
                    attribution: '© Sentinel-2 via TiTiler',
                    bounds: bounds,
                    maxZoom: 18,
                    minZoom: 6,
                    tileSize: 256,
                    crossOrigin: true,
                    pane: 'dataPane'  // Custom pane (z-index 450) - above tiles, below labels
                });
                
                console.log(`Dynamic tile layer created for ${imageId}:`, {
                    template: imageData.tile_template,
                    bounds: bounds
                });
                
            } else {
                // Fallback to static image overlay
                let imageUrl = null;
                let urlType = 'none';
                
                if (imageData.display_url) {
                    imageUrl = imageData.display_url;
                    urlType = 'display';
                } else if (imageData.assets && imageData.assets.thumbnail) {
                    imageUrl = imageData.assets.thumbnail;
                    urlType = 'thumbnail';
                }

                console.log(`Creating static image overlay for ${imageId} using ${urlType} URL: ${imageUrl}`);

                if (imageUrl) {
                    imageLayer = L.imageOverlay(imageUrl, bounds, {
                        opacity: 1.0,
                        interactive: true,
                        crossOrigin: true,
                        pane: 'dataPane'  // Custom pane (z-index 450) - above tiles, below labels
                    });

                    // Error handling for static images
                    imageLayer.on('error', (e) => {
                        console.error(`Failed to load static image ${imageId} from ${imageUrl}:`, e);
                        
                        // Remove failed layer
                        if (this.imageLayers[imageId]) {
                            this.map.removeLayer(this.imageLayers[imageId]);
                            delete this.imageLayers[imageId];
                        }
                    });

                    // Success handling for static images
                    imageLayer.on('load', () => {
                        console.log(`Successfully loaded static image ${imageId}`);
                    });
                }
            }

            if (imageLayer) {
                // Common click event for both tile and image layers
                imageLayer.on('click', () => {
                    this.selectImageLayer(imageId);
                });

                // Tooltip with AOI overlap information
                const dateStr = imageData.datetime ? new Date(imageData.datetime).toLocaleDateString() : 'Unknown';
                const aoiOverlap = imageData.aoi_overlap ? (imageData.aoi_overlap * 100).toFixed(1) + '%' : 'N/A';
                const layerType = imageData.display_type === 'tile' ? 'Dynamic Tile Layer' : 'Static Image';
                
                const tooltipContent = `
                    <div>
                        <strong>Image ID:</strong> ${imageId}<br>
                        <strong>Date:</strong> ${dateStr}<br>
                        <strong>AOI Coverage:</strong> ${aoiOverlap}<br>
                        <strong>Cloud Cover:</strong> ${imageData.cloud_cover ? imageData.cloud_cover.toFixed(1) : 'N/A'}%<br>
                        <strong>Type:</strong> ${layerType}
                    </div>
                `;

                imageLayer.bindTooltip(tooltipContent, {
                    permanent: false,
                    direction: 'top'
                });

                // Add to map and store reference
                this.imageLayers[imageId] = imageLayer;
                this.map.addLayer(imageLayer);
 
                // Ensure AOI outline-only style is applied after layer add
                this.outlineAOI();

                const layerTypeStr = imageData.display_type === 'tile' ? 'dynamic tile layer' : 'static image layer';
                console.log(`Successfully added ${layerTypeStr}: ${imageId}`);
                
                // Show legend (if it exists)
                const legend = document.getElementById('map-legend');
                if (legend) {
                    legend.classList.remove('hidden');
                }
                
            } else {
                console.warn(`No valid image URL found for ${imageId}, skipping layer creation`);
            }
        } catch (error) {
            console.error(`Error adding image layer ${imageId}:`, error);
        }
    }

    clearImageLayers() {
        console.log('Clearing all image layers');
        
        Object.keys(this.imageLayers).forEach(layerId => {
            try {
                const layer = this.imageLayers[layerId];
                
                // Remove layer from map
                this.map.removeLayer(layer);
                console.log(`Removed image layer: ${layerId}`);
                
            } catch (error) {
                console.error(`Error removing layer ${layerId}:`, error);
            }
        });
        
        // Clear references
        this.imageLayers = {};
        this.selectedImageLayer = null;
         
        console.log('All image layers cleared');
    }

    removeImageLayer(imageId) {
        if (this.imageLayers[imageId]) {
            try {
                this.map.removeLayer(this.imageLayers[imageId]);
                delete this.imageLayers[imageId];
                
                if (this.selectedImageLayer === imageId) {
                    this.selectedImageLayer = null;
                }
                
                console.log(`Removed image layer: ${imageId}`);
            } catch (error) {
                console.error(`Error removing image layer ${imageId}:`, error);
            }
        }
    }

    selectImageLayer(imageId) {
        // Highlight selected layer
        Object.keys(this.imageLayers).forEach(id => {
            const layer = this.imageLayers[id];
            if (id === imageId) {
                layer.setStyle ? layer.setStyle({opacity: 1.0}) : layer.setOpacity(1.0);
                this.selectedImageLayer = imageId;
                console.log(`Selected image layer: ${imageId}`);
            } else {
                layer.setStyle ? layer.setStyle({opacity: 0.6}) : layer.setOpacity(0.6);
            }
        });
    }

    fitToAOI() {
        if (this.currentAOI) {
            try {
                console.log('Fitting map view to AOI for optimal display');
                
                // Get combined AOI bounds (supports multi-polygon)
                const aoiBounds = this.getCombinedAOIBounds();
                
                // Fit map to AOI with padding
                this.map.fitBounds(aoiBounds, {
                    padding: [20, 20], // Add padding around AOI
                    maxZoom: 16,       // Optimal zoom for image detail
                    animate: true,     // Smooth animation
                    duration: 1.0      // Animation duration
                });
                
                // Enhance AOI styling for all polygons
                const aoiStyle = {
                    color: '#ff7800',
                    weight: 3,
                    opacity: 0.9,
                    fillColor: '#ff7800',
                    fillOpacity: 0.0
                };
                
                if (this.aoiPolygons && this.aoiPolygons.length > 0) {
                    this.aoiPolygons.forEach(polygon => polygon.setStyle(aoiStyle));
                } else {
                    this.currentAOI.setStyle(aoiStyle);
                }
                
                console.log('Map fitted to AOI bounds:', aoiBounds);
                
                // Log current zoom level
                setTimeout(() => {
                    const currentZoom = this.map.getZoom();
                    console.log(`Current zoom level: ${currentZoom}`);
                }, 1100); // After fitBounds animation
                
            } catch (error) {
                console.error('Error fitting to AOI:', error);
            }
        } else {
            console.warn('No AOI defined to fit to');
        }
    }

    getMapBounds() {
        return this.map.getBounds();
    }

    getMapCenter() {
        return this.map.getCenter();
    }

    getMapZoom() {
        return this.map.getZoom();
    }

    setMapView(center, zoom) {
        this.map.setView(center, zoom);
    }

    panTo(latLng) {
        this.map.panTo(latLng);
    }

    // Show processed image result
    showProcessedImage(imageUrl) {
        if (!this.currentAOI) {
            console.warn('No AOI defined for processed image display');
            return;
        }

        try {
            // Clear existing processed image layers
            Object.keys(this.processedLayers).forEach(id => {
                this.map.removeLayer(this.processedLayers[id]);
                delete this.processedLayers[id];
            });

            const aoiBounds = this.getCombinedAOIBounds();
            const bounds = [
                [aoiBounds.getSouth(), aoiBounds.getWest()],
                [aoiBounds.getNorth(), aoiBounds.getEast()]
            ];

            const processedLayer = L.imageOverlay(imageUrl, bounds, {
                opacity: 1.0,
                interactive: true,
                crossOrigin: true,
                pane: 'dataPane'  // Custom pane (z-index 450) - above tiles, below labels
            });

            processedLayer.bindTooltip(
                '<div><strong>Processed Image</strong><br>BGR Visualization</div>',
                { permanent: false, direction: 'top' }
            );

            const processedId = 'processed_' + Date.now();
            this.processedLayers[processedId] = processedLayer;
            
            if (this.showProcessedImages) {
                this.map.addLayer(processedLayer);
            }

            console.log('Processed image displayed on map');

        } catch (error) {
            console.error('Error displaying processed image:', error);
        }
    }

    // Toggle processed images visibility
    toggleProcessedImages() {
        this.showProcessedImages = !this.showProcessedImages;
        
        Object.keys(this.processedLayers).forEach(id => {
            const layer = this.processedLayers[id];
            if (this.showProcessedImages) {
                if (!this.map.hasLayer(layer)) {
                    this.map.addLayer(layer);
                }
            } else {
                if (this.map.hasLayer(layer)) {
                    this.map.removeLayer(layer);
                }
            }
        });
        
        return this.showProcessedImages;
    }

    // Show analysis layer for multi-model results
    showAnalysisLayer(modelId, imageUrl, modelName, boundsOverride = null) {
        if (!this.currentAOI) {
            console.warn('No AOI defined for analysis layer display');
            return;
        }

        try {
            // Remove existing analysis layer for this model
            this.hideAnalysisLayer(modelId);

            let bounds;
            if (boundsOverride && Array.isArray(boundsOverride) && boundsOverride.length === 2) {
                bounds = boundsOverride;
            } else {
                const aoiBounds = this.getCombinedAOIBounds();
                bounds = [
                    [aoiBounds.getSouth(), aoiBounds.getWest()],
                    [aoiBounds.getNorth(), aoiBounds.getEast()]
                ];
            }

            const analysisLayer = L.imageOverlay(imageUrl, bounds, {
                opacity: 1.0,
                interactive: true,
                crossOrigin: true,
                pane: 'dataPane'  // Custom pane (z-index 450) - above tiles, below labels
            });

            analysisLayer.on('error', (e) => {
                console.error(`Failed to load analysis overlay ${modelId} from ${imageUrl}:`, e);
            });
            analysisLayer.on('load', () => {
                console.log(`Analysis overlay ${modelId} loaded successfully`);
            });

            analysisLayer.bindTooltip(
                `<div><strong>${modelName}</strong><br>Analysis Model: ${modelId}</div>`,
                { permanent: false, direction: 'top' }
            );

            this.analysisLayers[modelId] = analysisLayer;
            this.map.addLayer(analysisLayer);
            // Note: Removed automatic map fitting to preserve user's current view

            console.log(`Analysis layer ${modelId} displayed on map`, { bounds });

        } catch (error) {
            console.error(`Error displaying analysis layer ${modelId}:`, error);
        }
    }

    // Hide analysis layer
    hideAnalysisLayer(modelId) {
        if (this.analysisLayers[modelId]) {
            try {
                this.map.removeLayer(this.analysisLayers[modelId]);
                delete this.analysisLayers[modelId];
                console.log(`Analysis layer ${modelId} removed from map`);
            } catch (error) {
                console.error(`Error removing analysis layer ${modelId}:`, error);
            }
        } else if (this.modelLayerCache[modelId]) {
            // Layer was restored from cache without analysisLayers being set
            try {
                const cachedLayer = this.modelLayerCache[modelId];
                if (this.map.hasLayer(cachedLayer)) {
                    this.map.removeLayer(cachedLayer);
                }
                console.log(`Cached analysis layer ${modelId} removed from map`);
            } catch (error) {
                console.error(`Error removing cached analysis layer ${modelId}:`, error);
            }
        }
    }

    // Show tile layer for period composites
    showTileLayer(layerId, tileTemplate, bounds, tooltip) {
        try {
            // Remove existing tile layer
            this.hideTileLayer(layerId);

            // Initialize tile layers storage if not exists
            if (!this.tileLayers) {
                this.tileLayers = {};
            }

            // Create Leaflet tile layer
            // Use custom dataPane to ensure data layers are above base map but below labels
            const tileLayer = L.tileLayer(tileTemplate, {
                attribution: 'Google Earth Engine',
                opacity: 1.0,
                maxZoom: 18,
                pane: 'dataPane'  // Custom pane (z-index 450) - above tiles, below labels
            });

            if (tooltip) {
                tileLayer.bindTooltip(tooltip, { permanent: false, direction: 'top' });
            }

            this.tileLayers[layerId] = tileLayer;
            this.map.addLayer(tileLayer);
            
            console.log(`Tile layer ${layerId} added with opacity 1.0 to hide OSM background`);

            // Note: Removed automatic map fitting to preserve user's current view

            console.log(`Tile layer ${layerId} displayed on map`);

        } catch (error) {
            console.error(`Error displaying tile layer ${layerId}:`, error);
        }
    }

    // Hide tile layer
    hideTileLayer(layerId) {
        if (this.tileLayers && this.tileLayers[layerId]) {
            try {
                this.map.removeLayer(this.tileLayers[layerId]);
                delete this.tileLayers[layerId];
                console.log(`Tile layer ${layerId} removed from map`);
            } catch (error) {
                console.error(`Error removing tile layer ${layerId}:`, error);
            }
        }
    }

    // Make AOI outline only (no fill) - supports multi-polygon
    outlineAOI() {
        const outlineStyle = {
            color: '#ff7800',
            weight: 3,
            opacity: 0.9,
            fillColor: '#ff7800',
            fillOpacity: 0.0
        };
        
        try {
            // Style all polygons if we have multiple
            if (this.aoiPolygons && this.aoiPolygons.length > 0) {
                this.aoiPolygons.forEach(polygon => {
                    if (polygon && polygon.setStyle) {
                        polygon.setStyle(outlineStyle);
                    }
                });
            } else if (this.currentAOI && this.currentAOI.setStyle) {
                this.currentAOI.setStyle(outlineStyle);
            }
        } catch (e) {
            console.warn('Failed to set AOI outline style', e);
        }
    }

	// Add COG from GCS (6-band export: B4,B3,B2,B8,B11,B12)
	async showOriginalCOG(signedUrl, options = {opacity: 0.9}) {
		try {
			const georaster = await this._ensureCOG(signedUrl);
			if (this.originalLayer) {
				this.map.removeLayer(this.originalLayer);
			}
			const LayerCtor = window.GeoRasterLayer || (window.georaster && window.georaster.GeoRasterLayer) || (window['georasterLayerForLeaflet'] && window['georasterLayerForLeaflet'].GeoRasterLayer);
			if (!LayerCtor) throw new Error('GeoRasterLayer not loaded');
			const toByte = v => {
				let x = Number.isFinite(v) ? v : 0;
				if (x > 1) x = x / 10000;
				if (!Number.isFinite(x)) x = 0;
				x = Math.max(0, Math.min(1, x));
				return Math.round(x * 255);
			};
			const layer = new LayerCtor({
				georaster,
				                opacity: options.opacity !== undefined ? options.opacity : 0.9,
				resolution: 256,
				pixelValuesToColorFn: values => {
					const r = values[0], g = values[1], b = values[2];
					return `rgb(${toByte(r)}, ${toByte(g)}, ${toByte(b)})`;
				}
			});
			this.originalLayer = layer;
			layer.addTo(this.map);
			this.fitToAOI();
			this.outlineAOI();
			console.log('Original COG displayed (cached)');
		} catch (e) {
			console.error('Failed to display original COG:', e);
		}
	}

	// Add analysis overlay from original COG with a model-specific pixel function
	async showModelFromCOG(modelId, signedUrl, options = {opacity: 0.7}) {
		try {
			// If already created once, just add back to map and track in analysisLayers
			if (this.modelLayerCache[modelId]) {
				const cachedLayer = this.modelLayerCache[modelId];
				this.analysisLayers[modelId] = cachedLayer;
				if (!this.map.hasLayer(cachedLayer)) {
					this.map.addLayer(cachedLayer);
				}
				console.log(`Model ${modelId} layer restored from cache`);
				return;
			}
			const georaster = await this._ensureCOG(signedUrl);
			this.hideAnalysisLayer(modelId);
			const pf = this._getPixelFnForModel(modelId);
			const LayerCtor = window.GeoRasterLayer || (window.georaster && window.georaster.GeoRasterLayer) || (window['georasterLayerForLeaflet'] && window['georasterLayerForLeaflet'].GeoRasterLayer);
			if (!LayerCtor) throw new Error('GeoRasterLayer not loaded');
			const layer = new LayerCtor({ georaster, opacity: options.opacity !== undefined ? options.opacity : 0.7, resolution: 256, pixelValuesToColorFn: pf });
			this.analysisLayers[modelId] = layer;
			this.modelLayerCache[modelId] = layer; // cache for fast toggle
			layer.addTo(this.map);
			console.log(`Model ${modelId} from COG displayed (cached)`);
		} catch (e) {
			console.error('Failed to display model from COG:', e);
		}
	}

	async showBinaryMaskCOG(modelId, signedUrl, options = {opacity: 1.0}) {
		try {
			if (this.modelLayerCache[modelId]) {
				const cachedLayer = this.modelLayerCache[modelId];
				this.analysisLayers[modelId] = cachedLayer;
				if (!this.map.hasLayer(cachedLayer)) {
					this.map.addLayer(cachedLayer);
				}
				console.log(`Binary mask ${modelId} layer restored from cache`);
				return;
			}
			const georaster = await this._ensureCOG(signedUrl);
			this.hideAnalysisLayer(modelId);
			const LayerCtor = window.GeoRasterLayer || (window.georaster && window.georaster.GeoRasterLayer) || (window['georasterLayerForLeaflet'] && window['georasterLayerForLeaflet'].GeoRasterLayer);
			if (!LayerCtor) throw new Error('GeoRasterLayer not loaded');
			const pixelValuesToColorFn = values => {
				const v = Array.isArray(values) ? values[0] : values;
				const val = Number.isFinite(v) ? v : 0;
				if (val >= 1) return 'rgba(255,0,0,0.4)';
				return 'rgba(0,0,0,0)';
			};
			const layer = new LayerCtor({ georaster, opacity: options.opacity !== undefined ? options.opacity : 1.0, resolution: 256, pixelValuesToColorFn });
			this.analysisLayers[modelId] = layer;
			this.modelLayerCache[modelId] = layer;
			layer.addTo(this.map);
			console.log(`Binary mask ${modelId} from COG displayed (cached)`);
		} catch (e) {
			console.error('Failed to display binary mask COG:', e);
		}
	}

	async prepareModelLayerFromCOG(modelId, signedUrl, options = {opacity: 0.7}) {
		try {
			// Ensure COG is cached
			const georaster = await this._ensureCOG(signedUrl);
			// If layer already cached, nothing to do
			if (this.modelLayerCache[modelId]) return;
			const pf = this._getPixelFnForModel(modelId);
			const LayerCtor = window.GeoRasterLayer || (window.georaster && window.georaster.GeoRasterLayer) || (window['georasterLayerForLeaflet'] && window['georasterLayerForLeaflet'].GeoRasterLayer);
			if (!LayerCtor) throw new Error('GeoRasterLayer not loaded');
			const layer = new LayerCtor({ georaster, opacity: options.opacity !== undefined ? options.opacity : 0.7, resolution: 256, pixelValuesToColorFn: pf });
			this.modelLayerCache[modelId] = layer; // cache only, do not add to map
			console.log(`Model ${modelId} layer prepared and cached`);
		} catch (e) {
			console.warn(`Failed to prepare model ${modelId} layer:`, e);
		}
	}

	async prepareBinaryMaskLayerFromCOG(modelId, signedUrl, options = {opacity: 1.0}) {
		try {
			const georaster = await this._ensureCOG(signedUrl);
			if (this.modelLayerCache[modelId]) return;
			const LayerCtor = window.GeoRasterLayer || (window.georaster && window.georaster.GeoRasterLayer) || (window['georasterLayerForLeaflet'] && window['georasterLayerForLeaflet'].GeoRasterLayer);
			if (!LayerCtor) throw new Error('GeoRasterLayer not loaded');
			const pixelValuesToColorFn = values => {
				const v = Array.isArray(values) ? values[0] : values;
				const val = Number.isFinite(v) ? v : 0;
				if (val >= 1) return 'rgba(255,0,0,0.4)';
				return 'rgba(0,0,0,0)';
			};
			const layer = new LayerCtor({ georaster, opacity: options.opacity !== undefined ? options.opacity : 1.0, resolution: 256, pixelValuesToColorFn });
			this.modelLayerCache[modelId] = layer;
			console.log(`Binary mask ${modelId} layer prepared and cached`);
		} catch (e) {
			console.warn(`Failed to prepare binary mask ${modelId} layer:`, e);
		}
	}

	_getPixelFnForModel(modelId) {
		// values: [B4(R), B3(G), B2(B), B8(NIR), B11(SWIR1), B12(SWIR2)]
		const toByte = v => {
			let x = Number.isFinite(v) ? v : 0;
			if (x > 1) x = x / 10000;
			if (!Number.isFinite(x)) x = 0;
			x = Math.max(0, Math.min(1, x));
			return Math.round(x * 255);
		};
		if (modelId === 'model1') {
			return values => `rgb(${toByte(values[0])}, ${toByte(values[1])}, ${toByte(values[2])})`;
		}
		if (modelId === 'model2') {
			// NIR-Red-Green -> map to RGB as NIR, Red, Green for visualization
			return values => `rgb(${toByte(values[3])}, ${toByte(values[0])}, ${toByte(values[1])})`;
		}
		if (modelId === 'model3') {
			// SWIR2, SWIR1, Red
			return values => `rgb(${toByte(values[5])}, ${toByte(values[4])}, ${toByte(values[0])})`;
		}
		if (modelId === 'model4') {
			// Vegetation highlight using NIR and Red (simple false color)
			return values => `rgb(${toByte(values[3])}, ${toByte(values[0])}, ${toByte(values[2])})`;
		}
		// default RGB
		return values => `rgb(${toByte(values[0])}, ${toByte(values[1])}, ${toByte(values[2])})`;
	}

    async prefetchCOG(url) {
        try {
            await this._ensureCOG(url);
            console.log('COG prefetched and cached');
        } catch (e) {
            console.warn('COG prefetch failed:', e);
        }
    }

    async _ensureCOG(url) {
        if (this.cogCache[url]) return this.cogCache[url];
        const response = await fetch(url, { mode: 'cors' });
        const arrayBuffer = await response.arrayBuffer();
        const parseFn = window.parseGeoraster || (window.georaster && window.georaster.parseGeoraster);
        if (!parseFn) throw new Error('parseGeoraster not loaded');
        const georaster = await parseFn(arrayBuffer);
        this.cogCache[url] = georaster;
        return georaster;
    }
}

// MapManager class ends here - initialization is now handled in index.html 