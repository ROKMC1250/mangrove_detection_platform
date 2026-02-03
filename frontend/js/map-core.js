/**
 * Map Core Module
 * Handles map initialization, base layers, and coordinate display
 */

class MapCore {
    constructor(mapId, manager) {
        this.mapId = mapId;
        this.manager = manager;
        this.map = null;
        this.osmLayer = null;
        this.satelliteLayer = null;
        this.labelsLayer = null;
        this.baseLayer = null;
        this.layerControl = null;
        this.currentBaseLayerType = 'osm';
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
            mapContainer._leaflet_id = null;
            mapContainer.innerHTML = '';
        }
        
        // Initialize map centered on Philippines (mangrove region)
        // Set maxZoom to 18 to prevent "tiles not available" issues
        this.map = L.map(this.mapId, {
            center: [13.0, 122.0],
            zoom: 8,
            minZoom: 3,
            maxZoom: 18,
            zoomControl: true,
            attributionControl: true
        });

        // Store reference in manager
        this.manager.map = this.map;

        // Create custom panes for proper layer ordering
        this.createCustomPanes();

        // Create base layers
        this.createBaseLayers();

        // Add default layer (OSM)
        this.baseLayer = this.osmLayer;
        this.osmLayer.addTo(this.map);
        this.currentBaseLayerType = 'osm';
        console.log('🗺️ Default layer (OSM) added to map');
        
        // Ensure satellite and labels are NOT added initially
        if (this.map.hasLayer(this.satelliteLayer)) {
            this.map.removeLayer(this.satelliteLayer);
        }
        if (this.map.hasLayer(this.labelsLayer)) {
            this.map.removeLayer(this.labelsLayer);
        }

        // Add custom layer control
        this.addCustomLayerControl();
        
        // Debug: Log initial layer state
        this.debugLayerState();

        // Add coordinate display control
        this.addCoordinateDisplay();

        console.log('✅ Map Core initialized');
    }

    createCustomPanes() {
        // Pane z-index order: basemapPane (100) < tilePane (200) < dataPane (450) < aoiPane (500) < labelsPane (650)
        
        // Basemap pane - LOWEST z-index for base maps (OSM/Satellite)
        this.map.createPane('basemapPane');
        this.map.getPane('basemapPane').style.zIndex = 100;
        
        // Data pane for analysis results (satellite tiles, NDVI overlays, etc.)
        this.map.createPane('dataPane');
        this.map.getPane('dataPane').style.zIndex = 450;
        
        // AOI pane for drawn items and KML polygons - ALWAYS on top of data
        this.map.createPane('aoiPane');
        this.map.getPane('aoiPane').style.zIndex = 500;
        
        // Labels pane for map labels
        this.map.createPane('labelsPane');
        this.map.getPane('labelsPane').style.zIndex = 650;
        
        console.log('✅ Custom panes created: basemapPane (100), dataPane (450), aoiPane (500), labelsPane (650)');
    }

    createBaseLayers() {
        // Map base layer with English labels (using CARTO Voyager)
        // Limited to maxZoom 18 to prevent "tiles not available" issues
        this.osmLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap contributors, © CARTO',
            maxZoom: 18,
            maxNativeZoom: 18,
            subdomains: 'abcd',
            pane: 'basemapPane'
        });
        
        // Satellite base layer - ESRI World Imagery
        // Limited to maxZoom 18 to prevent "tiles not available" issues
        this.satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '© Esri, Maxar, Earthstar Geographics',
            maxZoom: 18,
            maxNativeZoom: 18,
            pane: 'basemapPane'
        });

        // Labels overlay for Satellite
        this.labelsLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap contributors, © CARTO',
            maxZoom: 18,
            maxNativeZoom: 18,
            subdomains: 'abcd',
            pane: 'labelsPane'
        });

        // Store references in manager
        this.manager.osmLayer = this.osmLayer;
        this.manager.satelliteLayer = this.satelliteLayer;
        this.manager.labelsLayer = this.labelsLayer;
    }

    addCoordinateDisplay() {
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
        if (this.layerControl) {
            console.log('⚠️ Layer control already exists, skipping');
            return;
        }
        
        console.log('🎛️ Creating custom layer control...');
        
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
            
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            
            const osmOption = div.querySelector('.layer-option[data-layer="osm"]');
            const satelliteOption = div.querySelector('.layer-option[data-layer="satellite"]');
            const osmRadio = div.querySelector('#layer-osm');
            const satelliteRadio = div.querySelector('#layer-satellite');
            
            if (osmOption && osmRadio) {
                L.DomEvent.on(osmOption, 'click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    osmRadio.checked = true;
                    satelliteRadio.checked = false;
                    this.switchToOSM();
                }, this);
            }
            
            if (satelliteOption && satelliteRadio) {
                L.DomEvent.on(satelliteOption, 'click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    satelliteRadio.checked = true;
                    osmRadio.checked = false;
                    this.switchToSatellite();
                }, this);
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
        if (this.currentBaseLayerType === 'osm') {
            console.log('ℹ️ Already on OSM layer, skipping switch');
            return;
        }
        
        console.log('🔄 Switching to OSM layer...');
        
        if (this.map.hasLayer(this.satelliteLayer)) {
            this.map.removeLayer(this.satelliteLayer);
        }
        if (this.map.hasLayer(this.labelsLayer)) {
            this.map.removeLayer(this.labelsLayer);
        }
        
        if (!this.map.hasLayer(this.osmLayer)) {
            this.osmLayer.addTo(this.map);
        }
        
        this.baseLayer = this.osmLayer;
        this.manager.baseLayer = this.osmLayer;
        this.currentBaseLayerType = 'osm';
        this.manager.currentBaseLayerType = 'osm';
        
        console.log('✅ Switched to OSM layer');
        this.debugLayerState();
        
        this.bringDataLayersToFront();
    }

    switchToSatellite() {
        if (this.currentBaseLayerType === 'satellite') {
            console.log('ℹ️ Already on Satellite layer, skipping switch');
            return;
        }
        
        console.log('🔄 Switching to Satellite layer...');
        
        if (this.map.hasLayer(this.osmLayer)) {
            this.map.removeLayer(this.osmLayer);
        }
        
        if (!this.map.hasLayer(this.satelliteLayer)) {
            this.satelliteLayer.addTo(this.map);
        }
        
        if (!this.map.hasLayer(this.labelsLayer)) {
            this.labelsLayer.addTo(this.map);
        }
        
        this.baseLayer = this.satelliteLayer;
        this.manager.baseLayer = this.satelliteLayer;
        this.currentBaseLayerType = 'satellite';
        this.manager.currentBaseLayerType = 'satellite';
        
        console.log('✅ Switched to Satellite layer with English labels and borders');
        
        this.bringDataLayersToFront();
        
        setTimeout(() => {
            this.debugLayerState();
        }, 500);
    }

    bringDataLayersToFront() {
        console.log('🔝 Bringing data layers to front...');
        
        if (this.baseLayer && this.map.hasLayer(this.baseLayer)) {
            this.baseLayer.bringToBack();
        }
        
        // Bring image layers to front
        if (this.manager.imageLayers) {
            Object.keys(this.manager.imageLayers).forEach(id => {
                const layer = this.manager.imageLayers[id];
                if (this.map.hasLayer(layer)) {
                    layer.bringToFront();
                }
            });
        }
        
        // Bring processed layers to front
        if (this.manager.processedLayers) {
            Object.keys(this.manager.processedLayers).forEach(id => {
                const layer = this.manager.processedLayers[id];
                if (this.map.hasLayer(layer)) {
                    layer.bringToFront();
                }
            });
        }
        
        // Bring analysis layers to front
        if (this.manager.analysisLayers) {
            Object.keys(this.manager.analysisLayers).forEach(id => {
                const layer = this.manager.analysisLayers[id];
                if (this.map.hasLayer(layer)) {
                    layer.bringToFront();
                }
            });
        }
        
        // Bring tile layers to front
        if (this.manager.tileLayers) {
            Object.keys(this.manager.tileLayers).forEach(id => {
                const layer = this.manager.tileLayers[id];
                if (this.map.hasLayer(layer)) {
                    layer.bringToFront();
                }
            });
        }
        
        // Bring drawn items (AOI polygons) to front last
        if (this.manager.drawnItems && this.map.hasLayer(this.manager.drawnItems)) {
            this.manager.drawnItems.bringToFront();
        }
        
        // Also bring individual AOI polygons to front
        if (this.manager.aoiPolygons && this.manager.aoiPolygons.length > 0) {
            this.manager.aoiPolygons.forEach(polygon => {
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
        
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 10000);
    }

    // Map utility methods
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
}
