/**
 * Map Drawing Module
 * Handles AOI drawing, bounds management, and GeoJSON conversion
 */

class MapDrawing {
    constructor(manager) {
        this.manager = manager;
        this.drawnItems = null;
        this.currentAOI = null;
        this.aoiPolygons = [];
        this.drawControl = null;
        this.currentDrawer = null;
        this.onAOIChange = null;
    }

    initialize() {
        // Initialize drawing layer
        this.drawnItems = new L.FeatureGroup();
        this.manager.map.addLayer(this.drawnItems);
        
        // Store references in manager
        this.manager.drawnItems = this.drawnItems;
        this.manager.aoiPolygons = this.aoiPolygons;

        // Setup drawing controls
        this.setupDrawingControls();
        
        // Setup map event listeners
        this.setupMapEvents();

        console.log('✅ Map Drawing initialized');
    }

    setupDrawingControls() {
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
                        pane: 'aoiPane'
                    },
                    metric: true,
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

        console.log('✅ Drawing controls configured');
    }

    setupMapEvents() {
        const map = this.manager.map;

        // AOI creation event
        map.on(L.Draw.Event.CREATED, (event) => {
            const layer = event.layer;
            
            console.log('🎯 Draw event triggered:', event.layerType);
            
            // Disable the drawer immediately after creation
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
            this.manager.currentAOI = layer;
            
            // Clear and reset aoiPolygons array
            this.aoiPolygons = [layer];
            this.manager.aoiPolygons = this.aoiPolygons;
            
            // Get AOI bounds
            const bounds = layer.getBounds();
            console.log('✅ AOI Created:', bounds);
            
            // Trigger callback
            if (this.onAOIChange) {
                this.onAOIChange(true);
            }
            if (this.manager.onAOIChange) {
                this.manager.onAOIChange(true);
            }
            
            // Show legend
            const legend = document.getElementById('map-legend');
            if (legend) {
                legend.classList.remove('hidden');
            }
            
            // Re-enable drawing button
            const drawBtn = document.getElementById('draw-rectangle');
            if (drawBtn) {
                drawBtn.disabled = false;
                drawBtn.textContent = '📍 Draw AOI';
            }
        });

        // AOI editing events
        map.on(L.Draw.Event.EDITED, (event) => {
            if (this.currentAOI) {
                const bounds = this.currentAOI.getBounds();
                console.log('✏️ AOI Edited:', bounds);
                
                if (this.onAOIChange) {
                    this.onAOIChange(true);
                }
                if (this.manager.onAOIChange) {
                    this.manager.onAOIChange(true);
                }
            }
        });

        // AOI deletion event
        map.on(L.Draw.Event.DELETED, (event) => {
            this.currentAOI = null;
            this.manager.currentAOI = null;
            console.log('🗑️ AOI Deleted');
            
            if (this.onAOIChange) {
                this.onAOIChange(false);
            }
            if (this.manager.onAOIChange) {
                this.manager.onAOIChange(false);
            }
            
            const legend = document.getElementById('map-legend');
            if (legend) {
                legend.classList.add('hidden');
            }
        });

        // Drawing start event
        map.on(L.Draw.Event.DRAWSTART, (event) => {
            console.log('🎨 Drawing started');
            const drawBtn = document.getElementById('draw-rectangle');
            if (drawBtn) {
                drawBtn.disabled = true;
                drawBtn.textContent = '✏️ Drawing...';
            }
        });

        // Drawing stop event  
        map.on(L.Draw.Event.DRAWSTOP, (event) => {
            console.log('🛑 Drawing stopped');
            
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
            this.manager.currentAOI = null;
            
            // Disable previous drawer if exists
            if (this.currentDrawer) {
                this.currentDrawer.disable();
                this.currentDrawer = null;
            }
            
            // Create a new rectangle drawing handler
            const rectangleDrawer = new L.Draw.Rectangle(this.manager.map, {
                shapeOptions: {
                    color: '#ff7800',
                    weight: 3,
                    opacity: 0.8,
                    fillColor: '#ff7800',
                    fillOpacity: 0.1,
                    pane: 'aoiPane'
                },
                showArea: true,
                metric: true,
                showLength: false,
                allowIntersection: false,
                repeatMode: false,
                drawError: {
                    color: '#b00b00',
                    timeout: 1000
                }
            });
            
            this.currentDrawer = rectangleDrawer;
            rectangleDrawer.enable();
            
            console.log('✅ AOI drawing mode enabled - click and drag on the map');
            
            const drawBtn = document.getElementById('draw-rectangle');
            if (drawBtn) {
                drawBtn.disabled = true;
                drawBtn.textContent = '✏️ Drawing...';
            }
            
            return rectangleDrawer;
            
        } catch (error) {
            console.error('❌ Error starting AOI drawing:', error);
            
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
        this.manager.currentAOI = null;
        this.aoiPolygons = [];
        this.manager.aoiPolygons = [];
        
        // Clear all image layers via layers module
        if (this.manager.layers) {
            this.manager.layers.clearImageLayers();
            this.manager.layers.clearAllAnalysisLayers();
            this.manager.layers.clearAllTileLayers();
        }
        
        if (this.onAOIChange) {
            this.onAOIChange(false);
        }
        if (this.manager.onAOIChange) {
            this.manager.onAOIChange(false);
        }
        
        const mapLegend = document.getElementById('map-legend');
        if (mapLegend) {
            mapLegend.classList.add('hidden');
        }
        
        // Clear analysis UI results
        this.clearAnalysisUI();
        
        console.log('🧹 All drawings and analysis results cleared');
    }
    
    clearAnalysisUI() {
        // Clear inline search results
        const resultContainers = ['search-results-s2', 'search-results-s1', 'search-results-emit'];
        resultContainers.forEach(id => {
            const container = document.getElementById(id);
            if (container) container.innerHTML = '';
        });
        
        // Hide inline results panels
        const resultPanels = ['inline-results-s2', 'inline-results-s1', 'inline-results-emit'];
        resultPanels.forEach(id => {
            const panel = document.getElementById(id);
            if (panel) panel.classList.add('hidden');
        });
        
        // Clear analysis grid
        const analysisGrid = document.getElementById('analysis-grid');
        if (analysisGrid) {
            analysisGrid.innerHTML = '<div class="no-analysis-results"><p>No analysis results yet. Process an image to see results here.</p></div>';
        }
        
        // Reset target detection if active
        if (window.platform?.targetDetection) {
            window.platform.targetDetection.reset();
        }
        
        // Reset threshold controller
        if (window.platform?.thresholdController) {
            window.platform.thresholdController.disableAll();
        }
        
        console.log('🧹 Analysis UI cleared');
    }

    getCurrentBounds() {
        // If we have multiple polygons, calculate combined bounds
        if (this.aoiPolygons && this.aoiPolygons.length > 0) {
            const combinedBounds = L.latLngBounds();
            this.aoiPolygons.forEach(layer => {
                if (layer.getBounds) {
                    combinedBounds.extend(layer.getBounds());
                } else if (layer.getLatLng) {
                    combinedBounds.extend(layer.getLatLng());
                }
            });
            if (combinedBounds.isValid()) {
                return [
                    combinedBounds.getWest(),
                    combinedBounds.getSouth(),
                    combinedBounds.getEast(),
                    combinedBounds.getNorth()
                ];
            }
        }
        
        if (this.currentAOI) {
            if (this.currentAOI.getBounds) {
                const bounds = this.currentAOI.getBounds();
                return [
                    bounds.getWest(),
                    bounds.getSouth(),
                    bounds.getEast(),
                    bounds.getNorth()
                ];
            } else if (this.currentAOI.getLatLng) {
                const latlng = this.currentAOI.getLatLng();
                const offset = 0.001;
                return [
                    latlng.lng - offset,
                    latlng.lat - offset,
                    latlng.lng + offset,
                    latlng.lat + offset
                ];
            }
        }
        
        // Fallback: check drawnItems
        if (this.drawnItems && this.drawnItems.getLayers().length > 0) {
            const bounds = this.drawnItems.getBounds();
            if (bounds.isValid()) {
                return [
                    bounds.getWest(),
                    bounds.getSouth(),
                    bounds.getEast(),
                    bounds.getNorth()
                ];
            }
        }
        
        return null;
    }

    getCombinedAOIBounds() {
        if (this.aoiPolygons && this.aoiPolygons.length > 0) {
            const combinedBounds = L.latLngBounds();
            this.aoiPolygons.forEach(layer => {
                if (layer.getBounds) {
                    combinedBounds.extend(layer.getBounds());
                } else if (layer.getLatLng) {
                    combinedBounds.extend(layer.getLatLng());
                }
            });
            return combinedBounds;
        }
        
        if (this.currentAOI) {
            if (this.currentAOI.getBounds) {
                return this.currentAOI.getBounds();
            } else if (this.currentAOI.getLatLng) {
                const latlng = this.currentAOI.getLatLng();
                const offset = 0.001;
                return L.latLngBounds(
                    [latlng.lat - offset, latlng.lng - offset],
                    [latlng.lat + offset, latlng.lng + offset]
                );
            }
        }
        return null;
    }

    getCurrentGeoJSON() {
        if (this.aoiPolygons && this.aoiPolygons.length > 0) {
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

    fitToAOI() {
        if (this.currentAOI) {
            try {
                console.log('Fitting map view to AOI for optimal display');
                
                const aoiBounds = this.getCombinedAOIBounds();
                
                this.manager.map.fitBounds(aoiBounds, {
                    padding: [20, 20],
                    maxZoom: 16,
                    animate: true,
                    duration: 1.0
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
                    this.aoiPolygons.forEach(polygon => {
                        if (polygon.setStyle) {
                            polygon.setStyle(aoiStyle);
                        }
                    });
                } else if (this.currentAOI && this.currentAOI.setStyle) {
                    this.currentAOI.setStyle(aoiStyle);
                }
                
                console.log('Map fitted to AOI bounds:', aoiBounds);
                
                setTimeout(() => {
                    const currentZoom = this.manager.map.getZoom();
                    console.log(`Current zoom level: ${currentZoom}`);
                }, 1100);
                
            } catch (error) {
                console.error('Error fitting to AOI:', error);
            }
        } else {
            console.warn('No AOI defined to fit to');
        }
    }

    outlineAOI() {
        const outlineStyle = {
            color: '#ff7800',
            weight: 3,
            opacity: 0.9,
            fillColor: '#ff7800',
            fillOpacity: 0.0
        };
        
        try {
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

    // Update AOI polygons from external source (e.g., file loader)
    setAOIFromLayers(layers) {
        this.aoiPolygons = layers;
        this.manager.aoiPolygons = layers;
        if (layers.length > 0) {
            this.currentAOI = layers[0];
            this.manager.currentAOI = layers[0];
        }
    }
}
