/**
 * Map Layers Module
 * Handles image, analysis, tile layers and COG rendering
 */

class MapLayers {
    constructor(manager) {
        this.manager = manager;
        this.imageLayers = {};
        this.selectedImageLayer = null;
        this.processedLayers = {};
        this.showProcessedImages = true;
        this.analysisLayers = {};
        this.tileLayers = {};
        this.cogCache = {};
        this.modelLayerCache = {};
        this.originalLayer = null;
    }

    _dispatchLayerEvent(eventName, detail) {
        try {
            window.dispatchEvent(new CustomEvent(eventName, { detail }));
        } catch (e) {
            console.warn(`[MapLayers] Event dispatch error (${eventName}):`, e);
        }
    }

    initialize() {
        // Store references in manager
        this.manager.imageLayers = this.imageLayers;
        this.manager.processedLayers = this.processedLayers;
        this.manager.analysisLayers = this.analysisLayers;
        this.manager.tileLayers = this.tileLayers;
        this.manager.cogCache = this.cogCache;
        this.manager.modelLayerCache = this.modelLayerCache;
        
        console.log('✅ Map Layers initialized');
    }

    /**
     * Create a tile layer with automatic retry logic for failed tiles
     * Falls back to backend proxy if direct connection fails repeatedly
     * @param {string} tileTemplate - Tile URL template
     * @param {object} options - L.tileLayer options
     * @param {string} layerId - Layer identifier for logging
     * @returns {L.TileLayer} - Tile layer with retry logic
     */
    _createTileLayerWithRetry(tileTemplate, options, layerId = 'unknown') {
        const failedTiles = new Map();
        const maxRetries = 2;
        const retryDelay = 1500;
        
        // Store template for proxy fallback
        const originalTemplate = tileTemplate;

        const tileLayer = L.tileLayer(tileTemplate, options);

        tileLayer.on('tileerror', (e) => {
            const tileKey = `${e.coords.z}/${e.coords.x}/${e.coords.y}`;
            const retryCount = failedTiles.get(tileKey) || 0;
            
            // Store original src on first failure
            if (retryCount === 0 && e.tile && e.tile.src) {
                e.tile._originalSrc = e.tile.src;
            }
            
            if (retryCount < maxRetries) {
                failedTiles.set(tileKey, retryCount + 1);
                console.warn(`[${layerId}] Tile ${tileKey} failed (${retryCount + 1}/${maxRetries}), retrying...`);
                
                setTimeout(() => {
                    if (e.tile) {
                        const originalSrc = e.tile._originalSrc || e.tile.src;
                        e.tile.src = '';
                        e.tile.src = originalSrc;
                    }
                }, retryDelay * (retryCount + 1));
            } else if (retryCount === maxRetries) {
                // Try proxy as last resort
                failedTiles.set(tileKey, retryCount + 1);
                console.warn(`[${layerId}] Tile ${tileKey} trying proxy fallback...`);
                
                setTimeout(() => {
                    if (e.tile) {
                        // Use backend proxy
                        const proxyUrl = `/api/tile-proxy/${e.coords.z}/${e.coords.x}/${e.coords.y}?url=${encodeURIComponent(originalTemplate)}`;
                        e.tile.src = '';
                        e.tile.src = proxyUrl;
                    }
                }, retryDelay);
            } else {
                console.error(`[${layerId}] Tile ${tileKey} failed completely`);
            }
        });

        tileLayer.on('tileload', (e) => {
            const tileKey = `${e.coords.z}/${e.coords.x}/${e.coords.y}`;
            if (failedTiles.has(tileKey)) {
                const count = failedTiles.get(tileKey);
                console.log(`[${layerId}] Tile ${tileKey} recovered after ${count} attempts`);
                failedTiles.delete(tileKey);
            }
        });

        return tileLayer;
    }

    addImageLayer(imageId, imageData) {
        try {
            console.log(`Adding AOI-cropped image layer ${imageId} to map`, imageData);
            
            if (!this.manager.drawing || !this.manager.drawing.currentAOI) {
                console.warn('No AOI defined for image display');
                return;
            }

            const aoiBounds = this.manager.drawing.getCombinedAOIBounds();
            const defaultBounds = [
                [aoiBounds.getSouth(), aoiBounds.getWest()],
                [aoiBounds.getNorth(), aoiBounds.getEast()]
            ];

            let bounds = defaultBounds;
            if (imageData.tile_bounds) {
                if (Array.isArray(imageData.tile_bounds) && imageData.tile_bounds.length === 4) {
                    const [west, south, east, north] = imageData.tile_bounds;
                    bounds = [[south, west], [north, east]];
                } else if (Array.isArray(imageData.tile_bounds) && imageData.tile_bounds.length === 2) {
                    bounds = imageData.tile_bounds;
                }
            }

            let imageLayer = null;
            
            if (imageData.display_type === 'tile' && imageData.tile_template) {
                console.log(`Creating dynamic tile layer for ${imageId}`);
                
                imageLayer = this._createTileLayerWithRetry(imageData.tile_template, {
                    opacity: 1.0,
                    attribution: '© Sentinel-2 via TiTiler',
                    bounds: bounds,
                    maxZoom: 18,
                    minZoom: 6,
                    tileSize: 256,
                    crossOrigin: 'anonymous',
                    pane: 'dataPane',
                    keepBuffer: 2,
                    updateWhenZooming: false,
                    updateWhenIdle: true,
                    // Limit concurrent tile requests to avoid GEE rate limiting (429 errors)
                    maxNativeZoom: 14,  // Don't request tiles above zoom 14
                }, imageId);
                
            } else {
                let imageUrl = null;
                
                if (imageData.display_url) {
                    imageUrl = imageData.display_url;
                } else if (imageData.assets && imageData.assets.thumbnail) {
                    imageUrl = imageData.assets.thumbnail;
                }

                if (imageUrl) {
                    imageLayer = L.imageOverlay(imageUrl, bounds, {
                        opacity: 1.0,
                        interactive: true,
                        crossOrigin: true,
                        pane: 'dataPane'
                    });

                    imageLayer.on('error', (e) => {
                        console.error(`Failed to load static image ${imageId}:`, e);
                        if (this.imageLayers[imageId]) {
                            this.manager.map.removeLayer(this.imageLayers[imageId]);
                            delete this.imageLayers[imageId];
                        }
                    });

                    imageLayer.on('load', () => {
                        console.log(`Successfully loaded static image ${imageId}`);
                    });
                }
            }

            if (imageLayer) {
                imageLayer.on('click', () => {
                    this.selectImageLayer(imageId);
                });

                const dateStr = imageData.datetime ? new Date(imageData.datetime).toLocaleDateString() : 'Unknown';
                const aoiOverlap = imageData.aoi_overlap ? (imageData.aoi_overlap * 100).toFixed(1) + '%' : 'N/A';
                const layerType = imageData.display_type === 'tile' ? 'Dynamic Tile Layer' : 'Static Image';
                
                this.imageLayers[imageId] = imageLayer;
                this.manager.imageLayers[imageId] = imageLayer;
                this.manager.map.addLayer(imageLayer);

                // Ensure AOI outline-only style is applied
                if (this.manager.drawing) {
                    this.manager.drawing.outlineAOI();
                }

                this._dispatchLayerEvent('layer:added', { id: imageId, type: 'image', name: imageData.datetime ? `Image ${new Date(imageData.datetime).toLocaleDateString()}` : `Image ${imageId}`, layer: imageLayer });

                console.log(`Successfully added layer: ${imageId}`);
                
                const legend = document.getElementById('map-legend');
                if (legend) {
                    legend.classList.remove('hidden');
                }
                
            } else {
                console.warn(`No valid image URL found for ${imageId}`);
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
                this.manager.map.removeLayer(layer);
            } catch (error) {
                console.error(`Error removing layer ${layerId}:`, error);
            }
        });

        this.imageLayers = {};
        this.manager.imageLayers = {};
        this.selectedImageLayer = null;

        this._dispatchLayerEvent('layers:cleared', { type: 'image' });
        console.log('All image layers cleared');
    }

    removeImageLayer(imageId) {
        if (this.imageLayers[imageId]) {
            try {
                this.manager.map.removeLayer(this.imageLayers[imageId]);
                delete this.imageLayers[imageId];
                delete this.manager.imageLayers[imageId];
                
                if (this.selectedImageLayer === imageId) {
                    this.selectedImageLayer = null;
                }

                this._dispatchLayerEvent('layer:removed', { id: imageId, type: 'image' });
                console.log(`Removed image layer: ${imageId}`);
            } catch (error) {
                console.error(`Error removing image layer ${imageId}:`, error);
            }
        }
    }

    selectImageLayer(imageId) {
        Object.keys(this.imageLayers).forEach(id => {
            const layer = this.imageLayers[id];
            if (id === imageId) {
                layer.setStyle ? layer.setStyle({opacity: 1.0}) : layer.setOpacity(1.0);
                this.selectedImageLayer = imageId;
                this.manager.selectedImageLayer = imageId;
                console.log(`Selected image layer: ${imageId}`);
            } else {
                layer.setStyle ? layer.setStyle({opacity: 0.6}) : layer.setOpacity(0.6);
            }
        });
    }

    showProcessedImage(imageUrl) {
        if (!this.manager.drawing || !this.manager.drawing.currentAOI) {
            console.warn('No AOI defined for processed image display');
            return;
        }

        try {
            Object.keys(this.processedLayers).forEach(id => {
                this.manager.map.removeLayer(this.processedLayers[id]);
                delete this.processedLayers[id];
            });

            const aoiBounds = this.manager.drawing.getCombinedAOIBounds();
            const bounds = [
                [aoiBounds.getSouth(), aoiBounds.getWest()],
                [aoiBounds.getNorth(), aoiBounds.getEast()]
            ];

            const processedLayer = L.imageOverlay(imageUrl, bounds, {
                opacity: 1.0,
                interactive: true,
                crossOrigin: true,
                pane: 'dataPane'
            });

            const processedId = 'processed_' + Date.now();
            this.processedLayers[processedId] = processedLayer;
            this.manager.processedLayers[processedId] = processedLayer;
            
            if (this.showProcessedImages) {
                this.manager.map.addLayer(processedLayer);
            }

            this._dispatchLayerEvent('layer:added', { id: processedId, type: 'processed', name: 'Processed Image', layer: processedLayer });
            console.log('Processed image displayed on map');

        } catch (error) {
            console.error('Error displaying processed image:', error);
        }
    }

    toggleProcessedImages() {
        this.showProcessedImages = !this.showProcessedImages;
        
        Object.keys(this.processedLayers).forEach(id => {
            const layer = this.processedLayers[id];
            if (this.showProcessedImages) {
                if (!this.manager.map.hasLayer(layer)) {
                    this.manager.map.addLayer(layer);
                }
            } else {
                if (this.manager.map.hasLayer(layer)) {
                    this.manager.map.removeLayer(layer);
                }
            }
        });
        
        return this.showProcessedImages;
    }

    showAnalysisLayer(modelId, imageUrl, modelName, boundsOverride = null) {
        if (!boundsOverride && (!this.manager.drawing || !this.manager.drawing.currentAOI)) {
            console.warn('No AOI defined for analysis layer display');
            return;
        }

        try {
            this.hideAnalysisLayer(modelId);

            let bounds;
            if (boundsOverride && Array.isArray(boundsOverride) && boundsOverride.length === 2) {
                bounds = boundsOverride;
            } else {
                const aoiBounds = this.manager.drawing.getCombinedAOIBounds();
                bounds = [
                    [aoiBounds.getSouth(), aoiBounds.getWest()],
                    [aoiBounds.getNorth(), aoiBounds.getEast()]
                ];
            }

            const analysisLayer = L.imageOverlay(imageUrl, bounds, {
                opacity: 1.0,
                interactive: true,
                crossOrigin: true,
                pane: 'dataPane'
            });

            analysisLayer.on('error', (e) => {
                console.error(`Failed to load analysis overlay ${modelId}:`, e);
            });
            analysisLayer.on('load', () => {
                console.log(`Analysis overlay ${modelId} loaded successfully`);
            });

            this.analysisLayers[modelId] = analysisLayer;
            this.manager.analysisLayers[modelId] = analysisLayer;
            this.manager.map.addLayer(analysisLayer);

            this._dispatchLayerEvent('layer:added', { id: modelId, type: 'analysis', name: modelName || modelId, layer: analysisLayer });
            console.log(`Analysis layer ${modelId} displayed on map`);

        } catch (error) {
            console.error(`Error displaying analysis layer ${modelId}:`, error);
        }
    }

    hideAnalysisLayer(modelId) {
        if (this.analysisLayers[modelId]) {
            try {
                this.manager.map.removeLayer(this.analysisLayers[modelId]);
                delete this.analysisLayers[modelId];
                delete this.manager.analysisLayers[modelId];
                this._dispatchLayerEvent('layer:removed', { id: modelId, type: 'analysis' });
                console.log(`Analysis layer ${modelId} removed from map`);
            } catch (error) {
                console.error(`Error removing analysis layer ${modelId}:`, error);
            }
        } else if (this.modelLayerCache[modelId]) {
            try {
                const cachedLayer = this.modelLayerCache[modelId];
                if (this.manager.map.hasLayer(cachedLayer)) {
                    this.manager.map.removeLayer(cachedLayer);
                }
                this._dispatchLayerEvent('layer:removed', { id: modelId, type: 'analysis' });
                console.log(`Cached analysis layer ${modelId} removed from map`);
            } catch (error) {
                console.error(`Error removing cached analysis layer ${modelId}:`, error);
            }
        }
    }

    clearAllAnalysisLayers() {
        console.log('Clearing all analysis layers...');
        
        // Clear regular analysis layers
        Object.keys(this.analysisLayers).forEach(modelId => {
            try {
                this.manager.map.removeLayer(this.analysisLayers[modelId]);
            } catch (error) {
                console.error(`Error removing analysis layer ${modelId}:`, error);
            }
        });
        this.analysisLayers = {};
        this.manager.analysisLayers = {};

        // Clear cached model layers
        Object.keys(this.modelLayerCache).forEach(modelId => {
            try {
                const layer = this.modelLayerCache[modelId];
                if (this.manager.map.hasLayer(layer)) {
                    this.manager.map.removeLayer(layer);
                }
            } catch (error) {
                console.error(`Error removing cached layer ${modelId}:`, error);
            }
        });
        this.modelLayerCache = {};
        this.manager.modelLayerCache = {};
        
        // Clear processed layers
        Object.keys(this.processedLayers).forEach(id => {
            try {
                this.manager.map.removeLayer(this.processedLayers[id]);
            } catch (error) {
                console.error(`Error removing processed layer ${id}:`, error);
            }
        });
        this.processedLayers = {};
        this.manager.processedLayers = {};
        
        // Clear COG cache
        this.cogCache = {};
        this.manager.cogCache = {};

        this._dispatchLayerEvent('layers:cleared', { type: 'analysis' });
        console.log('All analysis layers cleared');
    }

    clearAllTileLayers() {
        console.log('Clearing all tile layers...');
        
        Object.keys(this.tileLayers).forEach(layerId => {
            try {
                this.manager.map.removeLayer(this.tileLayers[layerId]);
            } catch (error) {
                console.error(`Error removing tile layer ${layerId}:`, error);
            }
        });
        
        this.tileLayers = {};
        this.manager.tileLayers = {};

        this._dispatchLayerEvent('layers:cleared', { type: 'tile' });
        console.log('All tile layers cleared');
    }

    showTileLayer(layerId, tileTemplate, bounds, tooltip) {
        try {
            this.hideTileLayer(layerId);

            const layerBounds = bounds ? L.latLngBounds(bounds) : undefined;
            const opts = {
                attribution: 'Google Earth Engine',
                opacity: 1.0,
                maxZoom: 18,
                maxNativeZoom: 14,
                pane: 'dataPane',
                crossOrigin: 'anonymous',
                keepBuffer: 2,
                updateWhenZooming: false,
                updateWhenIdle: true,
            };
            if (layerBounds) opts.bounds = layerBounds;
            const tileLayer = this._createTileLayerWithRetry(tileTemplate, opts, layerId);

            this.tileLayers[layerId] = tileLayer;
            this.manager.tileLayers[layerId] = tileLayer;
            this.manager.map.addLayer(tileLayer);

            this._dispatchLayerEvent('layer:added', { id: layerId, type: 'tile', name: tooltip || `Tile ${layerId}`, layer: tileLayer });
            console.log(`Tile layer ${layerId} displayed on map`);

        } catch (error) {
            console.error(`Error displaying tile layer ${layerId}:`, error);
        }
    }

    hideTileLayer(layerId) {
        if (this.tileLayers[layerId]) {
            try {
                this.manager.map.removeLayer(this.tileLayers[layerId]);
                delete this.tileLayers[layerId];
                delete this.manager.tileLayers[layerId];
                this._dispatchLayerEvent('layer:removed', { id: layerId, type: 'tile' });
                console.log(`Tile layer ${layerId} removed from map`);
            } catch (error) {
                console.error(`Error removing tile layer ${layerId}:`, error);
            }
        }
    }

    /**
     * Force refresh all visible tile layers (reload all tiles)
     */
    refreshAllTileLayers() {
        Object.entries(this.tileLayers).forEach(([layerId, layer]) => {
            if (layer && layer.redraw) {
                console.log(`Refreshing tile layer: ${layerId}`);
                layer.redraw();
            }
        });
    }

    /**
     * Refresh a specific tile layer
     */
    refreshTileLayer(layerId) {
        const layer = this.tileLayers[layerId];
        if (layer && layer.redraw) {
            console.log(`Refreshing tile layer: ${layerId}`);
            layer.redraw();
        }
    }

    // COG Methods
    async showOriginalCOG(signedUrl, options = {opacity: 0.9}) {
        try {
            const georaster = await this._ensureCOG(signedUrl);
            if (this.originalLayer) {
                this.manager.map.removeLayer(this.originalLayer);
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
            layer.addTo(this.manager.map);
            
            if (this.manager.drawing) {
                this.manager.drawing.fitToAOI();
                this.manager.drawing.outlineAOI();
            }
            
            console.log('Original COG displayed (cached)');
        } catch (e) {
            console.error('Failed to display original COG:', e);
        }
    }

    async showModelFromCOG(modelId, signedUrl, options = {opacity: 0.7}) {
        try {
            if (this.modelLayerCache[modelId]) {
                const cachedLayer = this.modelLayerCache[modelId];
                this.analysisLayers[modelId] = cachedLayer;
                this.manager.analysisLayers[modelId] = cachedLayer;
                if (!this.manager.map.hasLayer(cachedLayer)) {
                    this.manager.map.addLayer(cachedLayer);
                }
                this._dispatchLayerEvent('layer:added', { id: modelId, type: 'analysis', name: `Model ${modelId}`, layer: cachedLayer });
                console.log(`Model ${modelId} layer restored from cache`);
                return;
            }
            
            const georaster = await this._ensureCOG(signedUrl);
            this.hideAnalysisLayer(modelId);
            const pf = this._getPixelFnForModel(modelId);
            const LayerCtor = window.GeoRasterLayer || (window.georaster && window.georaster.GeoRasterLayer) || (window['georasterLayerForLeaflet'] && window['georasterLayerForLeaflet'].GeoRasterLayer);
            if (!LayerCtor) throw new Error('GeoRasterLayer not loaded');
            
            const layer = new LayerCtor({ 
                georaster, 
                opacity: options.opacity !== undefined ? options.opacity : 0.7, 
                resolution: 256, 
                pixelValuesToColorFn: pf 
            });
            
            this.analysisLayers[modelId] = layer;
            this.manager.analysisLayers[modelId] = layer;
            this.modelLayerCache[modelId] = layer;
            this.manager.modelLayerCache[modelId] = layer;
            layer.addTo(this.manager.map);

            this._dispatchLayerEvent('layer:added', { id: modelId, type: 'analysis', name: `Model ${modelId}`, layer });
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
                this.manager.analysisLayers[modelId] = cachedLayer;
                if (!this.manager.map.hasLayer(cachedLayer)) {
                    this.manager.map.addLayer(cachedLayer);
                }
                this._dispatchLayerEvent('layer:added', { id: modelId, type: 'analysis', name: `Binary Mask ${modelId}`, layer: cachedLayer });
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
            
            const layer = new LayerCtor({ 
                georaster, 
                opacity: options.opacity !== undefined ? options.opacity : 1.0, 
                resolution: 256, 
                pixelValuesToColorFn 
            });
            
            this.analysisLayers[modelId] = layer;
            this.manager.analysisLayers[modelId] = layer;
            this.modelLayerCache[modelId] = layer;
            this.manager.modelLayerCache[modelId] = layer;
            layer.addTo(this.manager.map);

            this._dispatchLayerEvent('layer:added', { id: modelId, type: 'analysis', name: `Binary Mask ${modelId}`, layer });
            console.log(`Binary mask ${modelId} from COG displayed (cached)`);
        } catch (e) {
            console.error('Failed to display binary mask COG:', e);
        }
    }

    async prepareModelLayerFromCOG(modelId, signedUrl, options = {opacity: 0.7}) {
        try {
            const georaster = await this._ensureCOG(signedUrl);
            if (this.modelLayerCache[modelId]) return;
            
            const pf = this._getPixelFnForModel(modelId);
            const LayerCtor = window.GeoRasterLayer || (window.georaster && window.georaster.GeoRasterLayer) || (window['georasterLayerForLeaflet'] && window['georasterLayerForLeaflet'].GeoRasterLayer);
            if (!LayerCtor) throw new Error('GeoRasterLayer not loaded');
            
            const layer = new LayerCtor({ 
                georaster, 
                opacity: options.opacity !== undefined ? options.opacity : 0.7, 
                resolution: 256, 
                pixelValuesToColorFn: pf 
            });
            
            this.modelLayerCache[modelId] = layer;
            this.manager.modelLayerCache[modelId] = layer;
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
            
            const layer = new LayerCtor({ 
                georaster, 
                opacity: options.opacity !== undefined ? options.opacity : 1.0, 
                resolution: 256, 
                pixelValuesToColorFn 
            });
            
            this.modelLayerCache[modelId] = layer;
            this.manager.modelLayerCache[modelId] = layer;
            console.log(`Binary mask ${modelId} layer prepared and cached`);
        } catch (e) {
            console.warn(`Failed to prepare binary mask ${modelId} layer:`, e);
        }
    }

    _getPixelFnForModel(modelId) {
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
            return values => `rgb(${toByte(values[3])}, ${toByte(values[0])}, ${toByte(values[1])})`;
        }
        if (modelId === 'model3') {
            return values => `rgb(${toByte(values[5])}, ${toByte(values[4])}, ${toByte(values[0])})`;
        }
        if (modelId === 'model4') {
            return values => `rgb(${toByte(values[3])}, ${toByte(values[0])}, ${toByte(values[2])})`;
        }
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
        this.manager.cogCache[url] = georaster;
        return georaster;
    }
}
