/**
 * ===== MAP MANAGER =====
 * Google Maps Style Map Controller
 * Facade class that composes modular components
 */

class MapManager {
    constructor(mapId) {
        this.mapId = mapId;
        
        // Properties to be set by modules
        this.map = null;
        this.drawnItems = null;
        this.currentAOI = null;
        this.aoiPolygons = [];
        this.imageLayers = {};
        this.selectedImageLayer = null;
        this.onAOIChange = null;
        this.baseLayer = null;
        this.processedLayers = {};
        this.showProcessedImages = true;
        this.analysisLayers = {};
        this.cogCache = {};
        this.modelLayerCache = {};
        this.currentBaseLayerType = 'osm';
        this.tileLayers = {};
        
        // Base layer references
        this.osmLayer = null;
        this.satelliteLayer = null;
        this.labelsLayer = null;
        
        // Initialize modules
        this.core = new MapCore(mapId, this);
        this.drawing = null;
        this.layers = null;
        this.fileLoader = null;
        
        // Start initialization
        this.initializeMap();
    }

    initializeMap() {
        // Initialize core (creates the map)
        this.core.initializeMap();
        
        // Wait for map to be ready, then initialize other modules
        if (this.map) {
            this._initializeSubModules();
        }
    }

    _initializeSubModules() {
        // Initialize drawing module
        this.drawing = new MapDrawing(this);
        this.drawing.initialize();
        
        // Initialize layers module
        this.layers = new MapLayers(this);
        this.layers.initialize();
        
        // Initialize file loader module
        this.fileLoader = new MapFileLoader(this);
        
        console.log('✅ Map Manager Ready!');
    }

    // ===== Delegation methods for backward compatibility =====

    // Core methods
    switchToOSM() {
        this.core.switchToOSM();
    }

    switchToSatellite() {
        this.core.switchToSatellite();
    }

    bringDataLayersToFront() {
        this.core.bringDataLayersToFront();
    }

    getMapBounds() {
        return this.core.getMapBounds();
    }

    getMapCenter() {
        return this.core.getMapCenter();
    }

    getMapZoom() {
        return this.core.getMapZoom();
    }

    setMapView(center, zoom) {
        this.core.setMapView(center, zoom);
    }

    panTo(latLng) {
        this.core.panTo(latLng);
    }

    // Drawing methods
    startDrawing() {
        return this.drawing.startDrawing();
    }

    clearDrawings() {
        this.drawing.clearDrawings();
    }

    getCurrentBounds() {
        return this.drawing.getCurrentBounds();
    }

    getCombinedAOIBounds() {
        return this.drawing.getCombinedAOIBounds();
    }

    getCurrentGeoJSON() {
        return this.drawing.getCurrentGeoJSON();
    }

    fitToAOI() {
        this.drawing.fitToAOI();
    }

    outlineAOI() {
        this.drawing.outlineAOI();
    }

    // Layer methods
    addImageLayer(imageId, imageData) {
        this.layers.addImageLayer(imageId, imageData);
    }

    clearImageLayers() {
        this.layers.clearImageLayers();
    }

    removeImageLayer(imageId) {
        this.layers.removeImageLayer(imageId);
    }

    selectImageLayer(imageId) {
        this.layers.selectImageLayer(imageId);
    }

    showProcessedImage(imageUrl) {
        this.layers.showProcessedImage(imageUrl);
    }

    toggleProcessedImages() {
        return this.layers.toggleProcessedImages();
    }

    showAnalysisLayer(modelId, imageUrl, modelName, boundsOverride = null, isBinary = false) {
        return this.layers.showAnalysisLayer(modelId, imageUrl, modelName, boundsOverride, isBinary);
    }

    showAnalysisLayerAsync(modelId, imageUrl, modelName, boundsOverride = null, timeoutMs, isBinary = false) {
        return this.layers.showAnalysisLayerAsync(modelId, imageUrl, modelName, boundsOverride, timeoutMs, isBinary);
    }

    hideAnalysisLayer(modelId) {
        this.layers.hideAnalysisLayer(modelId);
    }

    showTileLayer(layerId, tileTemplate, bounds, tooltip) {
        this.layers.showTileLayer(layerId, tileTemplate, bounds, tooltip);
    }

    hideTileLayer(layerId) {
        this.layers.hideTileLayer(layerId);
    }

    applyRenderingFilter(tileLayer, state, prefix) {
        this.layers.applyRenderingFilter(tileLayer, state, prefix);
    }

    reapplyRenderingFilter(tileLayer) {
        this.layers.reapplyRenderingFilter(tileLayer);
    }

    // COG methods
    async showOriginalCOG(signedUrl, options) {
        return this.layers.showOriginalCOG(signedUrl, options);
    }

    async showModelFromCOG(modelId, signedUrl, options) {
        return this.layers.showModelFromCOG(modelId, signedUrl, options);
    }

    async showBinaryMaskCOG(modelId, signedUrl, options) {
        return this.layers.showBinaryMaskCOG(modelId, signedUrl, options);
    }

    async prepareModelLayerFromCOG(modelId, signedUrl, options) {
        return this.layers.prepareModelLayerFromCOG(modelId, signedUrl, options);
    }

    async prepareBinaryMaskLayerFromCOG(modelId, signedUrl, options) {
        return this.layers.prepareBinaryMaskLayerFromCOG(modelId, signedUrl, options);
    }

    async prefetchCOG(url) {
        return this.layers.prefetchCOG(url);
    }

    // File loader methods
    async loadShpFile(file) {
        return this.fileLoader.loadShpFile(file);
    }

    async loadKmlFile(file) {
        return this.fileLoader.loadKmlFile(file);
    }
}

// MapManager class ends here - initialization is now handled in index.html
