/**
 * Dual Map Controller
 * Manages split-screen compare mode with two synchronized Leaflet maps
 */

class DualMapController {
    constructor(platformController) {
        this.platform = platformController;
        this.isActive = false;
        this.secondaryMap = null;
        this.secondaryBaseLayers = null;
        this.secondaryMapLayers = {};  // layerId -> cloned layer on secondary map
        this.leftLayerIds = new Set();
        this.rightLayerIds = new Set();
        this._syncing = false;
        this._syncHandlers = {};
        this._pendingReassignment = {};        // exact id -> {left, right}
        this._pendingPrefixReassignment = {};  // prefix ('td', 'sam2-preview') -> {left, right}

        // Listen for base layer changes and window resize
        this._onBaseLayerChanged = (e) => this._handleBaseLayerChange(e.detail);
        this._onResize = () => this._handleResize();
        window.addEventListener('baselayer:changed', this._onBaseLayerChanged);
        window.addEventListener('resize', this._onResize);

        // Listen for real-time layer changes to update compare mode UI
        window.addEventListener('layer:added', (e) => {
            try { this._handleLayerAdded(e.detail); } catch (err) { console.warn('[DualMap] layer:added error:', err); }
        });
        window.addEventListener('layer:removed', (e) => {
            try { this._handleLayerRemoved(e.detail); } catch (err) { console.warn('[DualMap] layer:removed error:', err); }
        });
        window.addEventListener('layers:cleared', (e) => {
            try { this._handleLayersCleared(e.detail); } catch (err) { console.warn('[DualMap] layers:cleared error:', err); }
        });
    }

    get primaryMap() {
        return window.mapManager?.map;
    }

    get mapCore() {
        return window.mapManager?.core;
    }

    get layerPanel() {
        return this.platform?.layerControlPanel;
    }

    toggle() {
        if (this.isActive) {
            this.deactivate();
        } else {
            this.activate();
        }
    }

    activate() {
        if (this.isActive || !this.primaryMap) return;

        const layers = this.layerPanel?.getCurrentLayers() || [];
        if (layers.length < 2) return;

        this.isActive = true;

        // Split map container
        const mapContainer = document.getElementById('map-container');
        const rightContainer = document.getElementById('map-container-right');
        const dualControls = document.getElementById('dual-map-controls');

        mapContainer.classList.add('split-mode');
        rightContainer.style.display = '';
        if (dualControls) dualControls.style.display = '';

        // Invalidate primary map after resize
        setTimeout(() => this.primaryMap.invalidateSize(), 50);

        // Create secondary map
        this._createSecondaryMap();

        // Default assignment: all layers on left, none on right initially
        this.leftLayerIds.clear();
        this.rightLayerIds.clear();
        layers.forEach(l => this.leftLayerIds.add(l.id));

        // Render the layer assignment UI
        this._renderAssignmentUI();

        // Set up synchronization
        this._setupSync();

        // Update button text
        this._updateCompareButtonText(true);

        // Notify other controllers (e.g. target detection marker sync)
        window.dispatchEvent(new CustomEvent('comparemap:activated', { detail: { secondaryMap: this.secondaryMap } }));
    }

    deactivate() {
        if (!this.isActive) return;

        this.isActive = false;

        // Notify before destroying
        window.dispatchEvent(new CustomEvent('comparemap:deactivated'));

        // Remove sync handlers
        this._removeSync();

        // Remove cloned layers and destroy secondary map
        this._destroySecondaryMap();

        // Restore layers to primary map
        this._restoreLayersToPrimary();

        // Restore map container
        const mapContainer = document.getElementById('map-container');
        const rightContainer = document.getElementById('map-container-right');
        const dualControls = document.getElementById('dual-map-controls');

        mapContainer.classList.remove('split-mode');
        rightContainer.style.display = 'none';
        if (dualControls) dualControls.style.display = 'none';

        // Invalidate primary map
        setTimeout(() => {
            if (this.primaryMap) this.primaryMap.invalidateSize();
        }, 50);

        this.leftLayerIds.clear();
        this.rightLayerIds.clear();
        this.secondaryMapLayers = {};
        this._pendingReassignment = {};
        this._pendingPrefixReassignment = {};

        this._updateCompareButtonText(false);
    }

    _createSecondaryMap() {
        const center = this.primaryMap.getCenter();
        const zoom = this.primaryMap.getZoom();

        this.secondaryMap = L.map('map-right', {
            center: center,
            zoom: zoom,
            minZoom: 3,
            maxZoom: 18,
            zoomControl: false,
            attributionControl: false,
            maxBounds: [[-90, -180], [90, 180]],
            maxBoundsViscosity: 1.0
        });

        // Create panes
        this.secondaryMap.createPane('basemapPane');
        this.secondaryMap.getPane('basemapPane').style.zIndex = 100;
        this.secondaryMap.createPane('dataPane');
        this.secondaryMap.getPane('dataPane').style.zIndex = 450;
        this.secondaryMap.createPane('labelsPane');
        this.secondaryMap.getPane('labelsPane').style.zIndex = 650;

        // Add base layer matching primary
        const baseType = this.mapCore?.currentBaseLayerType || 'osm';
        this.secondaryBaseLayers = this.mapCore.createBaseLayerClone(baseType);
        this.secondaryBaseLayers.base.addTo(this.secondaryMap);
        if (this.secondaryBaseLayers.labels) {
            this.secondaryBaseLayers.labels.addTo(this.secondaryMap);
        }

        setTimeout(() => this.secondaryMap.invalidateSize(), 100);
    }

    _destroySecondaryMap() {
        if (this.secondaryMap) {
            // Remove all cloned layers
            Object.values(this.secondaryMapLayers).forEach(layer => {
                try { this.secondaryMap.removeLayer(layer); } catch (e) {}
            });

            this.secondaryMap.remove();
            this.secondaryMap = null;
            this.secondaryBaseLayers = null;
        }
    }

    _setupSync() {
        const onPrimaryMove = () => {
            if (this._syncing || !this.secondaryMap) return;
            this._syncing = true;
            this.secondaryMap.setView(this.primaryMap.getCenter(), this.primaryMap.getZoom(), { animate: false });
            this._syncing = false;
        };

        const onSecondaryMove = () => {
            if (this._syncing || !this.primaryMap) return;
            this._syncing = true;
            this.primaryMap.setView(this.secondaryMap.getCenter(), this.secondaryMap.getZoom(), { animate: false });
            this._syncing = false;
        };

        this.primaryMap.on('move', onPrimaryMove);
        this.primaryMap.on('zoom', onPrimaryMove);

        if (this.secondaryMap) {
            this.secondaryMap.on('move', onSecondaryMove);
            this.secondaryMap.on('zoom', onSecondaryMove);
        }

        this._syncHandlers = { onPrimaryMove, onSecondaryMove };
    }

    _removeSync() {
        if (this._syncHandlers.onPrimaryMove && this.primaryMap) {
            this.primaryMap.off('move', this._syncHandlers.onPrimaryMove);
            this.primaryMap.off('zoom', this._syncHandlers.onPrimaryMove);
        }
        if (this._syncHandlers.onSecondaryMove && this.secondaryMap) {
            this.secondaryMap.off('move', this._syncHandlers.onSecondaryMove);
            this.secondaryMap.off('zoom', this._syncHandlers.onSecondaryMove);
        }
        this._syncHandlers = {};
    }

    /**
     * Clone a layer for the secondary map
     */
    _cloneLayer(entry) {
        const layer = entry.leafletLayer;
        if (!layer) return null;

        try {
            // L.imageOverlay
            if (layer instanceof L.ImageOverlay) {
                return L.imageOverlay(layer._url, layer.getBounds(), {
                    opacity: layer.options.opacity || 1.0,
                    pane: 'dataPane',
                    crossOrigin: true
                });
            }

            // L.tileLayer
            if (layer instanceof L.TileLayer) {
                return L.tileLayer(layer._url, {
                    ...layer.options,
                    pane: 'dataPane'
                });
            }

            // GeoRasterLayer — check for georaster data
            if (layer.options && layer.options.georaster) {
                const LayerCtor = window.GeoRasterLayer ||
                    (window.georaster && window.georaster.GeoRasterLayer) ||
                    (window['georasterLayerForLeaflet'] && window['georasterLayerForLeaflet'].GeoRasterLayer);
                if (LayerCtor) {
                    return new LayerCtor({
                        georaster: layer.options.georaster,
                        opacity: layer.options.opacity || 0.7,
                        resolution: layer.options.resolution || 256,
                        pixelValuesToColorFn: layer.options.pixelValuesToColorFn,
                        pane: 'dataPane'
                    });
                }
            }
        } catch (e) {
            console.warn(`Failed to clone layer ${entry.id}:`, e);
        }

        return null;
    }

    /**
     * Assign a layer to a specific map side
     */
    assignLayer(layerId, side) {
        const layers = this.layerPanel?.getCurrentLayers() || [];
        const entry = layers.find(l => l.id === layerId);
        if (!entry) return;

        // Remove this layer from both sides first
        this.leftLayerIds.delete(layerId);
        this.rightLayerIds.delete(layerId);

        // Remove clone from secondary if exists
        if (this.secondaryMapLayers[layerId]) {
            try { this.secondaryMap.removeLayer(this.secondaryMapLayers[layerId]); } catch (e) {}
            delete this.secondaryMapLayers[layerId];
        }

        // Remove from primary if needed
        if (this.primaryMap.hasLayer(entry.leafletLayer)) {
            this.primaryMap.removeLayer(entry.leafletLayer);
        }

        if (side === 'left') {
            this._clearSide('left');
            this.leftLayerIds.add(layerId);
            if (!this.primaryMap.hasLayer(entry.leafletLayer)) {
                this.primaryMap.addLayer(entry.leafletLayer);
            }
        } else if (side === 'right') {
            this._clearSide('right');
            this.rightLayerIds.add(layerId);
            const clone = this._cloneLayer(entry);
            if (clone && this.secondaryMap) {
                this.secondaryMapLayers[layerId] = clone;
                this.secondaryMap.addLayer(clone);
            }
        } else if (side === 'both') {
            this._clearSide('left');
            this._clearSide('right');
            this.leftLayerIds.add(layerId);
            this.rightLayerIds.add(layerId);
            if (!this.primaryMap.hasLayer(entry.leafletLayer)) {
                this.primaryMap.addLayer(entry.leafletLayer);
            }
            const clone = this._cloneLayer(entry);
            if (clone && this.secondaryMap) {
                this.secondaryMapLayers[layerId] = clone;
                this.secondaryMap.addLayer(clone);
            }
        }

        this._renderAssignmentUI();
    }

    /**
     * Remove all layers from one side (exclusive selection helper)
     */
    _clearSide(side) {
        const layers = this.layerPanel?.getCurrentLayers() || [];
        if (side === 'left') {
            for (const id of [...this.leftLayerIds]) {
                this.leftLayerIds.delete(id);
                const entry = layers.find(l => l.id === id);
                if (entry && this.primaryMap.hasLayer(entry.leafletLayer)) {
                    this.primaryMap.removeLayer(entry.leafletLayer);
                }
            }
        } else {
            for (const id of [...this.rightLayerIds]) {
                this.rightLayerIds.delete(id);
                if (this.secondaryMapLayers[id]) {
                    try { this.secondaryMap.removeLayer(this.secondaryMapLayers[id]); } catch (e) {}
                    delete this.secondaryMapLayers[id];
                }
            }
        }
    }

    _restoreLayersToPrimary() {
        const layers = this.layerPanel?.getCurrentLayers() || [];
        layers.forEach(entry => {
            if (entry.visible && !this.primaryMap.hasLayer(entry.leafletLayer)) {
                this.primaryMap.addLayer(entry.leafletLayer);
            }
        });
    }

    _renderAssignmentUI() {
        const leftList = document.getElementById('left-map-layers');
        const rightList = document.getElementById('right-map-layers');
        if (!leftList || !rightList) return;

        const layers = this.layerPanel?.getCurrentLayers() || [];

        leftList.innerHTML = '';
        rightList.innerHTML = '';

        if (layers.length === 0) {
            leftList.innerHTML = '<div class="dual-layer-empty">No layers</div>';
            rightList.innerHTML = '<div class="dual-layer-empty">No layers</div>';
            return;
        }

        layers.forEach(entry => {
            const isLeft = this.leftLayerIds.has(entry.id);
            const isRight = this.rightLayerIds.has(entry.id);

            // Create items for both lists
            const leftItem = this._createAssignmentItem(entry, 'left', isLeft);
            const rightItem = this._createAssignmentItem(entry, 'right', isRight);

            leftList.appendChild(leftItem);
            rightList.appendChild(rightItem);
        });
    }

    _createAssignmentItem(entry, side, isActive) {
        const item = document.createElement('div');
        item.className = 'dual-layer-item' + (isActive ? ' active' : '');
        item.innerHTML = `
            <span class="layer-type-icon">${this._getTypeIcon(entry.type)}</span>
            <span class="layer-name">${entry.name}</span>
        `;

        item.addEventListener('click', () => {
            if (isActive) {
                // Deselect: remove from this side
                if (side === 'left') {
                    this.leftLayerIds.delete(entry.id);
                    if (this.primaryMap.hasLayer(entry.leafletLayer)) {
                        this.primaryMap.removeLayer(entry.leafletLayer);
                    }
                } else {
                    this.rightLayerIds.delete(entry.id);
                    if (this.secondaryMapLayers[entry.id]) {
                        try { this.secondaryMap.removeLayer(this.secondaryMapLayers[entry.id]); } catch (e) {}
                        delete this.secondaryMapLayers[entry.id];
                    }
                }
            } else {
                // Exclusive selection: clear side first, then add this layer
                this._clearSide(side);
                if (side === 'left') {
                    this.leftLayerIds.add(entry.id);
                    if (!this.primaryMap.hasLayer(entry.leafletLayer)) {
                        this.primaryMap.addLayer(entry.leafletLayer);
                    }
                } else {
                    this.rightLayerIds.add(entry.id);
                    const clone = this._cloneLayer(entry);
                    if (clone && this.secondaryMap) {
                        this.secondaryMapLayers[entry.id] = clone;
                        this.secondaryMap.addLayer(clone);
                    }
                }
            }
            this._renderAssignmentUI();
        });

        return item;
    }

    _getTypeIcon(type) {
        switch (type) {
            case 'analysis': return '📊';
            case 'tile': return '🗺️';
            case 'image': return '🛰️';
            case 'processed': return '🖼️';
            default: return '📄';
        }
    }

    /**
     * Get the prefix category for live-updating layers.
     * td-xxx → 'td', sam2-preview-xxx → 'sam2-preview'
     * Used to match reassignment when IDs change between remove/add cycles.
     */
    _getLayerPrefix(id) {
        if (id.startsWith('td-')) return 'td';
        if (id.startsWith('sam2-preview-')) return 'sam2-preview';
        return null;
    }

    /**
     * When a new layer is added while in compare mode,
     * restore previous assignment by exact ID or prefix match.
     */
    _handleLayerAdded(detail) {
        if (!this.isActive || !detail || !detail.id) return;

        const id = detail.id;
        const prefix = this._getLayerPrefix(id);

        // Check exact ID match first, then prefix match
        let pending = this._pendingReassignment[id];
        if (!pending && prefix && this._pendingPrefixReassignment[prefix]) {
            pending = this._pendingPrefixReassignment[prefix];
            delete this._pendingPrefixReassignment[prefix];
        }

        if (pending) {
            // Restore previous left/right assignment (exclusive per side)
            if (pending._timer) clearTimeout(pending._timer);
            if (pending._prefixTimer) clearTimeout(pending._prefixTimer);
            delete this._pendingReassignment[id];

            if (pending.left) {
                this._clearSide('left');
                this.leftLayerIds.add(id);
            }
            if (pending.right) {
                this._clearSide('right');
                this.rightLayerIds.add(id);
                // Re-clone for the secondary map using the NEW layer from the event
                const fakeEntry = { id, leafletLayer: detail.layer, type: detail.type };
                const clone = this._cloneLayer(fakeEntry);
                if (clone && this.secondaryMap) {
                    this.secondaryMapLayers[id] = clone;
                    this.secondaryMap.addLayer(clone);
                }
            }
        } else if (!this.leftLayerIds.has(id) && !this.rightLayerIds.has(id)) {
            // Brand new layer — default to left (primary) map, exclusive
            this._clearSide('left');
            this.leftLayerIds.add(id);
        }

        this._renderAssignmentUI();
    }

    /**
     * When a layer is removed while in compare mode,
     * save assignment by exact ID AND by prefix (for live-updating layers like td-, sam2-).
     */
    _handleLayerRemoved(detail) {
        if (!this.isActive || !detail || !detail.id) return;

        const id = detail.id;
        const wasLeft = this.leftLayerIds.has(id);
        const wasRight = this.rightLayerIds.has(id);

        if (wasLeft || wasRight) {
            const assignment = { left: wasLeft, right: wasRight };

            // Save by exact ID
            const pending = { ...assignment };
            pending._timer = setTimeout(() => {
                delete this._pendingReassignment[id];
            }, 500);
            this._pendingReassignment[id] = pending;

            // Also save by prefix for layers with changing IDs (td-xxx, sam2-preview-xxx)
            const prefix = this._getLayerPrefix(id);
            if (prefix) {
                const prefixPending = { ...assignment };
                prefixPending._prefixTimer = setTimeout(() => {
                    delete this._pendingPrefixReassignment[prefix];
                }, 500);
                this._pendingPrefixReassignment[prefix] = prefixPending;
            }
        }

        this.leftLayerIds.delete(id);
        this.rightLayerIds.delete(id);

        // Remove clone from secondary map if exists
        if (this.secondaryMapLayers[id]) {
            try { this.secondaryMap.removeLayer(this.secondaryMapLayers[id]); } catch (e) {}
            delete this.secondaryMapLayers[id];
        }

        // Don't re-render immediately — wait for potential re-add
        setTimeout(() => {
            if (!this._pendingReassignment[id]) {
                this._renderAssignmentUI();
            }
        }, 100);
    }

    /**
     * When layers are cleared while in compare mode
     */
    _handleLayersCleared(detail) {
        if (!this.isActive) return;

        // Get layer panel's current state to see what was cleared
        const currentLayers = this.layerPanel?.getCurrentLayers() || [];
        const currentIds = new Set(currentLayers.map(l => l.id));

        // Remove any IDs no longer in the panel
        for (const id of [...this.leftLayerIds]) {
            if (!currentIds.has(id)) this.leftLayerIds.delete(id);
        }
        for (const id of [...this.rightLayerIds]) {
            if (!currentIds.has(id)) {
                this.rightLayerIds.delete(id);
                if (this.secondaryMapLayers[id]) {
                    try { this.secondaryMap.removeLayer(this.secondaryMapLayers[id]); } catch (e) {}
                    delete this.secondaryMapLayers[id];
                }
            }
        }

        this._renderAssignmentUI();
    }

    _handleBaseLayerChange(detail) {
        if (!this.isActive || !this.secondaryMap) return;

        // Remove old base layers from secondary
        if (this.secondaryBaseLayers) {
            try { this.secondaryMap.removeLayer(this.secondaryBaseLayers.base); } catch (e) {}
            if (this.secondaryBaseLayers.labels) {
                try { this.secondaryMap.removeLayer(this.secondaryBaseLayers.labels); } catch (e) {}
            }
        }

        // Add new base layers
        this.secondaryBaseLayers = this.mapCore.createBaseLayerClone(detail.type);
        this.secondaryBaseLayers.base.addTo(this.secondaryMap);
        if (this.secondaryBaseLayers.labels) {
            this.secondaryBaseLayers.labels.addTo(this.secondaryMap);
        }
    }

    _handleResize() {
        if (!this.isActive) return;
        if (this.primaryMap) this.primaryMap.invalidateSize();
        if (this.secondaryMap) this.secondaryMap.invalidateSize();
    }

    _updateCompareButtonText(active) {
        const btn1 = document.getElementById('compare-btn');
        const btn2 = document.getElementById('local-compare-btn');
        const text = active ? '🔀 Exit Compare' : '🔀 Compare Layers';
        if (btn1) btn1.textContent = text;
        if (btn2) btn2.textContent = text;
    }
}
