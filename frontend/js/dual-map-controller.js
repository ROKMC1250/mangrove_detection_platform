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
        this._pendingPrefixReassignment = {};  // prefix ('td', 'sam3-preview') -> {left, right}

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

        const groups = this._gatherAllLayers();
        const totalCount = groups.A.length + groups.B.length + groups.cd.length;
        if (totalCount < 2) return;

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

        // Our policy: compare mode always paints clones on both sides, so
        // detach every currently-visible overlay from primary. They're still
        // tracked in mapManager.layers and will be restored on deactivate.
        this._detachActiveSlotLayersFromPrimary();

        // Default assignment: Time A's first layer → left, Time B's first → right.
        // Fall back to the active-slot entries or CD results when a slot is empty.
        this.leftLayerIds.clear();
        this.rightLayerIds.clear();
        const defaultLeft = groups.A[0] || groups.cd[0] || groups.B[0];
        const defaultRight = groups.B[0] || groups.cd[0] || groups.A[0];
        if (defaultLeft) this._applyAssignment(defaultLeft.id, 'left', { skipRender: true });
        if (defaultRight && defaultRight.id !== defaultLeft?.id) {
            this._applyAssignment(defaultRight.id, 'right', { skipRender: true });
        }

        // Render the layer assignment UI
        this._renderAssignmentUI();

        // Set up synchronization
        this._setupSync();

        // Update button text
        this._updateCompareButtonText(true);

        // Notify other controllers (e.g. target detection marker sync)
        window.dispatchEvent(new CustomEvent('comparemap:activated', { detail: { secondaryMap: this.secondaryMap } }));
    }

    /**
     * Pull every pickable overlay for compare mode, grouped by source:
     *   A:  Time A analyses (registered via LayerControlPanel, includes stashed)
     *   B:  Time B analyses
     *   cd: Change-detection results (not registered with the panel)
     *
     * Each returned entry has a uniform shape that _cloneLayerEntry /
     * _showOnPrimary understand:
     *   { id, name, type, source: 'A'|'B'|'cd', leafletLayer?,
     *     overlayUrl?, overlayMeta?, resultId? }
     */
    _gatherAllLayers() {
        const bySlot = this.layerPanel?.getAllAnalysisLayersBySlot?.() || { A: [], B: [] };
        const cdEntries = this.platform?.changeDetection?.listResultsForCompareMode?.() || [];

        const toEntry = (e, source) => ({
            id: e.id,
            name: e.name,
            type: e.type || 'analysis',
            source,
            leafletLayer: e.leafletLayer || null,
            overlayUrl: e.overlayUrl || null,
            overlayMeta: e.overlayMeta || null,
            resultId: e.resultId || null,
        });

        return {
            A: bySlot.A.map(e => toEntry(e, 'A')),
            B: bySlot.B.map(e => toEntry(e, 'B')),
            cd: cdEntries.map(e => toEntry(e, 'cd')),
        };
    }

    /**
     * Detach currently-attached analysis/processed/tile overlays from the
     * primary map so compare mode's clone-only policy has a clean canvas.
     * `_restoreLayersToPrimary` puts them back on deactivate.
     */
    _detachActiveSlotLayersFromPrimary() {
        const live = [
            window.mapManager?.layers?.analysisLayers,
            window.mapManager?.layers?.processedLayers,
            window.mapManager?.layers?.tileLayers,
        ];
        for (const dict of live) {
            if (!dict) continue;
            for (const layer of Object.values(dict)) {
                if (layer && this.primaryMap?.hasLayer(layer)) {
                    try { this.primaryMap.removeLayer(layer); } catch (e) {}
                }
            }
        }
    }

    _findEntryById(id) {
        const groups = this._gatherAllLayers();
        return groups.A.find(e => e.id === id)
            || groups.B.find(e => e.id === id)
            || groups.cd.find(e => e.id === id)
            || null;
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
        this._leftMapLayers = {};
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
     * Build a Leaflet layer for a given compare-mode entry. Always returns a
     * *new* instance (both primary and secondary sides get their own) so the
     * original registry layer is never touched — this matters because an
     * entry may come from a stashed slot whose `leafletLayer` mustn't be
     * silently reattached to the currently-active primary map.
     *
     * Entries from LayerControlPanel carry a live Leaflet layer in
     * `leafletLayer`; CD entries instead carry `overlayUrl` + `overlayMeta`.
     */
    _cloneLayerEntry(entry) {
        // CD entry: synthesize a fresh L.ImageOverlay from overlay metadata.
        if (entry.type === 'change-detection') {
            if (!entry.overlayUrl || !entry.overlayMeta) return null;
            const meta = entry.overlayMeta;
            let bounds;
            if (meta.mode === 'local') {
                const h = meta.height || 0;
                const w = meta.width || 0;
                bounds = [[0, 0], [h, w]];
            } else if (meta.bounds) {
                const [s, west, n, east] = meta.bounds;
                bounds = [[s, west], [n, east]];
            } else {
                return null;
            }
            return L.imageOverlay(entry.overlayUrl, bounds, {
                opacity: 1.0,
                pane: 'dataPane',
                crossOrigin: true,
            });
        }

        const layer = entry.leafletLayer;
        if (!layer) return null;

        try {
            if (layer instanceof L.ImageOverlay) {
                return L.imageOverlay(layer._url, layer.getBounds(), {
                    opacity: layer.options.opacity || 1.0,
                    pane: 'dataPane',
                    crossOrigin: true,
                });
            }

            if (layer instanceof L.TileLayer) {
                return L.tileLayer(layer._url, {
                    ...layer.options,
                    pane: 'dataPane',
                });
            }

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
                        pane: 'dataPane',
                    });
                }
            }
        } catch (e) {
            console.warn(`[DualMap] Failed to clone layer ${entry.id}:`, e);
        }
        return null;
    }

    // Kept for backwards compatibility with `_handleLayerAdded` (it passes a
    // synthetic entry with a live leafletLayer).
    _cloneLayer(entry) {
        return this._cloneLayerEntry({
            ...entry,
            type: entry.type || 'analysis',
        });
    }

    /**
     * Store references to layers we added to each side so we can cleanly
     * remove them when the user swaps selection or exits compare mode.
     */
    _getSideDict(side) {
        if (side === 'left') {
            if (!this._leftMapLayers) this._leftMapLayers = {};
            return this._leftMapLayers;
        }
        return this.secondaryMapLayers;
    }

    _getSideMap(side) {
        return side === 'left' ? this.primaryMap : this.secondaryMap;
    }

    _getSideIdSet(side) {
        return side === 'left' ? this.leftLayerIds : this.rightLayerIds;
    }

    _attachSide(side, entry) {
        const map = this._getSideMap(side);
        if (!map) return;
        const clone = this._cloneLayerEntry(entry);
        if (!clone) return;
        this._getSideDict(side)[entry.id] = clone;
        map.addLayer(clone);
    }

    _detachSide(side, layerId) {
        const map = this._getSideMap(side);
        const dict = this._getSideDict(side);
        if (dict[layerId] && map) {
            try { map.removeLayer(dict[layerId]); } catch (e) {}
            delete dict[layerId];
        }
    }

    _clearSide(side) {
        const ids = [...this._getSideIdSet(side)];
        for (const id of ids) {
            this._getSideIdSet(side).delete(id);
            this._detachSide(side, id);
        }
    }

    _applyAssignment(layerId, side, { skipRender = false } = {}) {
        const entry = this._findEntryById(layerId);
        if (!entry) return;

        // Exclusive per side: clear that side first.
        this._clearSide(side);
        this._getSideIdSet(side).add(layerId);
        this._attachSide(side, entry);

        if (!skipRender) this._renderAssignmentUI();
    }

    /** Kept for any external callers. */
    assignLayer(layerId, side) {
        if (side === 'both') {
            this._applyAssignment(layerId, 'left', { skipRender: true });
            this._applyAssignment(layerId, 'right');
        } else {
            this._applyAssignment(layerId, side);
        }
    }

    /**
     * On deactivate we just drop every compare-mode clone. The original
     * active-slot layers are still registered in `mapManager.analysisLayers`,
     * so we just need to re-attach them to the primary map.
     */
    _restoreLayersToPrimary() {
        // Detach every clone we put on the primary during compare mode.
        if (this._leftMapLayers) {
            for (const id of Object.keys(this._leftMapLayers)) {
                try { this.primaryMap.removeLayer(this._leftMapLayers[id]); } catch (e) {}
            }
            this._leftMapLayers = {};
        }

        // Re-attach the active-slot's registered analysis/tile/processed
        // layers in case any were detached as a side-effect of compare mode.
        const live = [
            window.mapManager?.layers?.analysisLayers,
            window.mapManager?.layers?.processedLayers,
            window.mapManager?.layers?.tileLayers,
        ];
        for (const dict of live) {
            if (!dict) continue;
            for (const layer of Object.values(dict)) {
                if (layer && this.primaryMap && !this.primaryMap.hasLayer(layer)) {
                    this.primaryMap.addLayer(layer);
                }
            }
        }
    }

    _renderAssignmentUI() {
        const leftList = document.getElementById('left-map-layers');
        const rightList = document.getElementById('right-map-layers');
        if (!leftList || !rightList) return;

        const groups = this._gatherAllLayers();

        leftList.innerHTML = '';
        rightList.innerHTML = '';

        const totalCount = groups.A.length + groups.B.length + groups.cd.length;
        if (totalCount === 0) {
            leftList.innerHTML = '<div class="dual-layer-empty">No layers</div>';
            rightList.innerHTML = '<div class="dual-layer-empty">No layers</div>';
            return;
        }

        const SECTION_META = [
            { key: 'A',  label: 'Time A',           className: 'dual-section-a' },
            { key: 'B',  label: 'Time B',           className: 'dual-section-b' },
            { key: 'cd', label: 'Change Detection', className: 'dual-section-cd' },
        ];

        for (const section of SECTION_META) {
            const entries = groups[section.key];
            if (!entries.length) continue;
            leftList.appendChild(this._buildSection(section, entries, 'left'));
            rightList.appendChild(this._buildSection(section, entries, 'right'));
        }
    }

    _buildSection(section, entries, side) {
        const wrap = document.createElement('div');
        wrap.className = `dual-layer-section ${section.className}`;
        const header = document.createElement('div');
        header.className = 'dual-layer-section-header';
        header.textContent = section.label;
        wrap.appendChild(header);

        for (const entry of entries) {
            const isActive = this._getSideIdSet(side).has(entry.id);
            wrap.appendChild(this._createAssignmentItem(entry, side, isActive));
        }
        return wrap;
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
                // Deselect — just drop from this side.
                this._getSideIdSet(side).delete(entry.id);
                this._detachSide(side, entry.id);
                this._renderAssignmentUI();
            } else {
                this._applyAssignment(entry.id, side);
            }
        });

        return item;
    }

    _getTypeIcon(type) {
        switch (type) {
            case 'analysis': return '📊';
            case 'tile': return '🗺️';
            case 'image': return '🛰️';
            case 'processed': return '🖼️';
            case 'change-detection': return '🔀';
            default: return '📄';
        }
    }

    /**
     * Get the prefix category for live-updating layers.
     * td-xxx → 'td', sam3-preview-xxx → 'sam3-preview', sam3-text-xxx → 'sam3-text', chg_xxx → 'chg'.
     * Used to match reassignment when IDs change between remove/add cycles.
     */
    _getLayerPrefix(id) {
        if (id.startsWith('td-')) return 'td';
        if (id.startsWith('sam3-preview-')) return 'sam3-preview';
        if (id.startsWith('sam3-text-')) return 'sam3-text';
        if (id.startsWith('chg_')) return 'chg';
        return null;
    }

    /**
     * Re-build the side's clone when an underlying layer is re-added
     * (typical trigger: TD threshold Apply, CD Apply, SAM3 re-run). We
     * rebuild from the current entry registry so even CD layers (which
     * aren't on the primary map) can restore correctly.
     */
    _rebuildSideForLayer(id, side) {
        // Drop the stale clone first.
        this._detachSide(side, id);
        const entry = this._findEntryById(id);
        if (!entry) return;
        this._attachSide(side, entry);
    }

    _handleLayerAdded(detail) {
        if (!this.isActive || !detail || !detail.id) return;

        const id = detail.id;
        const prefix = this._getLayerPrefix(id);

        let pending = this._pendingReassignment[id];
        if (!pending && prefix && this._pendingPrefixReassignment[prefix]) {
            pending = this._pendingPrefixReassignment[prefix];
            delete this._pendingPrefixReassignment[prefix];
        }

        if (pending) {
            if (pending._timer) clearTimeout(pending._timer);
            if (pending._prefixTimer) clearTimeout(pending._prefixTimer);
            delete this._pendingReassignment[id];

            if (pending.left) {
                this.leftLayerIds.add(id);
                this._rebuildSideForLayer(id, 'left');
            }
            if (pending.right) {
                this.rightLayerIds.add(id);
                this._rebuildSideForLayer(id, 'right');
            }

            // Our clone policy means the newly-added layer should NOT stay
            // on the primary map as a side-effect — we render via clones
            // only. Detach `detail.layer` if it's there but we didn't pick
            // left.
            if (!pending.left && detail.layer && this.primaryMap?.hasLayer(detail.layer)) {
                this.primaryMap.removeLayer(detail.layer);
            }
        }

        this._renderAssignmentUI();
    }

    _handleLayerRemoved(detail) {
        if (!this.isActive || !detail || !detail.id) return;

        const id = detail.id;
        const wasLeft = this.leftLayerIds.has(id);
        const wasRight = this.rightLayerIds.has(id);

        if (wasLeft || wasRight) {
            const assignment = { left: wasLeft, right: wasRight };

            const pending = { ...assignment };
            pending._timer = setTimeout(() => {
                delete this._pendingReassignment[id];
            }, 500);
            this._pendingReassignment[id] = pending;

            const prefix = this._getLayerPrefix(id);
            if (prefix) {
                const prefixPending = { ...assignment };
                prefixPending._prefixTimer = setTimeout(() => {
                    delete this._pendingPrefixReassignment[prefix];
                }, 500);
                this._pendingPrefixReassignment[prefix] = prefixPending;
            }
        }

        // Drop clones on both sides (even though only one typically has it).
        if (this._leftMapLayers?.[id]) this._detachSide('left', id);
        if (this.secondaryMapLayers?.[id]) this._detachSide('right', id);
        this.leftLayerIds.delete(id);
        this.rightLayerIds.delete(id);

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
