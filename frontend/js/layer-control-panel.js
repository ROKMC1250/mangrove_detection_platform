/**
 * Layer Control Panel Module
 * Shows active overlay layers with drag-to-reorder and visibility toggle
 */

class LayerControlPanel {
    constructor(mapManager) {
        this.mapManager = mapManager;
        this.currentTab = 'search';

        // Per-tab layer registries (ordered arrays, bottom-to-top)
        this.layers = {
            'search': [],
            'local-image': []
        };

        this.sectionEl = document.getElementById('layer-control-section');
        this.listEl = document.getElementById('layer-list');
        this.toggleBtn = document.getElementById('layer-control-toggle');
        this.collapsed = false;
        this._dragSrcIndex = null;

        if (!this.sectionEl || !this.listEl) {
            console.warn('[LayerControlPanel] DOM elements not found, will retry.');
            setTimeout(() => {
                this.sectionEl = document.getElementById('layer-control-section');
                this.listEl = document.getElementById('layer-list');
                this.toggleBtn = document.getElementById('layer-control-toggle');
                if (this.toggleBtn) {
                    this.toggleBtn.addEventListener('click', () => this._toggleCollapse());
                }
                console.log('[LayerControlPanel] Retry DOM lookup:', !!this.sectionEl, !!this.listEl);
            }, 500);
        }

        this._bindEvents();

        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this._toggleCollapse());
        }

        console.log('[LayerControlPanel] Initialized');
    }

    _bindEvents() {
        window.addEventListener('layer:added', (e) => {
            try { this._onLayerAdded(e.detail); } catch (err) { console.error('[LayerControlPanel] layer:added error:', err); }
        });
        window.addEventListener('layer:removed', (e) => {
            try { this._onLayerRemoved(e.detail); } catch (err) { console.error('[LayerControlPanel] layer:removed error:', err); }
        });
        window.addEventListener('layers:cleared', (e) => {
            try { this._onLayersCleared(e.detail); } catch (err) { console.error('[LayerControlPanel] layers:cleared error:', err); }
        });
        window.addEventListener('tab:switched', (e) => {
            try { this._onTabSwitched(e.detail); } catch (err) { console.error('[LayerControlPanel] tab:switched error:', err); }
        });
    }

    _getActiveTab() {
        return this.currentTab;
    }

    _onLayerAdded(detail) {
        if (!detail || !detail.id) return;

        const tab = this._getActiveTab();
        const registry = this.layers[tab];

        // Remove existing entry first so re-added layers move to top
        const existing = registry.findIndex(l => l.id === detail.id);
        if (existing !== -1) {
            registry.splice(existing, 1);
        }

        // Always push to end of array = top of list = highest z-order
        registry.push({
            id: detail.id,
            type: detail.type || 'analysis',
            name: detail.name || detail.id,
            leafletLayer: detail.layer,
            visible: true
        });

        this._render();
        // Apply z-order after short delay so DOM elements are ready
        setTimeout(() => this._applyZOrder(), 100);
        this._updateCompareButtons();

        // Scroll to top so newest layer is visible
        if (this.listEl) this.listEl.scrollTop = 0;
    }

    _onLayerRemoved(detail) {
        if (!detail || !detail.id) return;

        // Search across both tabs for the layer
        for (const tab of Object.keys(this.layers)) {
            const registry = this.layers[tab];
            const idx = registry.findIndex(l => l.id === detail.id);
            if (idx !== -1) {
                registry.splice(idx, 1);
                break;
            }
        }
        this._render();
        this._updateCompareButtons();
    }

    _onLayersCleared(detail) {
        const tab = this._getActiveTab();
        const registry = this.layers[tab];

        if (detail && detail.type) {
            this.layers[tab] = registry.filter(l => l.type !== detail.type);
        } else {
            this.layers[tab] = [];
        }

        this._render();
        this._updateCompareButtons();
    }

    _onTabSwitched(detail) {
        if (!detail || !detail.tab) return;
        this.currentTab = detail.tab;
        this._render();
        this._updateCompareButtons();
    }

    _toggleCollapse() {
        this.collapsed = !this.collapsed;
        if (this.listEl) {
            this.listEl.style.display = this.collapsed ? 'none' : '';
        }
        if (this.toggleBtn) {
            this.toggleBtn.textContent = this.collapsed ? '▶' : '▼';
        }
    }

    _render() {
        if (!this.sectionEl || !this.listEl) return;

        const registry = this.layers[this._getActiveTab()];

        if (!registry || registry.length === 0) {
            this.sectionEl.style.display = 'none';
            return;
        }

        this.sectionEl.style.display = '';
        this.listEl.innerHTML = '';

        // Render top-to-bottom (reverse of z-order: last in array = top)
        for (let i = registry.length - 1; i >= 0; i--) {
            const entry = registry[i];
            const item = document.createElement('div');
            item.className = 'layer-item' + (entry.visible ? ' layer-visible' : ' layer-hidden');
            item.dataset.index = i;

            const typeIcon = this._getTypeIcon(entry.type);

            item.innerHTML = `
                <span class="layer-drag-handle" draggable="true" title="Drag to reorder">⠿</span>
                <span class="layer-type-icon">${typeIcon}</span>
                <span class="layer-name" title="${entry.name}">${entry.name}</span>
                <span class="layer-visibility-indicator">${entry.visible ? 'ON' : 'OFF'}</span>
            `;

            // Click anywhere on item to toggle visibility
            item.addEventListener('click', (e) => {
                // Don't toggle if clicking drag handle
                if (e.target.closest('.layer-drag-handle')) return;
                this._toggleVisibility(i);
            });

            // Drag only from the handle
            const handle = item.querySelector('.layer-drag-handle');
            handle.addEventListener('dragstart', (e) => this._onDragStart(e, i));
            item.addEventListener('dragover', (e) => this._onDragOver(e));
            item.addEventListener('dragenter', (e) => this._onDragEnter(e));
            item.addEventListener('dragleave', (e) => this._onDragLeave(e));
            item.addEventListener('drop', (e) => this._onDrop(e, i));
            handle.addEventListener('dragend', (e) => this._onDragEnd(e));

            this.listEl.appendChild(item);
        }
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

    _toggleVisibility(registryIndex) {
        const registry = this.layers[this._getActiveTab()];
        const entry = registry[registryIndex];
        if (!entry || !this.mapManager?.map) return;

        const map = this.mapManager.map;
        if (entry.visible) {
            if (map.hasLayer(entry.leafletLayer)) {
                map.removeLayer(entry.leafletLayer);
            }
            entry.visible = false;
        } else {
            if (!map.hasLayer(entry.leafletLayer)) {
                map.addLayer(entry.leafletLayer);
            }
            entry.visible = true;
            // Delay z-order apply so tile layer DOM is ready
            setTimeout(() => this._applyZOrder(), 100);
        }

        this._render();
    }

    _applyZOrder() {
        const registry = this.layers[this._getActiveTab()];
        // Set z-index on each layer's DOM element directly
        // Registry index 0 = bottom, last = top
        for (let i = 0; i < registry.length; i++) {
            const entry = registry[i];
            if (!entry.visible || !entry.leafletLayer) continue;
            const zVal = 451 + i; // dataPane is 450, stack above it
            try {
                // L.TileLayer has getContainer()
                if (typeof entry.leafletLayer.getContainer === 'function') {
                    const el = entry.leafletLayer.getContainer();
                    if (el) el.style.zIndex = zVal;
                }
                // L.ImageOverlay has getElement()
                if (typeof entry.leafletLayer.getElement === 'function') {
                    const el = entry.leafletLayer.getElement();
                    if (el) el.style.zIndex = zVal;
                }
                // GeoRasterLayer — try both
                if (entry.leafletLayer._container) {
                    entry.leafletLayer._container.style.zIndex = zVal;
                }
            } catch (e) {}
        }
    }

    // Drag and drop handlers
    _onDragStart(e, registryIndex) {
        this._dragSrcIndex = registryIndex;
        const item = e.target.closest('.layer-item');
        if (item) item.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(registryIndex));
    }

    _onDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    }

    _onDragEnter(e) {
        const item = e.target.closest('.layer-item');
        if (item) item.classList.add('drag-over');
    }

    _onDragLeave(e) {
        const item = e.target.closest('.layer-item');
        if (item) item.classList.remove('drag-over');
    }

    _onDrop(e, displayTargetIndex) {
        e.preventDefault();
        e.stopPropagation();

        const targetItem = e.target.closest('.layer-item');
        const targetRegistryIndex = targetItem ? parseInt(targetItem.dataset.index) : NaN;
        const srcRegistryIndex = this._dragSrcIndex;

        if (srcRegistryIndex === null || srcRegistryIndex === targetRegistryIndex || isNaN(targetRegistryIndex)) return;

        const registry = this.layers[this._getActiveTab()];
        const [moved] = registry.splice(srcRegistryIndex, 1);

        const adjustedTarget = targetRegistryIndex > srcRegistryIndex ? targetRegistryIndex - 1 : targetRegistryIndex;
        registry.splice(adjustedTarget, 0, moved);

        this._applyZOrder();
        this._render();
    }

    _onDragEnd(e) {
        this._dragSrcIndex = null;
        if (this.listEl) {
            this.listEl.querySelectorAll('.layer-item').forEach(item => {
                item.classList.remove('dragging', 'drag-over');
            });
        }
    }

    _updateCompareButtons() {
        const registry = this.layers[this._getActiveTab()];
        const count = registry ? registry.length : 0;
        const compareBtn = document.getElementById('compare-btn');
        const localCompareBtn = document.getElementById('local-compare-btn');

        if (this.currentTab === 'search' && compareBtn) {
            compareBtn.disabled = count < 2;
        }
        if (this.currentTab === 'local-image' && localCompareBtn) {
            localCompareBtn.disabled = count < 2;
        }
    }

    /** Get all layers for the current tab */
    getCurrentLayers() {
        return this.layers[this._getActiveTab()] || [];
    }

    /** Get visible layers for the current tab */
    getVisibleLayers() {
        return (this.layers[this._getActiveTab()] || []).filter(l => l.visible);
    }
}
