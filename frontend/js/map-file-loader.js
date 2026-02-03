/**
 * Map File Loader Module
 * Handles SHP and KML file loading and parsing
 */

class MapFileLoader {
    constructor(manager) {
        this.manager = manager;
    }

    async loadShpFile(file) {
        console.log('📂 Loading SHP file:', file.name);
        
        try {
            const arrayBuffer = await file.arrayBuffer();
            let geojson = await shp(arrayBuffer);
            
            console.log('✅ SHP parsed:', geojson);
            
            if (Array.isArray(geojson)) {
                geojson = geojson[0];
            }
            
            // Clear existing drawings
            this.manager.drawing.drawnItems.clearLayers();
            
            const aoiStyle = {
                color: '#ff7800',
                weight: 3,
                opacity: 0.8,
                fillColor: '#ff7800',
                fillOpacity: 0.1,
                pane: 'aoiPane'
            };
            
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
            
            if (geojson.features) {
                geojson.features.forEach(feature => {
                    const type = feature.geometry?.type;
                    
                    if (type === 'Polygon' || type === 'MultiPolygon') {
                        const layer = L.geoJSON(feature, { style: aoiStyle });
                        layer.eachLayer(l => {
                            l.options.pane = 'aoiPane';
                            createdLayers.push(l);
                            polygonLayers.push(l);
                            this.manager.drawing.drawnItems.addLayer(l);
                        });
                    } else if (type === 'LineString') {
                        console.log('Converting LineString to Polygon');
                        const coords = feature.geometry.coordinates.map(c => [c[1], c[0]]);
                        const polygon = L.polygon(coords, aoiStyle);
                        polygon.options.pane = 'aoiPane';
                        createdLayers.push(polygon);
                        polygonLayers.push(polygon);
                        this.manager.drawing.drawnItems.addLayer(polygon);
                    } else if (type === 'MultiLineString') {
                        console.log('Converting MultiLineString to Polygons');
                        feature.geometry.coordinates.forEach(lineCoords => {
                            const coords = lineCoords.map(c => [c[1], c[0]]);
                            const polygon = L.polygon(coords, aoiStyle);
                            polygon.options.pane = 'aoiPane';
                            createdLayers.push(polygon);
                            polygonLayers.push(polygon);
                            this.manager.drawing.drawnItems.addLayer(polygon);
                        });
                    } else if (type === 'Point') {
                        const latlng = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
                        const marker = L.circleMarker(latlng, pointStyle);
                        marker.options.pane = 'aoiPane';
                        createdLayers.push(marker);
                        pointLayers.push(marker);
                        this.manager.drawing.drawnItems.addLayer(marker);
                    } else if (type === 'MultiPoint') {
                        feature.geometry.coordinates.forEach(coord => {
                            const latlng = [coord[1], coord[0]];
                            const marker = L.circleMarker(latlng, pointStyle);
                            marker.options.pane = 'aoiPane';
                            createdLayers.push(marker);
                            pointLayers.push(marker);
                            this.manager.drawing.drawnItems.addLayer(marker);
                        });
                    }
                });
            } else {
                const geoJsonLayer = L.geoJSON(geojson, { 
                    style: aoiStyle,
                    pointToLayer: (feature, latlng) => {
                        return L.circleMarker(latlng, pointStyle);
                    }
                });
                
                geoJsonLayer.eachLayer(l => {
                    l.options.pane = 'aoiPane';
                    createdLayers.push(l);
                    this.manager.drawing.drawnItems.addLayer(l);
                    
                    if (l instanceof L.Polygon || l instanceof L.Rectangle) {
                        polygonLayers.push(l);
                    } else if (l instanceof L.Polyline) {
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
                // Update drawing module
                this.manager.drawing.setAOIFromLayers(createdLayers);
                
                // Calculate combined bounds
                const combinedBounds = L.latLngBounds();
                createdLayers.forEach(layer => {
                    if (layer.getBounds) {
                        combinedBounds.extend(layer.getBounds());
                    } else if (layer.getLatLng) {
                        combinedBounds.extend(layer.getLatLng());
                    }
                });
                
                this.manager.map.fitBounds(combinedBounds);
                
                console.log(`✅ AOI set from SHP: ${polygonLayers.length} polygon(s), ${lineLayers.length} line(s), ${pointLayers.length} point(s)`);
                
                // Trigger callback
                if (this.manager.drawing.onAOIChange) {
                    this.manager.drawing.onAOIChange(true);
                }
                if (this.manager.onAOIChange) {
                    this.manager.onAOIChange(true);
                }
                
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
            const text = await file.text();
            const parser = new DOMParser();
            const kml = parser.parseFromString(text, 'text/xml');
            
            const parseError = kml.querySelector('parsererror');
            if (parseError) {
                throw new Error('Invalid KML file format');
            }
            
            console.log('KML parsed, searching for coordinates...');
            
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
                    }
                }
            }
            
            // Look for LinearRing
            if (polygonCoords.length === 0) {
                const linearRings = kml.getElementsByTagName('LinearRing');
                for (const ring of linearRings) {
                    const coordsElements = ring.getElementsByTagName('coordinates');
                    if (coordsElements.length > 0) {
                        const coordsText = coordsElements[0].textContent.trim();
                        const coords = this._parseKmlCoordinates(coordsText);
                        if (coords.length >= 3) {
                            polygonCoords.push(coords);
                        }
                    }
                }
            }
            
            // Look for LineString elements
            const lineStrings = kml.getElementsByTagName('LineString');
            for (const line of lineStrings) {
                const coordsElements = line.getElementsByTagName('coordinates');
                if (coordsElements.length > 0) {
                    const coordsText = coordsElements[0].textContent.trim();
                    const coords = this._parseKmlCoordinates(coordsText);
                    if (coords.length >= 2) {
                        lineCoords.push(coords);
                    }
                }
            }
            
            // Look for Point elements
            const points = kml.getElementsByTagName('Point');
            for (const point of points) {
                const coordsElements = point.getElementsByTagName('coordinates');
                if (coordsElements.length > 0) {
                    const coordsText = coordsElements[0].textContent.trim();
                    const coords = this._parseKmlCoordinates(coordsText);
                    if (coords.length >= 1) {
                        pointCoords.push(coords[0]);
                    }
                }
            }
            
            // Fallback to generic coordinates
            if (polygonCoords.length === 0 && lineCoords.length === 0 && pointCoords.length === 0) {
                const allCoords = kml.getElementsByTagName('coordinates');
                for (const coordEl of allCoords) {
                    const coordsText = coordEl.textContent.trim();
                    const coords = this._parseKmlCoordinates(coordsText);
                    if (coords.length >= 3) {
                        polygonCoords.push(coords);
                    } else if (coords.length === 2) {
                        lineCoords.push(coords);
                    } else if (coords.length === 1) {
                        pointCoords.push(coords[0]);
                    }
                }
            }
            
            // Clear existing drawings
            this.manager.drawing.drawnItems.clearLayers();
            
            const aoiStyle = {
                color: '#ff7800',
                weight: 3,
                opacity: 0.8,
                fillColor: '#ff7800',
                fillOpacity: 0.1,
                pane: 'aoiPane'
            };
            
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
            
            // Create polygons
            if (polygonCoords.length > 0) {
                for (let i = 0; i < polygonCoords.length; i++) {
                    const latlngs = polygonCoords[i].map(coord => [coord[1], coord[0]]);
                    const polygon = L.polygon(latlngs, aoiStyle);
                    createdLayers.push(polygon);
                    polygonLayers.push(polygon);
                    this.manager.drawing.drawnItems.addLayer(polygon);
                }
            }
            
            // Convert lines to polygons
            if (lineCoords.length > 0) {
                for (let i = 0; i < lineCoords.length; i++) {
                    const latlngs = lineCoords[i].map(coord => [coord[1], coord[0]]);
                    const polygon = L.polygon(latlngs, aoiStyle);
                    createdLayers.push(polygon);
                    polygonLayers.push(polygon);
                    this.manager.drawing.drawnItems.addLayer(polygon);
                }
            }
            
            // Create points
            if (pointCoords.length > 0) {
                for (let i = 0; i < pointCoords.length; i++) {
                    const latlng = [pointCoords[i][1], pointCoords[i][0]];
                    const marker = L.circleMarker(latlng, pointStyle);
                    createdLayers.push(marker);
                    pointLayers.push(marker);
                    this.manager.drawing.drawnItems.addLayer(marker);
                }
            }
            
            if (createdLayers.length > 0) {
                console.log(`✅ Created ${polygonLayers.length} polygon(s), ${lineLayers.length} line(s), ${pointLayers.length} point(s) from KML`);
            } else {
                // Fallback: Try toGeoJSON library
                console.log('No coordinates found directly, trying toGeoJSON library...');
                
                if (typeof toGeoJSON !== 'undefined' && toGeoJSON.kml) {
                    const geojson = toGeoJSON.kml(kml);
                    
                    if (geojson && geojson.features && geojson.features.length > 0) {
                        geojson.features.forEach(feature => {
                            const type = feature.geometry?.type;
                            
                            if (type === 'Polygon' || type === 'MultiPolygon') {
                                const layer = L.geoJSON(feature, { style: aoiStyle });
                                layer.eachLayer(l => {
                                    l.options.pane = 'aoiPane';
                                    createdLayers.push(l);
                                    polygonLayers.push(l);
                                    this.manager.drawing.drawnItems.addLayer(l);
                                });
                            } else if (type === 'LineString') {
                                const coords = feature.geometry.coordinates.map(c => [c[1], c[0]]);
                                const polygon = L.polygon(coords, aoiStyle);
                                polygon.options.pane = 'aoiPane';
                                createdLayers.push(polygon);
                                polygonLayers.push(polygon);
                                this.manager.drawing.drawnItems.addLayer(polygon);
                            } else if (type === 'MultiLineString') {
                                feature.geometry.coordinates.forEach(lineCoords => {
                                    const coords = lineCoords.map(c => [c[1], c[0]]);
                                    const polygon = L.polygon(coords, aoiStyle);
                                    polygon.options.pane = 'aoiPane';
                                    createdLayers.push(polygon);
                                    polygonLayers.push(polygon);
                                    this.manager.drawing.drawnItems.addLayer(polygon);
                                });
                            } else if (type === 'Point') {
                                const latlng = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
                                const marker = L.circleMarker(latlng, pointStyle);
                                marker.options.pane = 'aoiPane';
                                createdLayers.push(marker);
                                pointLayers.push(marker);
                                this.manager.drawing.drawnItems.addLayer(marker);
                            } else if (type === 'MultiPoint') {
                                feature.geometry.coordinates.forEach(coord => {
                                    const latlng = [coord[1], coord[0]];
                                    const marker = L.circleMarker(latlng, pointStyle);
                                    marker.options.pane = 'aoiPane';
                                    createdLayers.push(marker);
                                    pointLayers.push(marker);
                                    this.manager.drawing.drawnItems.addLayer(marker);
                                });
                            }
                        });
                        
                        if (createdLayers.length === 0) {
                            const tempLayer = L.geoJSON(geojson);
                            const bounds = tempLayer.getBounds();
                            if (bounds.isValid()) {
                                console.warn('⚠️ Using bounds rectangle as fallback');
                                const rect = L.rectangle(bounds, aoiStyle);
                                createdLayers.push(rect);
                                polygonLayers.push(rect);
                                this.manager.drawing.drawnItems.addLayer(rect);
                            }
                        }
                    }
                }
            }
            
            if (createdLayers.length > 0) {
                // Update drawing module
                this.manager.drawing.setAOIFromLayers(createdLayers);
                
                // Calculate combined bounds
                const combinedBounds = L.latLngBounds();
                createdLayers.forEach(layer => {
                    if (layer.getBounds) {
                        combinedBounds.extend(layer.getBounds());
                    } else if (layer.getLatLng) {
                        combinedBounds.extend(layer.getLatLng());
                    }
                });
                
                if (combinedBounds.isValid()) {
                    this.manager.map.fitBounds(combinedBounds);
                }
                
                console.log(`✅ AOI set from KML: ${polygonLayers.length} polygon(s), ${lineLayers.length} line(s), ${pointLayers.length} point(s)`);
                
                if (this.manager.drawing.onAOIChange) {
                    this.manager.drawing.onAOIChange(true);
                }
                if (this.manager.onAOIChange) {
                    this.manager.onAOIChange(true);
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

    _parseKmlCoordinates(coordString) {
        const coords = [];
        const parts = coordString.split(/\s+/).filter(p => p.length > 0);
        
        for (const part of parts) {
            const values = part.split(',').map(v => parseFloat(v.trim()));
            if (values.length >= 2 && !isNaN(values[0]) && !isNaN(values[1])) {
                coords.push([values[0], values[1]]);
            }
        }
        
        return coords;
    }
}
