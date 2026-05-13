/**
 * UI Manager Module
 * Handles notifications, loading overlay, panels, and tab controls
 */

class UIManager {
    constructor(platformController) {
        this.platform = platformController;
        this.currentNotification = null;
        this._progressStop = false;
    }

    // Panel methods
    showPanel(panelId) {
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.remove('hidden');
            panel.classList.add('slide-in');
            setTimeout(() => panel.classList.remove('slide-in'), 300);
        }
    }

    hidePanel(panelId) {
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.add('slide-out');
            setTimeout(() => {
                panel.classList.add('hidden');
                panel.classList.remove('slide-out');
            }, 300);
        }
    }

    // Notification methods
    showNotification(message, type = 'info', duration = 5000) {
        if (this.currentNotification) {
            clearTimeout(this.currentNotification.timeout);
        }

        const notification = document.getElementById('notification');
        if (!notification) {
            console.warn('Notification element not found');
            return;
        }
        
        notification.textContent = message;
        notification.className = `notification ${type} show`;

        this.currentNotification = {
            timeout: setTimeout(() => {
                notification.classList.remove('show');
                this.currentNotification = null;
            }, duration)
        };
    }

    // Loading overlay methods
    showLoading(message = 'Processing...', canCancel = true) {
        console.log(`[LOADING] Showing overlay: ${message}`);
        const overlay = document.getElementById('loading-overlay');
        const titleElement = document.getElementById('loading-title');
        const cancelBtn = document.getElementById('loading-cancel-btn');

        if (!overlay) {
            console.error('[LOADING] Overlay element not found!');
            return;
        }

        // Set title
        const loadingTexts = this._getLoadingTexts(message);
        if (titleElement) titleElement.textContent = loadingTexts.title;
        
        // Show/hide cancel button
        if (cancelBtn) {
            cancelBtn.style.display = canCancel ? 'inline-block' : 'none';
            cancelBtn.onclick = () => this._handleLoadingCancel();
        }
        
        this._progressStop = false;
        this._loadingCancelled = false;
        overlay.classList.remove('hidden');
        console.log('[LOADING] Overlay displayed');
    }

    _getLoadingTexts(message) {
        // Simple text mappings
        const textMap = {
            'Searching images...': { title: 'Searching...' },
            'Processing...': { title: 'Extracting analysis results...' },
            'Processing image...': { title: 'Extracting analysis results...' },
            'Processing image on server (export + analysis)...': { title: 'Extracting analysis results...' },
            'Downloading...': { title: 'Downloading...' },
            'Downloading image...': { title: 'Downloading...' },
            'Running detection...': { title: 'Running detection...' },
            'Applying threshold...': { title: 'Applying...' },
            'Loading...': { title: 'Loading...' },
            'Applying visualization...': { title: 'Applying...' },
            'Starting...': { title: 'Initializing...' },
            'Extracting analysis results...': { title: 'Extracting analysis results...' }
        };

        // Check for partial matches
        if (message.includes('Applying visualization')) {
            return { title: 'Applying...' };
        }
        if (message.includes('Extracting') || message.includes('analysis')) {
            return { title: 'Extracting analysis results...' };
        }
        if (message.includes('Processing') || message.includes('Analyzing')) {
            return { title: 'Extracting analysis results...' };
        }
        if (message.includes('Running')) {
            return { title: message };
        }

        return textMap[message] || { title: message };
    }

    _handleLoadingCancel() {
        console.log('[LOADING] Cancel requested');
        this._loadingCancelled = true;
        this.hideLoading();
        this.showNotification('Operation cancelled', 'info');
        
        // Emit cancel event for any listeners
        window.dispatchEvent(new CustomEvent('loadingCancelled'));
    }

    isLoadingCancelled() {
        return this._loadingCancelled;
    }

    updateLoadingProgress(percent, note = '') {
        // Progress is handled by CSS animation
        // We can optionally update the title if note is provided
        if (note) {
            const titleElement = document.getElementById('loading-title');
            if (titleElement) titleElement.textContent = note;
        }
        console.log(`[PROGRESS] ${percent}% - ${note}`);
    }

    hideLoading() {
        console.log('[LOADING] Hiding overlay');
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
            console.log('[LOADING] Overlay hidden');
        } else {
            console.error('[LOADING] Overlay element not found!');
        }
        this._progressStop = true;
    }

    // Tab control methods
    setupTabControls() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');

                // Remove active class from all tabs and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab and corresponding content
                button.classList.add('active');
                document.getElementById(`${targetTab}-tab-content`).classList.add('active');

                // Dispatch tab switch event for layer control panel
                window.dispatchEvent(new CustomEvent('tab:switched', { detail: { tab: targetTab }}));

            });
        });
    }

    // ===== Right Panel =====

    setupRightPanelTabControls() {
        const tabButtons = document.querySelectorAll('.right-tab-btn');
        const tabContents = document.querySelectorAll('.right-tab-content');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-right-tab');
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                button.classList.add('active');
                const el = document.getElementById(`right-${targetTab}-tab-content`);
                if (el) el.classList.add('active');
            });
        });
    }

    showRightPanel() {
        const panel = document.getElementById('right-panel');
        if (panel) {
            panel.style.display = 'flex';
            panel.classList.remove('collapsed');
            const toggleBtn = document.getElementById('right-panel-toggle-btn');
            if (toggleBtn) toggleBtn.textContent = '▶';
            setTimeout(() => {
                if (window.mapManager?.map) window.mapManager.map.invalidateSize();
            }, 350);
        }
    }

    hideRightPanel() {
        const panel = document.getElementById('right-panel');
        if (panel) {
            panel.classList.add('collapsed');
            const toggleBtn = document.getElementById('right-panel-toggle-btn');
            if (toggleBtn) toggleBtn.textContent = '◀';
            setTimeout(() => {
                if (window.mapManager?.map) window.mapManager.map.invalidateSize();
            }, 350);
        }
    }

    switchToAnalysisTab() {
        this.showRightPanel();
        const tabButtons = document.querySelectorAll('.right-tab-btn');
        const tabContents = document.querySelectorAll('.right-tab-content');
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));

        const analysisTab = document.querySelector('[data-right-tab="analysis"]');
        if (analysisTab) analysisTab.classList.add('active');
        const analysisContent = document.getElementById('right-analysis-tab-content');
        if (analysisContent) analysisContent.classList.add('active');
    }

    showNoAnalysisResults() {
        const analysisList = document.querySelector('.analysis-list');
        if (analysisList) {
            analysisList.innerHTML = `
                <div class="no-analysis-results">
                    <div class="empty-state-icon">📊</div>
                    <h4 class="empty-state-title">No Analysis Results</h4>
                    <p class="empty-state-message">Select an image and click "Process Image" to see analysis results here.</p>
                </div>
            `;
        }
    }
}
