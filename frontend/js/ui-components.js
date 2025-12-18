/**
 * UI Components and Utilities
 */
class UIComponents {
    
    // Notification System
    static showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '12px 24px',
            borderRadius: '6px',
            color: 'white',
            fontWeight: '500',
            zIndex: '10000',
            maxWidth: '400px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
        });

        // Set background color based on type
        const colors = {
            'success': '#4CAF50',
            'error': '#F44336',
            'warning': '#FF9800',
            'info': '#2196F3'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        document.body.appendChild(notification);

        // Auto remove after duration
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, duration);
    }

    // Loading Indicator
    static showLoading(message = 'Loading...') {
        const existing = document.getElementById('loading-overlay');
        if (existing) {
            existing.remove();
        }

        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-message">${message}</div>
            </div>
        `;

        // Style the loading overlay
        Object.assign(overlay.style, {
            position: 'fixed',
            top: '0',
            left: '0',
            width: '100%',
            height: '100%',
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: '9999'
        });

        // Add CSS for spinner and content
        const style = document.createElement('style');
        style.textContent = `
            .loading-content {
                background: white;
                padding: 30px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .loading-spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .loading-message {
                color: #333;
                font-weight: 500;
            }
        `;
        document.head.appendChild(style);

        document.body.appendChild(overlay);
    }

    static hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    // Modal Creation
    static createModal(title, content, options = {}) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                ${options.footer ? `<div class="modal-footer">${options.footer}</div>` : ''}
            </div>
        `;

        // Add modal styles
        const style = document.createElement('style');
        style.textContent = `
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
            }
            .modal-content {
                background: white;
                border-radius: 8px;
                max-width: 90vw;
                max-height: 90vh;
                overflow-y: auto;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .modal-header {
                padding: 20px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .modal-header h3 {
                margin: 0;
                color: #333;
            }
            .modal-close {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #999;
            }
            .modal-close:hover {
                color: #333;
            }
            .modal-body {
                padding: 20px;
            }
            .modal-footer {
                padding: 20px;
                border-top: 1px solid #eee;
                text-align: right;
            }
        `;
        document.head.appendChild(style);

        // Close modal functionality
        const closeBtn = modal.querySelector('.modal-close');
        const closeModal = () => modal.remove();
        
        closeBtn.onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };

        document.body.appendChild(modal);
        return modal;
    }

    // Colorbar Creation
    static createColorbar(colormap) {
        if (!colormap) return '';
        
        const { name, min_val, max_val, label } = colormap;
        const minFormatted = Number(min_val).toFixed(3);
        const maxFormatted = Number(max_val).toFixed(3);

        return `
            <div class="colorbar-container">
                <div class="colorbar-label">${label || name}</div>
                <div class="colorbar">
                    <div class="colorbar-gradient ${name}"></div>
                </div>
                <div class="colorbar-values">
                    <span class="colorbar-min">${minFormatted}</span>
                    <span class="colorbar-max">${maxFormatted}</span>
                </div>
            </div>
        `;
    }

    // Progress Bar
    static createProgressBar(containerId, initialMessage = 'Starting...') {
        const container = document.getElementById(containerId);
        if (!container) return null;

        container.innerHTML = `
            <div class="progress-container">
                <div class="progress-message">${initialMessage}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="progress-percentage">0%</div>
            </div>
        `;

        return {
            update: (percentage, message) => {
                const fill = container.querySelector('.progress-fill');
                const messageEl = container.querySelector('.progress-message');
                const percentEl = container.querySelector('.progress-percentage');
                
                if (fill) fill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
                if (messageEl) messageEl.textContent = message || initialMessage;
                if (percentEl) percentEl.textContent = `${Math.round(percentage)}%`;
            },
            complete: (message = 'Complete!') => {
                const messageEl = container.querySelector('.progress-message');
                const fill = container.querySelector('.progress-fill');
                
                if (fill) fill.style.width = '100%';
                if (messageEl) messageEl.textContent = message;
            }
        };
    }

    // Tab Management
    static initializeTabs(tabsContainer) {
        const tabBtns = tabsContainer.querySelectorAll('.tab-btn');
        const tabContents = tabsContainer.querySelectorAll('.tab-content');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.dataset.tab;

                // Remove active class from all tabs and contents
                tabBtns.forEach(b => b.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));

                // Add active class to clicked tab and corresponding content
                btn.classList.add('active');
                const targetContent = document.getElementById(`${targetTab}-tab-content`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
            });
        });
    }

    // Format Period Title for Change Monitoring
    static formatPeriodTitle(startDate, frequency) {
        const date = new Date(startDate);
        
        switch (frequency) {
            case 'monthly':
                return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            case 'weekly':
                const weekStart = new Date(startDate);
                const weekNumber = Math.ceil((weekStart.getDate()) / 7);
                return `${date.getFullYear()}-${weekNumber}week`;
            case 'daily':
            default:
                return startDate;
        }
    }
} 