/**
 * Session gate client.
 *
 * On page load, ask the server if we can take the single available session.
 *   - status=active → let the rest of the app boot.
 *   - status=waiting → show a full-screen overlay and poll until granted.
 *                      When granted, reload to initialise cleanly.
 *
 * While active, send a heartbeat every ~30s. On tab close, release the
 * session via sendBeacon so the next user gets in immediately.
 */

class SessionGateClient {
    constructor() {
        this.acquired = false;
        this._heartbeatTimer = null;
        this._pollTimer = null;
        this._readyResolvers = [];
        this._wasWaiting = false;
    }

    ready() {
        return new Promise((resolve) => {
            if (this.acquired) { resolve(); return; }
            this._readyResolvers.push(resolve);
        });
    }

    async start() {
        window.addEventListener('pagehide', () => this._release());
        window.addEventListener('beforeunload', () => this._release());
        await this._tryAcquire();
    }

    async _tryAcquire() {
        try {
            const resp = await fetch('/api/session/acquire', {
                method: 'POST',
                credentials: 'same-origin',
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (data.status === 'active') {
                if (this._wasWaiting) {
                    window.location.reload();
                    return;
                }
                this._onActive(data);
            } else {
                this._wasWaiting = true;
                this._onWaiting(data);
            }
        } catch (err) {
            console.error('[session-gate] acquire failed:', err);
            this._showOverlay({
                title: '연결 오류',
                body: '서버에 접속할 수 없습니다. 재시도 중...',
            });
            clearTimeout(this._pollTimer);
            this._pollTimer = setTimeout(() => this._tryAcquire(), 3000);
        }
    }

    _onActive(data) {
        this.acquired = true;
        this._hideOverlay();
        const interval = (data.heartbeat_interval_s || 30) * 1000;
        clearInterval(this._heartbeatTimer);
        this._heartbeatTimer = setInterval(() => this._heartbeat(), interval);
        const resolvers = this._readyResolvers;
        this._readyResolvers = [];
        resolvers.forEach(r => r());
    }

    _onWaiting(data) {
        const ahead = data.people_ahead;
        const position = data.queue_position;
        const remain = data.timeout_seconds_remaining;

        let line;
        if (ahead === 0) {
            line = (remain != null && remain > 0)
                ? `당신이 다음 차례입니다. 현재 사용자의 세션이 ${Math.ceil(remain)}초 내 종료되면 자동으로 연결됩니다.`
                : '당신이 다음 차례입니다. 현재 사용자가 마무리하는 대로 자동으로 연결됩니다.';
        } else {
            line = `대기 순서: ${position}번째 (앞에 ${ahead}명)`;
        }

        this._showOverlay({
            title: '다른 사용자가 사용 중입니다',
            body: line,
        });
        clearTimeout(this._pollTimer);
        this._pollTimer = setTimeout(() => this._tryAcquire(), 3000);
    }

    async _heartbeat() {
        try {
            const resp = await fetch('/api/session/heartbeat', {
                method: 'POST',
                credentials: 'same-origin',
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (data.status !== 'active') {
                console.warn('[session-gate] heartbeat rejected, reloading');
                window.location.reload();
            }
        } catch (err) {
            console.warn('[session-gate] heartbeat failed:', err);
        }
    }

    _release() {
        try {
            if (navigator.sendBeacon) {
                navigator.sendBeacon('/api/session/release');
            } else {
                fetch('/api/session/release', {
                    method: 'POST',
                    keepalive: true,
                    credentials: 'same-origin',
                });
            }
        } catch (e) { /* no-op */ }
    }

    _showOverlay(opts) {
        let el = document.getElementById('session-gate-overlay');
        if (!el) {
            el = document.createElement('div');
            el.id = 'session-gate-overlay';
            el.innerHTML = `
                <div class="sg-box">
                    <div class="sg-spinner"></div>
                    <h2 class="sg-title"></h2>
                    <p class="sg-body"></p>
                </div>
            `;
            document.body.appendChild(el);
        }
        el.querySelector('.sg-title').textContent = opts.title || '';
        el.querySelector('.sg-body').textContent = opts.body || '';
        el.style.display = 'flex';
    }

    _hideOverlay() {
        const el = document.getElementById('session-gate-overlay');
        if (el) el.style.display = 'none';
    }
}

window.SESSION_GATE = new SessionGateClient();
