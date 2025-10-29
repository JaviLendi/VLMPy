/**
 * MessageCenter - Sistema unificado de mensajería para VLMPy
 * Usa las clases CSS: .message .success .error .info
 */
class MessageCenter {
    #container;
    #timeout;
    #activeMessages = new Set();

    constructor(selector = '#message-container', defaultTimeout = 5000) {
        this.#container = document.querySelector(selector) ?? this.#createContainer(selector);
        this.#timeout = defaultTimeout;

        // Delegated click: cierra con botón
        this.#container.addEventListener('click', e => {
            if (e.target.closest('[data-dismiss]')) {
                const message = e.target.closest('.message');
                this.#removeMessage(message);
            }
        });

        // Esc para limpiar todos los mensajes
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') this.clear();
        });
    }

    /* Métodos públicos */
    info(msg, timeout)    { this.#push(msg, 'info', timeout); }
    success(msg, timeout) { this.#push(msg, 'success', timeout); }
    warning(msg, timeout) { this.#push(msg, 'warning', timeout); }
    error(msg, timeout)   { this.#push(msg, 'error', timeout); }

    clear() {
        this.#activeMessages.clear();
        this.#container.innerHTML = '';
    }

    showMessage(text, type = 'info', timeout) {
        const key = String(type).toLowerCase();
        const handlers = {
            success: this.success.bind(this),
            error: this.error.bind(this),
            warning: this.warning.bind(this),
            info: this.info.bind(this),
        };
        const handler = handlers[key] || handlers.info;
        handler(text, timeout);
    }

    /* Privados */
    #push(text, type, timeout = this.#timeout) {
        const hash = `${type}:${text}`;
        if (this.#activeMessages.has(hash)) return;

        const message = this.#createMessage(text, type, hash);
        this.#container.appendChild(message);
        this.#activeMessages.add(hash);

        if (timeout > 0) {
            setTimeout(() => this.#removeMessage(message), timeout);
        }

        return message;
    }

    #createMessage(text, type, hash) {
        const message = document.createElement('div');
        // Aplica clase base 'message' y la clase de severidad como clase separada
        message.classList.add('message', String(type || 'info').toLowerCase());
        message.dataset.hash = hash;

        // Accesibilidad según severidad
        if (type === 'error' || type === 'warning') {
            message.setAttribute('role', 'alert');
            message.setAttribute('aria-live', 'assertive');
        } else {
            message.setAttribute('role', 'status');
            message.setAttribute('aria-live', 'polite');
        }

        // Estructura (coincide con tu CSS)
        message.innerHTML = `
            <div class="message-content">
                <span class="message-text">${text}</span>
                <button class="message-close" data-dismiss aria-label="Cerrar mensaje" type="button">
                    <i class="fas fa-times" aria-hidden="true"></i>
                </button>
            </div>
        `;

        // Animación de entrada
        requestAnimationFrame(() => message.classList.add('message-show'));

        return message;
    }

    #removeMessage(message) {
        if (!message) return;

        const hash = message.dataset.hash;
        if (hash) this.#activeMessages.delete(hash);

        message.classList.add('message-hide');

        setTimeout(() => {
            if (message.parentNode) message.parentNode.removeChild(message);
        }, 300);
    }

    #createContainer(selector) {
        const div = document.createElement('div');
        div.id = selector.replace('#', '');
        div.className = 'message-container';
        document.body.appendChild(div);
        return div;
    }
}

/* Export / instancia global */
window.MessageCenter = MessageCenter;
window.messageCenter = new MessageCenter();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MessageCenter };
}
