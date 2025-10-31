// static/js/auto-translate.js
(function(){
  if (window._vlmpy_translate_loaded) return;
  window._vlmpy_translate_loaded = true;

  // Opciones: idiomas que quieres ofrecer
  const SUPPORTED_LANGS = ['es','fr','de','it','pt']; // añade/elimina códigos ISO
  const PAGE_LANG = (document.documentElement.lang || 'en').split('-')[0];

  // Helper: set cookie for Google Translate (used by widget)
  function setGoogTransCookie(from, to) {
    // No domain => cookie para el host actual
    document.cookie = 'googtrans=/' + from + '/' + to + ';path=/';
    document.cookie = 'googtrans=/' + from + '/' + to + ';path=/;domain=' + location.hostname;
  }

  // Create container for the Google widget (it will render the select if present)
  const container = document.createElement('div');
  container.id = 'google_translate_element';
  // minimal hidden container: we will use our custom UI, but widget needs an element
  container.style.display = 'none';
  document.body.appendChild(container);

  // Build a small floating UI (bottom-right)
  const floatWrap = document.createElement('div');
  floatWrap.setAttribute('aria-hidden','false');
  floatWrap.style.position = 'fixed';
  floatWrap.style.right = '1rem';
  floatWrap.style.bottom = '1rem';
  floatWrap.style.zIndex = 999999;
  floatWrap.style.fontFamily = 'Inter, Arial, sans-serif';
  floatWrap.style.userSelect = 'none';
  floatWrap.style.boxShadow = '0 6px 18px rgba(0,0,0,0.12)';
  floatWrap.style.borderRadius = '10px';
  floatWrap.style.overflow = 'hidden';
  floatWrap.style.background = 'var(--card-bg, #fff)';
  floatWrap.style.backdropFilter = 'blur(4px)';
  floatWrap.style.minWidth = '48px';

  // Toggle button
  const toggleBtn = document.createElement('button');
  toggleBtn.type = 'button';
  toggleBtn.title = 'Idioma / Language';
  toggleBtn.style.border = 'none';
  toggleBtn.style.background = 'transparent';
  toggleBtn.style.padding = '0.6rem';
  toggleBtn.style.cursor = 'pointer';
  toggleBtn.style.display = 'flex';
  toggleBtn.style.alignItems = 'center';
  toggleBtn.style.gap = '0.5rem';
  toggleBtn.style.fontSize = '0.9rem';

  const globe = document.createElement('span');
  globe.textContent = '🌐';
  toggleBtn.appendChild(globe);
  const label = document.createElement('span');
  label.textContent = (navigator.language && navigator.language.startsWith('es')) ? 'ES' : 'Lang';
  label.style.fontWeight = '600';
  toggleBtn.appendChild(label);
  floatWrap.appendChild(toggleBtn);

  // Panel with language buttons (hidden by default)
  const panel = document.createElement('div');
  panel.style.display = 'none';
  panel.style.padding = '0.4rem';
  panel.style.borderTop = '1px solid rgba(0,0,0,0.06)';
  panel.style.background = 'transparent';
  panel.style.display = 'grid';
  panel.style.gridTemplateColumns = 'repeat(2, auto)';
  panel.style.gap = '0.25rem';

  // Add language buttons
  SUPPORTED_LANGS.forEach(code => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = code.toUpperCase();
    b.dataset.lang = code;
    b.style.border = 'none';
    b.style.padding = '0.4rem 0.6rem';
    b.style.borderRadius = '6px';
    b.style.cursor = 'pointer';
    b.style.fontWeight = '600';
    b.style.background = 'rgba(0,0,0,0.04)';
    b.addEventListener('click', () => {
      // Set cookie then reload to apply translation via Google widget
      setGoogTransCookie(PAGE_LANG, code);
      // give a tiny delay for cookie to be set
      setTimeout(() => location.reload(), 300);
    });
    panel.appendChild(b);
  });

  // Button to reset to original language
  const reset = document.createElement('button');
  reset.type = 'button';
  reset.textContent = 'ORIG';
  reset.style.border = 'none';
  reset.style.padding = '0.4rem 0.6rem';
  reset.style.borderRadius = '6px';
  reset.style.cursor = 'pointer';
  reset.style.gridColumn = '1 / -1';
  reset.style.fontWeight = '600';
  reset.style.background = 'rgba(0,0,0,0.04)';
  reset.addEventListener('click', () => {
    setGoogTransCookie(PAGE_LANG, PAGE_LANG);
    setTimeout(() => location.reload(), 300);
  });
  panel.appendChild(reset);

  floatWrap.appendChild(panel);
  document.body.appendChild(floatWrap);

  // Toggle panel open/close
  let open = false;
  toggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    open = !open;
    panel.style.display = open ? 'grid' : 'none';
  });
  document.addEventListener('click', (e) => {
    if (!floatWrap.contains(e.target)) {
      open = false;
      panel.style.display = 'none';
    }
  });

  // Define callback expected by Google's script
  window.googleTranslateElementInit = function() {
    try {
      new google.translate.TranslateElement({
        pageLanguage: PAGE_LANG,
        layout: google.translate.TranslateElement.InlineLayout.SIMPLE
      }, 'google_translate_element');
    } catch (e) {
      // ignore
      // console.warn('Google translate init failed', e);
    }
  };

  // If the browser language is one of supported, auto-set cookie to translate automatically
  const userLang = (navigator.language || navigator.userLanguage || 'en').split('-')[0];
  if (SUPPORTED_LANGS.includes(userLang) && userLang !== PAGE_LANG) {
    // set cookie to auto-translate from PAGE_LANG -> userLang (applied after script loads)
    setGoogTransCookie(PAGE_LANG, userLang);
    label.textContent = userLang.toUpperCase();
  }

  // Load Google Translate script
  const s = document.createElement('script');
  s.src = '//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
  s.async = true;
  s.defer = true;
  document.head.appendChild(s);

})();
