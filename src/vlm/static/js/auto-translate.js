(function() {
  if (window._vlmpy_translate_loaded) return;
  window._vlmpy_translate_loaded = true;

  const SUPPORTED_LANGS = [
    { code: 'en', label: 'EN', flag: '🇬🇧' },
    { code: 'es', label: 'ES', flag: '🇪🇸' },
    { code: 'fr', label: 'FR', flag: '🇫🇷' },
    { code: 'de', label: 'DE', flag: '🇩🇪' },
    { code: 'it', label: 'IT', flag: '🇮🇹' },
    { code: 'pt', label: 'PT', flag: '🇵🇹' },
    { code: 'zh', label: 'ZH', flag: '🇨🇳' },
    { code: 'ja', label: 'JA', flag: '🇯🇵' }
  ];
  const PAGE_LANG = (document.documentElement.lang || 'en').split('-')[0];

  function setGoogTransCookie(from, to) {
    document.cookie = 'googtrans=/' + from + '/' + to + ';path=/';
    document.cookie = 'googtrans=/' + from + '/' + to + ';path=/;domain=' + location.hostname;
  }
  function getStoredLanguage() {
    return localStorage.getItem('vlmpy-google-translate-lang');
  }
  function saveLanguage(lang) {
    localStorage.setItem('vlmpy-google-translate-lang', lang);
  }

  // Modern floating selector UI
  const selectorWrap = document.createElement('div');
  selectorWrap.className = 'language-selector modern';
  selectorWrap.setAttribute('role', 'group');
  selectorWrap.setAttribute('aria-label', 'Selector de idioma');

  SUPPORTED_LANGS.forEach(({code, label, flag}) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'language-btn';
    btn.innerHTML = `<span class="label">${label}</span>`;
    btn.dataset.lang = code;
    btn.setAttribute('aria-label', 'Cambiar a ' + code);

    if (
      (getStoredLanguage() && getStoredLanguage() === code) ||
      (!getStoredLanguage() && PAGE_LANG === code)
    ) {
      btn.classList.add('active');
    }

    btn.addEventListener('click', () => {
      setGoogTransCookie(PAGE_LANG, code);
      saveLanguage(code);

      selectorWrap.querySelectorAll('.language-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      setTimeout(() => location.reload(), 300);
    });

    selectorWrap.appendChild(btn);
  });

  // Reset button
  const resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.className = 'language-btn';
  resetBtn.innerHTML = `<span class="flag">🌐</span> <span class="label">ORIG</span>`;
  resetBtn.setAttribute('aria-label', 'Idioma original');
  resetBtn.dataset.lang = PAGE_LANG;

  resetBtn.addEventListener('click', () => {
    setGoogTransCookie(PAGE_LANG, PAGE_LANG);
    localStorage.removeItem('vlmpy-google-translate-lang');

    selectorWrap.querySelectorAll('.language-btn').forEach(b => b.classList.remove('active'));
    resetBtn.classList.add('active');

    setTimeout(() => location.reload(), 300);
  });

  selectorWrap.appendChild(resetBtn);
  document.body.appendChild(selectorWrap);

  // Hidden Google Translate widget
  const container = document.createElement('div');
  container.id = 'google_translate_element';
  container.style.display = 'none';
  document.body.appendChild(container);

  window.googleTranslateElementInit = function() {
    try {
      new google.translate.TranslateElement({
        pageLanguage: PAGE_LANG,
        layout: google.translate.TranslateElement.InlineLayout.SIMPLE
      }, 'google_translate_element');
    } catch (e) {
      console.warn('[VLMPy AutoTranslate] Google Translate init failed:', e);
    }
  };

  const userLang = (navigator.language || navigator.userLanguage || 'en').split('-')[0];
  const storedLang = getStoredLanguage();

  if (storedLang && SUPPORTED_LANGS.some(l => l.code === storedLang) && storedLang !== PAGE_LANG) {
    setGoogTransCookie(PAGE_LANG, storedLang);
  } else if (SUPPORTED_LANGS.some(l => l.code === userLang) && userLang !== PAGE_LANG) {
    setGoogTransCookie(PAGE_LANG, userLang);
    saveLanguage(userLang);
  }

  const script = document.createElement('script');
  script.src = '//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
  script.async = true;
  script.defer = true;
  document.head.appendChild(script);

  console.log('[VLMPy AutoTranslate] Modern selector initialized with page language:', PAGE_LANG);
  console.log('[VLMPy AutoTranslate] Supported languages:', SUPPORTED_LANGS.map(l => l.code));
})();
