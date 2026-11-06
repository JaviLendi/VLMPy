/*
 * Author: Javier Lendínez
 * SPDX-License-Identifier: AGPL-3.0-only
 */
/* Shared info popups (wing, airfoil, results). */
const INFO_ANIM_MS = 150;
let activeInfoTrigger = null;

/* Block interaction with the page while a popup is open. */
function setInfoBackdrop(show) {
  let backdrop = document.querySelector('.info-backdrop');
  if (!backdrop && show) {
    backdrop = document.createElement('div');
    backdrop.className = 'info-backdrop';
    document.body.appendChild(backdrop);
  }
  if (!backdrop) return;
  clearTimeout(backdrop._hideTimer);
  if (show) {
    backdrop.classList.remove('hide');
    backdrop.style.display = 'block';
  } else if (backdrop.style.display !== 'none') {
    backdrop.classList.add('hide');
    backdrop._hideTimer = setTimeout(() => { backdrop.style.display = 'none'; backdrop.classList.remove('hide'); }, INFO_ANIM_MS);
  }
}

/* Hide a popup with an exit animation. */
function hideInfoPopup(popup) {
  if (popup.getAttribute('aria-hidden') === 'true' && popup.style.display === 'none') return;
  popup.setAttribute('aria-hidden', 'true');
  popup.classList.add('hide');
  clearTimeout(popup._hideTimer);
  popup._hideTimer = setTimeout(() => {
    popup.style.display = 'none';
    popup.classList.remove('hide');
  }, INFO_ANIM_MS);
}

function openInfo(id) {
  const popup = document.getElementById(id);
  if (!popup) return;
  activeInfoTrigger = document.activeElement;
  setInfoBackdrop(true);
  // Move the popup out of stacking contexts so it appears above the backdrop.
  if (popup.parentElement !== document.body) document.body.appendChild(popup);
  clearTimeout(popup._hideTimer);
  popup.classList.remove('hide');
  popup.style.display = 'flex';
  popup.setAttribute('aria-hidden', 'false');
  popup.focus?.();
}

function closeInfo(id) {
  const popup = document.getElementById(id);
  if (!popup) return;
  hideInfoPopup(popup);
  activeInfoTrigger?.focus?.();
  activeInfoTrigger = null;
  setInfoBackdrop(false);
}

function closeAllInfo() {
  document.querySelectorAll('.info-popup').forEach(hideInfoPopup);
  activeInfoTrigger?.focus?.();
  activeInfoTrigger = null;
  setInfoBackdrop(false);
}

document.addEventListener('mousedown', (event) => {
  document.querySelectorAll('.info-popup').forEach((popup) => {
    if (popup.getAttribute('aria-hidden') === 'false' && !popup.contains(event.target)) {
      closeInfo(popup.id);
    }
  });
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeAllInfo();
});
