const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

navToggle?.addEventListener('click', () => {
  const open = navToggle.getAttribute('aria-expanded') === 'true';
  navToggle.setAttribute('aria-expanded', String(!open));
  navLinks?.classList.toggle('open', !open);
});

navLinks?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    navToggle?.setAttribute('aria-expanded', 'false');
    navLinks.classList.remove('open');
  });
});

const modelNames = {
  'gemini-3-flash': 'Gemini 3 Flash',
  'qwen-27b': 'Qwen3.5 27B',
  'gemma-4-31b': 'Gemma 4 31B',
  'qwen-9b': 'Qwen3.5 9B',
  'gpt-oss-120b': 'GPT-OSS 120B',
  'gpt-oss-20b': 'GPT-OSS 20B'
};

const modelImages = {
  'gemini-3-flash': new URL('./img/gemini-3-flash.png', import.meta.url).href,
  'qwen-27b': new URL('./img/qwen-27b.png', import.meta.url).href,
  'gemma-4-31b': new URL('./img/gemma-4-31b.png', import.meta.url).href,
  'qwen-9b': new URL('./img/qwen-9b.png', import.meta.url).href,
  'gpt-oss-120b': new URL('./img/gpt-oss-120b.png', import.meta.url).href,
  'gpt-oss-20b': new URL('./img/gpt-oss-20b.png', import.meta.url).href
};

const modelFigure = document.querySelector('#model-figure');
const modelZoom = document.querySelector('.model-zoom');
document.querySelectorAll('.model-tabs button').forEach((button) => {
  button.addEventListener('click', () => {
    const model = button.dataset.model;
    if (!model || !modelFigure) return;
    document.querySelectorAll('.model-tabs button').forEach((item) => item.setAttribute('aria-selected', 'false'));
    button.setAttribute('aria-selected', 'true');
    modelFigure.classList.add('switching');
    setTimeout(() => {
      modelFigure.src = modelImages[model];
      modelFigure.alt = `Detailed Lost-in-Choice results for ${modelNames[model]}`;
      modelZoom?.setAttribute('aria-label', `Enlarge detailed results for ${modelNames[model]}`);
      modelFigure.classList.remove('switching');
    }, 120);
  });
});

const lightbox = document.querySelector('#figure-lightbox');
const lightboxImage = document.querySelector('#lightbox-image');
const lightboxCaption = document.querySelector('#lightbox-caption');
const lightboxClose = document.querySelector('.lightbox-close');

document.querySelectorAll('.zoom-trigger').forEach((trigger) => {
  trigger.addEventListener('click', () => {
    const source = trigger.querySelector('img');
    if (!source || !lightbox || !lightboxImage || !lightboxCaption) return;
    lightboxImage.src = source.currentSrc || source.src;
    lightboxImage.alt = source.alt;
    lightboxCaption.textContent = source.alt;
    lightbox.showModal();
    document.body.classList.add('lightbox-open');
  });
});

lightboxClose?.addEventListener('click', () => lightbox?.close());
lightbox?.addEventListener('click', (event) => {
  if (event.target === lightbox) lightbox.close();
});
lightbox?.addEventListener('close', () => {
  document.body.classList.remove('lightbox-open');
  if (lightboxImage) lightboxImage.src = '';
});

const copyButton = document.querySelector('.copy-button');
copyButton?.addEventListener('click', async () => {
  const code = document.querySelector('#bibtex-code')?.textContent;
  if (!code) return;
  try {
    await navigator.clipboard.writeText(code);
    copyButton.textContent = 'Copied';
    setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
  } catch {
    copyButton.textContent = 'Select text';
  }
});

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('visible'));
}
