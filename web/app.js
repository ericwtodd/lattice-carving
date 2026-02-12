const DATA_BASE = '../demo_data/synthetic_bagel';

const mainImg = document.getElementById('main-image');
const latticeImg = document.getElementById('lattice-image');
const energyImg = document.getElementById('energy-image');
const seamImg = document.getElementById('seam-image');
const slider = document.getElementById('step-slider');
const stepLabel = document.getElementById('step-label');

// Preloaded Image objects: imageCache[step][type] = Image
let imageCache = {};
let maxStep = 0;

function stepDir(step) {
  return `${DATA_BASE}/step_${String(step).padStart(3, '0')}`;
}

const IMAGE_TYPES = ['image', 'lattice_space', 'energy', 'seam_overlay'];

function preloadStep(step) {
  if (imageCache[step]) return;
  imageCache[step] = {};
  const dir = stepDir(step);
  for (const type of IMAGE_TYPES) {
    const img = new Image();
    img.src = `${dir}/${type}.png`;
    imageCache[step][type] = img;
  }
}

function showStep(step) {
  const dir = stepDir(step);
  mainImg.src = `${dir}/image.png`;
  latticeImg.src = `${dir}/lattice_space.png`;
  energyImg.src = `${dir}/energy.png`;
  seamImg.src = `${dir}/seam_overlay.png`;
  stepLabel.textContent = `Step: ${step} / ${maxStep}`;
}

slider.addEventListener('input', () => {
  const step = parseInt(slider.value, 10);
  showStep(step);
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
  const current = parseInt(slider.value, 10);
  if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
    const next = Math.max(0, current - 1);
    slider.value = next;
    showStep(next);
  } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
    const next = Math.min(maxStep, current + 1);
    slider.value = next;
    showStep(next);
  }
});

// Load metadata and initialize
fetch(`${DATA_BASE}/metadata.json`)
  .then(r => r.json())
  .then(meta => {
    maxStep = meta.steps.length - 1;
    slider.max = maxStep;
    slider.value = 0;

    // Preload all steps
    for (let i = 0; i <= maxStep; i++) {
      preloadStep(i);
    }

    showStep(0);
  })
  .catch(err => {
    console.error('Failed to load metadata:', err);
    stepLabel.textContent = 'Error loading data — run generate_demo_data.py first';
  });
