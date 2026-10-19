// The presentation's cat mascot: an original, simple SVG cat face
// (the models only ever see cat faces too).
//
//   mascot({ expression: "happy" })              -> SVG string
//   mountMascot(el, { expression, animate })     -> puts it into el
//   mountAll(root)                                -> fills every [data-mascot] element
//
// Expressions: happy, thinking, confused, sleepy, shocked, monster.
// Accessory: "photocopier" (Photocopier Cat, the identity-trap running gag).

const FUR = "#e8a15a";
const FUR_DARK = "#c46b3a";
const INK = "#24170f";
const PINK = "#d24b3a";

// Eyes, mouth and extras for each expression, drawn on a 200x200 face.
const FACES = {
  happy: {
    eyes: `<path d="M60 102 Q72 88 84 102" /><path d="M116 102 Q128 88 140 102" />`,
    eyeStyle: "stroke",
    mouth: `<path d="M86 128 Q93 136 100 128 Q107 136 114 128" />`,
  },
  thinking: {
    eyes: roundEyes(72, 100, 128, 100, 10, -3, -4),
    mouth: `<path d="M92 130 Q101 135 110 128" />`,
    extra: `<g fill="#24170f"><circle cx="168" cy="52" r="5" /><circle cx="180" cy="36" r="8" /><circle cx="194" cy="14" r="12" /></g>`,
  },
  confused: {
    eyes: roundEyes(72, 100, 128, 100, 11, 0, 0, 7),
    mouth: `<path d="M86 132 Q93 126 100 132 Q107 138 114 130" />`,
    extra: `<text x="168" y="54" font-family="Fredoka, sans-serif" font-weight="700" font-size="46" fill="#94500f">?</text>`,
  },
  sleepy: {
    eyes: `<path d="M60 102 Q72 110 84 102" /><path d="M116 102 Q128 110 140 102" />`,
    eyeStyle: "stroke",
    mouth: `<ellipse cx="100" cy="132" rx="5" ry="4" fill="${INK}" stroke="none" />`,
    extra: `<g font-family="Fredoka, sans-serif" font-weight="700" fill="#1e3f66"><text x="158" y="58" font-size="22">z</text><text x="172" y="40" font-size="30">Z</text></g>`,
  },
  shocked: {
    eyes: `<circle cx="72" cy="100" r="15" fill="#fff" /><circle cx="128" cy="100" r="15" fill="#fff" />
           <circle cx="72" cy="100" r="5" fill="${INK}" /><circle cx="128" cy="100" r="5" fill="${INK}" />`,
    mouth: `<ellipse cx="100" cy="136" rx="9" ry="11" fill="${INK}" stroke="none" />`,
  },
  monster: {
    // A cat decoded from a hole in latent space: mismatched eyes, a third one, wobbly teeth.
    eyes: `<circle cx="68" cy="98" r="14" fill="#1f8a5b" /><circle cx="68" cy="98" r="5" fill="${INK}" />
           <circle cx="132" cy="104" r="9" fill="#d24b3a" /><circle cx="132" cy="104" r="3" fill="${INK}" />
           <circle cx="100" cy="74" r="8" fill="#e07a3d" /><circle cx="101" cy="75" r="3" fill="${INK}" />`,
    mouth: `<path d="M78 130 L86 140 L93 130 L100 141 L107 130 L114 140 L122 130" fill="#fffaf3" stroke="${INK}" />`,
  },
};

function roundEyes(lx, ly, rx, ry, r, dx, dy, rightR = r) {
  return `<circle cx="${lx}" cy="${ly}" r="${r}" fill="${INK}" /><circle cx="${rx}" cy="${ry}" r="${rightR}" fill="${INK}" />
    <circle cx="${lx + dx - 3}" cy="${ly + dy - 2}" r="${r * 0.35}" fill="#fff" />
    <circle cx="${rx + dx - 3}" cy="${ry + dy - 2}" r="${rightR * 0.35}" fill="#fff" />`;
}

function face(expression) {
  const f = FACES[expression] ?? FACES.happy;
  const monster = expression === "monster";
  const fur = monster ? "#6b3050" : FUR;
  const furDark = monster ? "#4a2038" : FUR_DARK;
  const eyeAttrs =
    f.eyeStyle === "stroke" ? `fill="none" stroke="${INK}" stroke-width="7" stroke-linecap="round"` : "";
  return `
  <g class="mascot-face">
    ${monster ? `<g opacity="0.45" transform="translate(-5 2)"><ellipse cx="100" cy="108" rx="74" ry="62" fill="#e07a3d" /></g>` : ""}
    <path d="M38 78 L44 16 L92 50 Z" fill="${fur}" stroke="${furDark}" stroke-width="4" stroke-linejoin="round" />
    <path d="M162 78 L156 16 L108 50 Z" fill="${fur}" stroke="${furDark}" stroke-width="4" stroke-linejoin="round" />
    <path d="M50 64 L53 32 L78 50 Z" fill="${PINK}" />
    <path d="M150 64 L147 32 L122 50 Z" fill="${PINK}" />
    <ellipse cx="100" cy="108" rx="74" ry="62" fill="${fur}" stroke="${furDark}" stroke-width="4" />
    <g stroke="${furDark}" stroke-width="6" stroke-linecap="round">
      <path d="M100 50 L100 66" /><path d="M84 53 L87 66" /><path d="M116 53 L113 66" />
    </g>
    <ellipse cx="56" cy="124" rx="12" ry="7" fill="${PINK}" opacity="0.55" />
    <ellipse cx="144" cy="124" rx="12" ry="7" fill="${PINK}" opacity="0.55" />
    <g class="mascot-eyes" ${eyeAttrs}>${f.eyes}</g>
    <path d="M93 116 Q100 112 107 116 Q100 124 93 116 Z" fill="${PINK}" stroke="${INK}" stroke-width="2" stroke-linejoin="round" />
    <g fill="none" stroke="${INK}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">${f.mouth}</g>
    <g stroke="${INK}" stroke-width="2.5" stroke-linecap="round" opacity="0.7">
      <path d="M40 116 L12 110" /><path d="M40 126 L10 128" /><path d="M160 116 L188 110" /><path d="M160 126 L190 128" />
    </g>
    ${f.extra ?? ""}
  </g>`;
}

// Photocopier Cat: the face sits on a copier that prints a perfect (useless) copy.
function photocopier() {
  return `
  <g class="mascot-photocopier">
    <rect x="18" y="176" width="164" height="70" fill="#e7dcc6" stroke="#24170f" stroke-width="4" />
    <rect x="12" y="166" width="176" height="18" fill="#fffaf3" stroke="#24170f" stroke-width="4" />
    <circle cx="160" cy="212" r="7" fill="#1f5c3a" />
    <rect x="34" y="200" width="70" height="8" fill="#5c4636" />
    <rect x="34" y="216" width="50" height="8" fill="#5c4636" />
    <g transform="translate(150 224) rotate(8)">
      <rect width="62" height="52" fill="#fffaf3" stroke="#24170f" stroke-width="3" />
      <g transform="translate(9 6) scale(0.22)">${face("happy").replace('class="mascot-face"', "")}</g>
    </g>
  </g>`;
}

export function mascot({ expression = "happy", accessory = null, title = "" } = {}) {
  if (expression === "photocopier") [expression, accessory] = ["happy", "photocopier"];
  const tall = accessory === "photocopier";
  const viewBox = tall ? "0 0 220 260" : "0 0 200 180";
  const label = title || `Cat mascot, ${expression}${tall ? ", sitting on a photocopier" : ""}`;
  return `<svg class="mascot" viewBox="${viewBox}" role="img" aria-label="${label}" xmlns="http://www.w3.org/2000/svg">
    ${tall ? photocopier() : ""}
    ${tall ? `<g transform="translate(10 0)">${face(expression)}</g>` : face(expression)}
  </svg>`;
}

export function mountMascot(el, { expression = "happy", accessory = null, animate = false } = {}) {
  el.innerHTML = mascot({ expression, accessory });
  el.classList.toggle("mascot-anim", animate);
  return el.firstElementChild;
}

export function mountAll(root = document) {
  for (const el of root.querySelectorAll("[data-mascot]")) {
    mountMascot(el, { expression: el.dataset.mascot, animate: el.hasAttribute("data-animate") });
  }
}
