// Typeset maths with KaTeX (vendored), the LaTeX look of the 3Blue1Brown / Deepia videos.
//
//   <span data-tex="z = \mu + \sigma\epsilon"></span>            inline
//   <div data-tex="..." data-display></div>                       display style (big operators, limits)
//
// Explainable symbols: wrap a piece in \htmlData{tip=KEY}{...} and give the element (or an ancestor)
// data-tips='{"KEY": "what it means"}'. The piece becomes a .sym with a .tip bubble, the
// same widget the hand-written formulas used (css/components.css), opened by hover, click or tap.
// \htmlClass{recon}{...} / \htmlClass{kl}{...} colour a term like the rest of the slide.

import katex from "../vendor/katex-0.16.47/katex.mjs";

const OPTIONS = {
  throwOnError: false,
  strict: "ignore",
  output: "htmlAndMathml", // MathML for screen readers, HTML for the eyes
  trust: ({ command }) => command === "\\htmlData" || command === "\\htmlClass",
};

/** Render one TeX string into an element. */
export function tex(el, source, { display = false } = {}) {
  katex.render(source, el, { ...OPTIONS, displayMode: display });
  addTips(el);
}

/** Render every [data-tex] element under root (once). */
export function renderMath(root = document) {
  for (const el of root.querySelectorAll("[data-tex]:not([data-tex-done])")) {
    tex(el, el.dataset.tex, { display: el.hasAttribute("data-display") });
    el.dataset.texDone = "";
  }
}

function addTips(el) {
  const holder = el.closest("[data-tips]");
  if (!holder) return;
  const tips = JSON.parse(holder.dataset.tips);
  for (const piece of el.querySelectorAll("[data-tip]")) {
    const text = tips[piece.dataset.tip];
    if (!text) continue;
    // Not focusable: KaTeX's visual layer is aria-hidden (screen readers read its MathML copy).
    // Hover, click or tap shows the tip.
    piece.classList.add("sym");
    const bubble = document.createElement("span");
    bubble.className = "tip";
    bubble.textContent = text;
    piece.append(bubble);
  }
}
