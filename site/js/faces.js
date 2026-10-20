// A hand-written "decoder" for the faces slide (s06b): two numbers in, a face out.
// It is an illustration, drawn with plain SVG, not a trained model.
//   smile: 0 (frown) .. 1 (big grin)    hair: 0 (short crop) .. 1 (past the shoulders)
// Every face has the same skin and hair colour, so only the two numbers change.

const SKIN = "#e8b98f";
const HC = "#5c3a1e";

/** A face as an SVG string, in a 100 x 100 box. */
export function faceSVG(smile, hair, { detail = true } = {}) {
  const skin = SKIN, hc = HC;
  const cx = 50, cy = 46, r = 30;
  const parts = [];

  // The back hair grows continuously with `hair`: at 0 it hides behind the head (a short crop),
  // at 1 it is wider than the head and falls past the shoulders. No jumps, so a walk morphs smoothly.
  if (hair > 0) {
    const len = cy + hair * 52; // bottom of the hair, in y: from the middle of the head to below the shoulders
    const w = r - 4 + hair * 10; // half-width: inside the head at 0, wider than it at 1
    const top = cy - r - hair * 10; // the crown gains volume too
    parts.push(
      `<path d="M ${cx - w} ${cy} Q ${cx - w} ${top} ${cx} ${top} Q ${cx + w} ${top} ${cx + w} ${cy}
        L ${cx + w + 2 * hair} ${len} Q ${cx} ${len + 6 * hair} ${cx - w - 2 * hair} ${len} Z" fill="${hc}"/>`,
    );
  }
  // Neck and shoulders.
  parts.push(`<path d="M ${cx - 9} ${cy + 24} L ${cx - 9} ${cy + 34} Q ${cx - 30} ${cy + 38} ${cx - 36} ${cy + 54} L ${cx + 36} ${cy + 54} Q ${cx + 30} ${cy + 38} ${cx + 9} ${cy + 34} L ${cx + 9} ${cy + 24} Z" fill="${skin}" stroke="#24170f" stroke-width="2"/>`);
  // Head.
  parts.push(`<ellipse cx="${cx}" cy="${cy}" rx="${r - 3}" ry="${r}" fill="${skin}" stroke="#24170f" stroke-width="2.4"/>`);
  // Hair cap: thicker fringe for more hair.
  const fringe = 6 + hair * 12;
  parts.push(
    `<path d="M ${cx - r + 2} ${cy - 2} Q ${cx - r} ${cy - r - 6} ${cx} ${cy - r - 4} Q ${cx + r} ${cy - r - 6} ${cx + r - 2} ${cy - 2}
      Q ${cx + r - 8} ${cy - r + fringe} ${cx + 4} ${cy - r + fringe - 2} Q ${cx - 14} ${cy - r + fringe + 2} ${cx - r + 2} ${cy - 2} Z" fill="${hc}"/>`,
  );
  // Eyes, and brows that lift a little with the smile.
  const brow = 4 - smile * 3;
  for (const dx of [-11, 11]) {
    parts.push(`<circle cx="${cx + dx}" cy="${cy + 2}" r="3.2" fill="#24170f"/>`);
    if (detail) {
      const inner = dx < 0 ? cx + dx + 5 : cx + dx - 5;
      const outer = dx < 0 ? cx + dx - 5 : cx + dx + 5;
      parts.push(`<path d="M ${outer} ${cy - 6} L ${inner} ${cy - 6 + (smile < 0.35 ? brow : 0) - (smile >= 0.35 ? 1 : 0)}" stroke="#24170f" stroke-width="2.2" stroke-linecap="round"/>`);
    }
  }
  // Mouth: the curve bends with the smile; a big smile opens up.
  const my = cy + 15;
  const bend = (smile - 0.4) * 22;
  if (smile > 0.72) {
    parts.push(`<path d="M ${cx - 11} ${my} Q ${cx} ${my + bend + 2} ${cx + 11} ${my} Z" fill="#7a2620" stroke="#24170f" stroke-width="2.2" stroke-linejoin="round"/>`);
  } else {
    parts.push(`<path d="M ${cx - 10} ${my} Q ${cx} ${my + bend} ${cx + 10} ${my}" fill="none" stroke="#24170f" stroke-width="2.4" stroke-linecap="round"/>`);
  }
  if (detail && smile > 0.6) {
    const a = (smile - 0.6) * 1.6;
    parts.push(`<circle cx="${cx - 18}" cy="${cy + 10}" r="4" fill="#e07a3d" opacity="${a.toFixed(2)}"/><circle cx="${cx + 18}" cy="${cy + 10}" r="4" fill="#e07a3d" opacity="${a.toFixed(2)}"/>`);
  }
  return parts.join("");
}

/** A small seeded random generator, so the crowd is the same every time. */
export function seeded(seed = 1) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
