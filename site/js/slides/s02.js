// Warm-up show of hands. The tally lives in the page only: nothing is saved or sent.

const counts = { cat: 0, dog: 0 };

export function init(section) {
  const live = section.querySelector(".hands-live");
  const nums = {
    cat: section.querySelector('[data-count="cat"]'),
    dog: section.querySelector('[data-count="dog"]'),
  };
  const say = () => {
    live.textContent = `Cat people ${counts.cat}. Dog people ${counts.dog}.`;
  };
  for (const side of ["cat", "dog"]) {
    section.querySelector(`[data-side="${side}"]`).addEventListener("click", () => {
      counts[side] += 1;
      nums[side].textContent = String(counts[side]);
      say();
    });
  }
  section.querySelector("[data-reset]").addEventListener("click", () => {
    counts.cat = counts.dog = 0;
    nums.cat.textContent = nums.dog.textContent = "0";
    say();
  });
  say();
}
