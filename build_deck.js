/*
 * Presentazione d'esame — Progetto 8, Computer Vision A.A. 2025-2026.
 *
 * VINCOLI dalle Exam guidelines (slide 15-16): sfondo bianco, alto contrasto,
 * numero "corrente / totale" su ogni slide, 10 minuti, inglese.
 *
 * NARRATIVA: il problema e' geometrico -> quindi bag-size bias -> come lo
 * abbiamo tolto -> le teste -> la novita' -> i risultati -> conclusioni.
 *
 * IL NASTRO DI CONTESTO. Ogni slide analitica porta una riga che dichiara
 * ENCODER / CONFRONTO / PROTOCOLLO / METRICA. Senza, chi ascolta non sa mai
 * quale coppia di cose sta guardando, e con nove protocolli e quattro
 * metriche simili fra loro si perde entro la terza tabella. E' il difetto
 * piu' grave della versione precedente.
 */
const P = require("pptxgenjs");

const INK = "111418", MUT = "58626C", LINE = "D6DBDF";
const DEEP = "0B3C49", TEAL = "1C7293";
const RED = "A62B1F", GRN = "1F6F43", AMB = "8A5A12";
const F_H = "Cambria", F_B = "Calibri";

const pres = new P();
pres.layout = "LAYOUT_WIDE";
pres.author = "Progetto 8 - Computer Vision 2025-2026";
pres.title = "Self-Supervised Latent Representations for Imbalanced Apical Periodontitis Grading";

const W = 13.333, H = 7.5, M = 0.62;
const TOT = 26;
let n = 0;

function slide(titolo, occhiello, ctx) {
  n += 1;
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  if (occhiello) {
    s.addText(occhiello.toUpperCase(), {
      x: M, y: 0.28, w: 10.5, h: 0.24, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 10.5, bold: true, color: TEAL, charSpacing: 1.4,
    });
  }
  if (titolo) {
    s.addText(titolo, {
      x: M, y: occhiello ? 0.54 : 0.42, w: W - 2 * M, h: 0.62, isTextBox: true,
      margin: 0, fontFace: F_H, fontSize: 27, bold: true, color: DEEP,
    });
  }
  /* Il nastro: sempre le stesse quattro voci, sempre nello stesso posto. */
  if (ctx) {
    s.addShape(pres.ShapeType.rect, { x: M, y: 1.22, w: W - 2 * M, h: 0.42, fill: { color: "F1F4F6" } });
    const celle = [["MODEL", ctx.m], ["COMPARED WITH", ctx.c], ["PROTOCOL", ctx.p], ["METRIC", ctx.k]];
    let x = M + 0.16;
    const larg = (W - 2 * M - 0.32) / 4;
    celle.forEach(([et, val], i) => {
      s.addText(et, { x: x, y: 1.27, w: larg - 0.15, h: 0.16, isTextBox: true, margin: 0,
        fontFace: F_B, fontSize: 7.5, bold: true, color: MUT, charSpacing: 0.8 });
      s.addText(val, { x: x, y: 1.42, w: larg - 0.15, h: 0.19, isTextBox: true, margin: 0,
        fontFace: F_B, fontSize: 10.5, bold: true, color: i === 0 ? GRN : (i === 1 ? RED : DEEP) });
      if (i < 3) s.addShape(pres.ShapeType.line, { x: x + larg - 0.11, y: 1.29, w: 0, h: 0.28,
        line: { color: LINE, width: 1 } });
      x += larg;
    });
  }
  s.addText(`${n} / ${TOT}`, {
    x: W - M - 1.1, y: H - 0.5, w: 1.1, h: 0.26, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 10.5, color: MUT, align: "right",
  });
  return s;
}
const TY = 1.85;   // prima riga utile sotto il nastro

function th(t) { return { text: t, options: { bold: true, color: "FFFFFF", fill: DEEP, fontSize: 11 } }; }
function td(t, o) { return { text: t, options: Object.assign({ fontSize: 11.5 }, o || {}) }; }
function tab(s, righe, o) {
  s.addTable(righe, Object.assign({
    fontFace: F_B, fontSize: 11.5, color: INK, valign: "middle",
    border: { type: "solid", color: LINE, pt: 0.75 },
  }, o));
}
/* I RIQUADRI MOSTRANO UNA COSA E NE DICONO ALTRE.
 *
 * Misurato sul deck precedente: 1.142 caratteri medi sulla slide contro 80
 * nelle note del relatore. Il rapporto era rovesciato - la slide portava
 * quello che deve dire chi parla, e chi ascolta leggeva invece di ascoltare.
 *
 * Ora di ogni riquadro resta a schermo SOLO IL PRIMO PARAGRAFO; i successivi
 * finiscono nelle note, dove servono. Il contenuto non si perde, cambia
 * supporto. */
let BUF = [];
function nota(s, x, y, w, h, tit, testo, col, bg) {
  const parti = String(testo).split("\n\n");
  if (parti.length > 1) BUF.push(tit.toUpperCase() + " — " + parti.slice(1).join("  "));
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: bg || "F4F6F7" }, rectRadius: 0.07 });
  s.addText(tit, { x: x + 0.2, y: y + 0.14, w: w - 0.4, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F_H, fontSize: 14.5, bold: true, color: col || DEEP });
  s.addText(parti[0], { x: x + 0.2, y: y + 0.5, w: w - 0.4, h: h - 0.68, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 12, color: INK, lineSpacing: 18 });
}

/* Le note del relatore raccolgono anche cio' che i riquadri non mostrano. */
function note(s, testo) {
  s.addNotes([testo].concat(BUF).filter(Boolean).join("\n\n"));
  BUF = [];
}

// ══════════════════════════════════════════════ 1  TITLE
{
  const s = slide(null, null, null);
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: W, h: 2.35, fill: { color: DEEP } });
  s.addText("Self-Supervised Latent Representations for\nImbalanced Apical Periodontitis Grading", {
    x: M, y: 0.55, w: W - 2 * M, h: 1.25, isTextBox: true, margin: 0,
    fontFace: F_H, fontSize: 28, bold: true, color: "FFFFFF", lineSpacing: 33 });
  s.addText("Project 8  ·  Computer Vision A.A. 2025-2026", {
    x: M, y: 1.85, w: W - 2 * M, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 13.5, color: "AFC7D0" });
  s.addText([{ text: "Team\n", options: { bold: true, color: DEEP, fontSize: 12.5 } },
    { text: "‹ name, surname, student ID ›\n‹ name, surname, student ID ›\n‹ name, surname, student ID ›",
      options: { color: INK, fontSize: 13.5 } }],
    { x: M, y: 2.9, w: 5.6, h: 1.5, isTextBox: true, margin: 0, fontFace: F_B, lineSpacing: 21 });
  s.addText([{ text: "Course\n", options: { bold: true, color: DEEP, fontSize: 12.5 } },
    { text: "Prof. Irene Amerini\nSapienza Università di Roma · ALCOR Lab\nSeptember 11, 2026",
      options: { color: INK, fontSize: 13.5 } }],
    { x: 6.9, y: 2.9, w: 5.8, h: 1.5, isTextBox: true, margin: 0, fontFace: F_B, lineSpacing: 21 });
  s.addShape(pres.ShapeType.roundRect, { x: M, y: 4.75, w: W - 2 * M, h: 1.15, fill: { color: "F1F4F6" }, rectRadius: 0.08 });
  s.addText([
    { text: "In one sentence:  ", options: { bold: true, color: DEEP } },
    { text: "grading this disease is largely a ", options: { color: INK } },
    { text: "geometric", options: { bold: true, color: RED } },
    { text: " task, and that fact quietly breaks the evaluation protocol the assignment prescribes. This talk shows how we detected it, removed it, and what the encoder is worth once it is gone.", options: { color: INK } },
  ], { x: M + 0.25, y: 4.95, w: W - 2 * M - 0.5, h: 0.8, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 14, lineSpacing: 21 });
  s.addText("PyTorch  ·  github.com/Gennaro-finance/Computer_Vision", {
    x: M, y: 6.4, w: W - 2 * M, h: 0.28, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11, color: MUT });
  note(s, "15 s. Read the boxed sentence aloud: it is the whole talk in one line.");
}

// ══════════════════════════════════════════════ 2  OUTLINE
{
  const s = slide("Outline", "where we are going", null);
  const v = [
    ["The problem is geometric", "PAI grade is defined by lesion size", RED],
    ["Therefore: bag-size bias", "The assignment's protocol lets the label leak through token count", RED],
    ["Removing the bias", "Fixed-count protocol + a two-factor control", GRN],
    ["Classification heads", "Flat, attention pooling, instance-level MIL", TEAL],
    ["Our novelty", "Balanced token sampling — rebalancing inside latent space", TEAL],
    ["Results", "What the frozen encoder is actually worth", GRN],
    ["Conclusions", "Findings, declared limits, future work", DEEP],
  ];
  let y = 1.5;
  v.forEach((r, i) => {
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.03, w: 0.34, h: 0.34, fill: { color: r[2] } });
    s.addText(String(i + 1), { x: M, y: y + 0.03, w: 0.34, h: 0.34, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText([{ text: r[0] + "   ", options: { bold: true, color: DEEP, fontSize: 15 } },
      { text: r[1], options: { color: MUT, fontSize: 12.5 } }],
      { x: M + 0.5, y: y, w: 11.4, h: 0.4, isTextBox: true, margin: 0, fontFace: F_B, valign: "middle" });
    y += 0.6;
  });
  nota(s, M, 5.85, W - 2 * M, 1.0, "How to read every result slide",
    "Each analytical slide opens with the same strip: which MODEL, what it is COMPARED WITH, under which PROTOCOL, on which METRIC. Nine protocols and four related metrics appear in this work — the strip tells you at a glance which pair you are looking at.",
    DEEP, "F1F4F6");
  note(s, "30 s. Point at the strip explanation — it is how they will follow the numbers.");
}

// ══════════════════════════════════════════════ 3  PROBLEM = GEOMETRY
{
  const s = slide("The problem is geometric", "problem statement");
  s.addText([
    { text: "The Periapical Index grades apical periodontitis 3 < 4 < 5. Clinically, the grade is read from ", options: {} },
    { text: "how large and how dark", options: { bold: true, color: RED } },
    { text: " the periapical radiolucency is. Size is not a correlate of the label — it is close to the definition of the label.", options: {} },
  ], { x: M, y: 1.3, w: 7.3, h: 0.85, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 14.5, color: INK, lineSpacing: 21 });
  tab(s, [
    [th("PAI grade"), th("Median bbox side"), th("Tokens in box (14×14 grid)"), th("Share of dataset")],
    [td("3  — mild"), td("57 px"), td("19"), td("63.5 %")],
    [td("4  — moderate"), td("80 px"), td("34"), td("26.0 %")],
    [td("5  — severe", { bold: true }), td("127 px", { bold: true, color: RED }), td("77", { bold: true, color: RED }), td("10.5 %", { bold: true })],
  ], { x: M, y: 2.35, w: 7.3, colW: [1.85, 1.85, 2.15, 1.45], align: "right" });
  s.addText("Two thresholds on the box side alone — no network at all — already reach macro-F1 0.7567 on the test set.", {
    x: M, y: 4.25, w: 7.3, h: 0.5, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13.5, bold: true, color: RED });

  nota(s, 8.25, 1.3, 4.47, 2.45, "Why this matters for a CNN-free pipeline",
    "A randomly initialised ViT already encodes lesion intensity (R² = 0.99) and lesion size (R² = 0.89) with no training whatsoever. Architecture plus geometry starts close to the ceiling — before any learning happens.",
    DEEP);
  nota(s, 8.25, 3.95, 4.47, 2.3, "The consequence we had to confront",
    "Any evaluation that gives the model access to lesion size through a side channel is measuring the disease definition, not the representation. This is what happened, and detecting it is the core of this work.",
    RED, "FBF3F2");
  s.addText("Dataset: 2,746 panoramic radiographs · 6,741 annotated lesions · patient-level split (4,719 / 1,009 / 1,013)   —   Do et al., Data in Brief 54:110486 (2024)", {
    x: M, y: 6.5, w: W - 2 * M, h: 0.3, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11, color: MUT });
  note(s, "50 s. Land hard on 0.7567 with two thresholds. That number frames everything after it.");
}

// ══════════════════════════════════════════════ 4  MIL / BAG SIZE
{
  const s = slide("The prescribed task is Multiple Instance Learning", "problem statement");
  s.addText([
    { text: "The assignment: ", options: { bold: true } },
    { text: "“Using the provided bounding box coordinates […] extract the latent vectors corresponding to the lesion areas.”", options: { italic: true } },
  ], { x: M, y: 1.3, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13.5, color: INK });
  s.addText([
    { text: "Latent vector", options: { bold: true } },
    { text: "s", options: { bold: true, color: RED } },
    { text: " — plural, and the count varies per lesion. One label, a variable-size set of instances: this is exactly the Multiple Instance Learning setting (Ilse et al., 2018), and its known failure mode is ", options: {} },
    { text: "bag-size bias", options: { bold: true, color: RED } },
    { text: " — the classifier learns to read how many instances are in the bag instead of what they contain.", options: {} },
  ], { x: M, y: 1.85, w: 7.4, h: 1.3, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 14, color: INK, lineSpacing: 21 });

  // schema: bag piccolo vs bag grande
  const C = 0.17;
  function bag(x, y, lato, col, et, ntok) {
    s.addShape(pres.ShapeType.roundRect, { x, y, w: 2.5, h: 1.9, fill: { color: "F4F6F7" }, rectRadius: 0.07 });
    for (let r = 0; r < lato; r++) for (let c = 0; c < lato; c++)
      s.addShape(pres.ShapeType.rect, { x: x + 0.42 + c * C, y: y + 0.42 + r * C, w: C - 0.03, h: C - 0.03, fill: { color: col } });
    s.addText(et, { x: x, y: y + 0.1, w: 2.5, h: 0.26, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 12, bold: true, color: DEEP, align: "center" });
    s.addText(ntok, { x: x, y: y + 1.55, w: 2.5, h: 0.26, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 12, bold: true, color: col, align: "center" });
  }
  bag(M, 3.4, 4, TEAL, "PAI 3 — small bag", "16 instances");
  bag(M + 2.9, 3.4, 8, RED, "PAI 5 — large bag", "64 instances");
  s.addText("The bag size alone separates the classes.", {
    x: M, y: 5.5, w: 5.4, h: 0.3, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13,
    bold: true, italic: true, color: RED, align: "center" });

  nota(s, 8.25, 1.85, 4.47, 4.1, "Why it is a trap here specifically",
    "In a generic MIL problem bag size is incidental. Here it is not: the number of tokens inside the box is a deterministic function of lesion area, and lesion area is the grading criterion.\n\nSo the shortcut is not noise the model might latch onto — it is a near-perfect predictor handed to the classifier for free, requiring no image content at all.\n\nAny encoder, trained or not, is evaluated through this channel.",
    RED, "FBF3F2");
  note(s, "55 s. Name Multiple Instance Learning explicitly — it shows we placed the problem in the literature.");
}

// ══════════════════════════════════════════════ 5  BIAS MISURATO
{
  const s = slide("The bag-size bias, measured", "diagnosis  ·  the central finding", {
    m: "Random ViT (frozen)", c: "I-JEPA (frozen) and no encoder at all",
    p: "As prescribed (bbox mask)", k: "macro-F1, test, 5 seeds" });
  s.addText("If bag size is the signal, then removing the image entirely should not hurt. We tested exactly that.", {
    x: M, y: TY, w: 7.4, h: 0.35, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 14, color: INK });
  tab(s, [
    [th("What the classifier is given"), th("Image content?"), th("macro-F1")],
    [td("Bounding-box mask only — all pixels zeroed", { bold: true }), td("none", { color: RED, bold: true }), td("0.7708", { bold: true, color: RED, align: "right" })],
    [td("Random ViT, full 1,152-d latent vector"), td("full"), td("0.7705", { color: RED, align: "right" })],
    [td("I-JEPA, full 1,152-d latent vector"), td("full"), td("0.7663", { align: "right" })],
    [td("Two thresholds on bbox side length — no network"), td("none", { color: RED, bold: true }), td("0.7567", { align: "right" })],
  ], { x: M, y: 2.35, w: 7.4, colW: [4.2, 1.5, 1.7] });
  nota(s, M, 4.85, 7.4, 1.5, "Read the first two rows together",
    "A mask with no image and a random projection with the full image land within 0.0003 of each other. The encoder contributes nothing measurable — not because the encoders are equivalent, but because this protocol never asks them anything.",
    RED, "FBF3F2");
  nota(s, 8.25, TY, 4.47, 4.4, "What this does and does not prove",
    "It does NOT prove that self-supervised pre-training is useless.\n\nIt proves that the prescribed evaluation cannot distinguish encoders, because a channel exists that answers the question without them.\n\nEvery comparison run under this protocol — including the one the assignment asks for — is therefore uninformative about representation quality. That is the finding, and the rest of the talk follows from it.",
    DEEP);
  note(s, "70 s. The most important slide. Pause after 'within 0.0003 of each other'.");
}

// ══════════════════════════════════════════════ 6  LA SOLUZIONE
{
  const s = slide("Removing the bias: the box becomes a pointer", "solution  ·  step 1", {
    m: "Same frozen encoders", c: "Each other, unchanged",
    p: "Fixed count K — new", k: "Everything else held constant" });
  s.addText("Keep the box centre, take the K nearest tokens, same K for every class. Bag size becomes constant, so it can no longer carry the label — while localisation, which is legitimate information, is preserved.", {
    x: M, y: TY, w: 6.4, h: 0.75, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13.5, color: INK, lineSpacing: 20 });
  const C = 0.145, G = 14;
  function panel(x0, y0, bb, sel, col) {
    s.addShape(pres.ShapeType.rect, { x: x0, y: y0, w: G * C, h: G * C, fill: { color: "EDF0F2" }, line: { color: LINE, width: 0.75 } });
    s.addShape(pres.ShapeType.rect, { x: x0 + sel[1] * C, y: y0 + sel[0] * C, w: sel[2] * C, h: sel[2] * C, fill: { color: col } });
    s.addShape(pres.ShapeType.rect, { x: x0 + bb[1] * C, y: y0 + bb[0] * C, w: bb[2] * C, h: bb[2] * C,
      fill: { type: "none" }, line: { color: INK, width: 1.5, dashType: "dash" } });
  }
  const cx = [M + 0.85, M + 3.05];
  ["Prescribed: bag varies", "Fixed count K = 16"].forEach((t, i) =>
    s.addText(t, { x: cx[i], y: 2.85, w: 2.03, h: 0.26, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 11.5, bold: true, color: i ? GRN : RED, align: "center" }));
  s.addText("PAI 3", { x: M, y: 3.25, w: 0.78, h: 0.3, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11, color: INK, valign: "middle" });
  panel(cx[0], 3.18, [5, 5, 4], [5, 5, 4], RED); panel(cx[1], 3.18, [5, 5, 4], [5, 5, 4], GRN);
  s.addText("PAI 5", { x: M, y: 5.3, w: 0.78, h: 0.3, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11, color: INK, valign: "middle" });
  panel(cx[0], 5.23, [3, 3, 8], [3, 3, 8], RED); panel(cx[1], 5.23, [3, 3, 8], [5, 5, 4], GRN);
  s.addText("16 vs 64 tokens", { x: cx[0], y: 7.0, w: 2.03, h: 0.24, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 11, bold: true, color: RED, align: "center" });
  s.addText("16 vs 16 tokens", { x: cx[1], y: 7.0, w: 2.03, h: 0.24, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 11, bold: true, color: GRN, align: "center" });

  nota(s, 7.3, TY, 5.42, 2.15, "What is held constant — say this out loud",
    "Same 224 px crop, same resolution, same apparent lesion scale, same frozen encoder tokens, same pooling, same head, same five seeds. The only thing that changes is which tokens are aggregated.",
    DEEP);
  tab(s, [
    [th("K"), th("tokens inside lesion"), th("lesion covered")],
    [td("16"), td("99.3 %"), td("73.2 %")],
    [td("36"), td("65.1 %"), td("92.5 %")],
    [td("64"), td("41.9 %"), td("97.9 %")],
  ], { x: 7.3, y: 4.3, w: 5.42, colW: [0.9, 2.42, 2.1], align: "right" });
  s.addText("K controls how much healthy bone enters the window, so the task shifts from “what does this tissue look like” to “how much of this window is lesion” — a question only an encoder that distinguishes lesion from bone can answer.", {
    x: 7.3, y: 5.9, w: 5.42, h: 1.0, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11.5, color: INK, lineSpacing: 17 });
  note(s, "60 s. Show, do not read. Left column: bag changes with class. Right: it does not.");
}

// ══════════════════════════════════════════════ 7  IL CONTROLLO
{
  const s = slide("The control that makes the correction defensible", "solution  ·  step 2", {
    m: "I-JEPA (frozen, ep. 69)", c: "Random ViT (frozen)",
    p: "Two-factor design", k: "macro-F1, test, 5 seeds" });
  s.addText("A protocol built to favour our model would help it everywhere. We therefore built a design in which it must NOT help in two of three cells.", {
    x: M, y: TY, w: 7.3, h: 0.5, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 14, color: INK });
  tab(s, [
    [th("Protocol"), th("Bag-size cue"), th("Localisation"), th("I-JEPA − random"), th("z"), th("Verdict")],
    [td("As prescribed (bbox)"), td("present", { color: RED }), td("present", { color: GRN }), td("−0.0041", { align: "right" }), td("−0.67", { align: "right" }), td("tie — as predicted")],
    [td("Fixed 6×6 grid"), td("removed", { color: GRN }), td("removed", { color: RED }), td("−0.0046", { align: "right" }), td("−0.69", { align: "right" }), td("tie — as predicted")],
    [td("Fixed count K", { bold: true }), td("removed", { color: GRN }), td("present", { color: GRN }), td("+0.05 … +0.07", { bold: true, color: GRN, align: "right" }), td("+3.1 … +5.7", { bold: true, color: GRN, align: "right" }), td("I-JEPA wins", { bold: true, color: GRN })],
  ], { x: M, y: 2.5, w: W - 2 * M, colW: [2.5, 1.6, 1.6, 2.3, 1.5, 2.6] });
  nota(s, M, 4.55, 5.9, 2.0, "Answering the hostile question",
    "“Did you design a protocol that flatters your model?” — On the fixed grid, which removes the cue but also removes localisation, I-JEPA does not win. If we had gone looking for a favourable protocol, that row would be one to hide. We report it because it is what makes the third row interpretable.",
    RED, "FBF3F2");
  nota(s, 6.75, 4.55, 5.97, 2.0, "What the design isolates",
    "The advantage appears only when the bag-size cue is gone AND localisation is kept. So the bounding box is genuinely useful — as a pointer, telling the model where to look — and harmful as a counter, telling it the answer. That distinction is the contribution.",
    GRN, "F1F6F2");
  note(s, "55 s. The two 'tie' rows are the evidence, not the weakness. Say so.");
}

// ══════════════════════════════════════════════ ENCODER — architettura
{
  const s = slide("The network: two encoders and a deliberately small predictor", "the encoder  ·  objective 1");
  const mod = (x, y, w, h, tit, sub, col, sp) => {
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: "FFFFFF" },
      line: { color: col, width: 1.6 }, rectRadius: 0.06 });
    s.addText(tit, { x: x + 0.1, y: y + 0.12, w: w - 0.2, h: 0.28, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 12.5, bold: true, color: col, align: "center" });
    s.addText(sub, { x: x + 0.1, y: y + 0.42, w: w - 0.2, h: h - 0.85, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 10.5, color: INK, align: "center", lineSpacing: 15 });
    s.addText(sp, { x: x + 0.1, y: y + h - 0.4, w: w - 0.2, h: 0.28, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 11.5, bold: true, color: col, align: "center" });
  };
  mod(M, 1.4, 3.5, 2.5, "Context encoder",
      "ViT-S/16\npatch embed 16×16 → 384\n12 × Block (384, 6 heads)\nLayerNorm", DEEP, "21,589,632 par.");
  mod(M + 3.85, 1.4, 3.0, 2.5, "Predictor",
      "384 → 96\n4 × Block (96, 3 heads)\n96 → 384", GRN, "521,856 par.");
  mod(M + 7.2, 1.4, 3.5, 2.5, "Target encoder",
      "identical architecture\nsame 12 blocks, same widths\nonly the weight VALUES differ", TEAL, "21,589,632 par.");
  s.addText("EMA:  θtarget ← τ·θtarget + (1−τ)·θcontext", {
    x: M + 3.5, y: 4.0, w: 6.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 12, bold: true, color: TEAL, align: "center" });
  nota(s, M, 4.45, 6.0, 2.1, "One block class, three instantiations",
    "The same Block — LayerNorm, multi-head attention, residual, LayerNorm, MLP (d→4d→d), residual — is used everywhere. The predictor is not a different architecture: it is the same one at width 96 instead of 384, repeated 4 times instead of 12.",
    DEEP);
  nota(s, 6.85, 4.45, 5.87, 2.1, "Why the predictor is only 2.4 % of an encoder",
    "The assignment asks for a SHALLOW predictor, and the reason is structural. If the predictor were capable enough to guess the targets on its own, the encoder would never be forced to build useful representations. The 384→96→384 bottleneck prevents that by construction.",
    GRN, "F1F6F2");
  note(s, "50 s. One block class everywhere; the predictor's smallness is a design constraint, not a shortcut.");
}

// ══════════════════════════════════════════════ ENCODER — congelato
{
  const s = slide("What “frozen” means, and what is actually trained", "the encoder  ·  objective 2", {
    m: "Target encoder — 21.6 M par.", c: "Head — 5.3 M par.",
    p: "Encoder frozen, as required", k: "no backbone weight moves" });
  s.addText("The assignment requires the representations to be evaluated FROZEN. After pre-training, no weight of the backbone is updated again — not by the classifier's gradient, not by fine-tuning.", {
    x: M, y: TY, w: 7.3, h: 0.65, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13.5, color: INK, lineSpacing: 20 });
  tab(s, [
    [th("Component"), th("Parameters"), th("During pre-training"), th("Downstream")],
    [td("Context encoder"), td("21,589,632", { align: "right" }), td("trained by gradient", { color: GRN }), td("discarded", { color: MUT })],
    [td("Predictor"), td("521,856", { align: "right" }), td("trained by gradient", { color: GRN }), td("discarded", { color: MUT })],
    [td("Target encoder", { bold: true }), td("21,589,632", { bold: true, align: "right" }), td("EMA only, no gradient", { color: TEAL }), td("FROZEN — this is what we ship", { bold: true, color: TEAL })],
    [td("Attention pooling"), td("5,316,480", { align: "right" }), td("—", { color: MUT }), td("trained", { color: GRN })],
    [td("Head (flat)"), td("3,459", { align: "right" }), td("—", { color: MUT }), td("trained", { color: GRN })],
  ], { x: M, y: 2.7, w: W - 2 * M, colW: [2.7, 2.2, 3.4, 3.8] });
  nota(s, M, 4.85, 6.0, 1.7, "Two consequences worth stating",
    "The encoder we deliver never saw a label — not one, at any point.\n\nAnd the same frozen encoder can be compared against a randomly initialised one on equal terms: with fine-tuning that comparison would measure adaptability, not representation.",
    TEAL, "E4F0F2");
  nota(s, 6.85, 4.85, 5.87, 1.7, "The trainable part is smaller than it looks",
    "5,319,939 parameters downstream — and 5,316,480 of them are the pooling. The classifier itself is 3,459: a single Linear from 1,152 to 3. Almost all downstream capacity sits in deciding HOW to aggregate tokens, not in classifying.",
    DEEP);
  note(s, "50 s. 'Never saw a label' is the sentence that lands. Then the pooling/head asymmetry.");
}

// ══════════════════════════════════════════════ ENCODER — la griglia
{
  const s = slide("From a 224 px window to 196 tokens", "the encoder  ·  spatial decomposition", {
    m: "ViT-S/16 patch embedding", c: "—",
    p: "224 px input, fixed", k: "14 × 14 = 196 tokens, always" });
  s.addText("The window is cut into a regular grid of square patches. Each patch becomes exactly one token — no pooling, no overlap, no resizing at this stage.", {
    x: M, y: TY, w: 3.9, h: 0.62, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 12.5, color: INK, lineSpacing: 18 });

  // Griglia in scala, con una cella chiamata fuori. Il richiamo sta a DESTRA
  // della griglia, non sopra: sopra finiva sotto il paragrafo.
  const C = 0.15, G = 14, gx = M + 0.28, gy = 2.62;
  s.addShape(pres.ShapeType.rect, { x: gx, y: gy, w: G * C, h: G * C,
    fill: { color: "F4F6F7" }, line: { color: DEEP, width: 1.4 } });
  for (let i = 1; i < G; i++) {
    s.addShape(pres.ShapeType.line, { x: gx + i * C, y: gy, w: 0, h: G * C, line: { color: LINE, width: 0.5 } });
    s.addShape(pres.ShapeType.line, { x: gx, y: gy + i * C, w: G * C, h: 0, line: { color: LINE, width: 0.5 } });
  }
  s.addShape(pres.ShapeType.rect, { x: gx + 5 * C, y: gy + 4 * C, w: C, h: C, fill: { color: GRN } });
  s.addShape(pres.ShapeType.line, { x: gx + 6 * C, y: gy + 4.5 * C, w: 1.05, h: -0.2,
    line: { color: GRN, width: 1.2, endArrowType: "triangle" } });
  s.addText("one patch\n16 × 16 px\n→ one token", { x: gx + G * C + 0.12, y: 2.95, w: 1.45, h: 0.75,
    isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11, bold: true, color: GRN, lineSpacing: 14 });
  s.addText("224 px  ·  14 × 14 cells", { x: gx, y: gy + G * C + 0.08, w: G * C, h: 0.24,
    isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11, color: DEEP, align: "center" });

  tab(s, [
    [th("Quantity"), th("Value"), th("Why it cannot change")],
    [td("Input window"), td("224 × 224 px", { bold: true }), td("fixed crop, chosen so lesion scale is preserved")],
    [td("Patch size"), td("16 × 16 px", { bold: true }), td("Conv2d kernel = stride = 16 → exact, non-overlapping tiling")],
    [td("Grid"), td("14 × 14", { bold: true }), td("224 ÷ 16 = 14, with no remainder and no padding")],
    [td("Tokens per window", { bold: true }), td("196", { bold: true, color: DEEP }), td("the positional embedding has exactly 196 entries", { bold: true })],
  ], { x: 4.6, y: 2.55, w: 8.1, colW: [1.9, 1.5, 4.7], fontSize: 11 });

  nota(s, 4.6, 4.5, 8.1, 0.95, "The grid is fixed by construction, not by convention",
    "The model could not accept a different grid without interpolating its positional embedding: 196 is baked into the architecture. Verified empirically — input (2, 3, 224, 224) gives tokens (2, 196, 384).",
    DEEP);

  note(s, "35 s. Two numbers only: 16 px per patch, and 196 tokens fixed by the positional embedding.");
}

// ══════════════════════════════════════════════ ENCODER — dove entra la geometria
{
  const s = slide("The bias enters at tiling time", "the encoder  ·  back to the thesis", {
    m: "Nothing learned yet", c: "—",
    p: "Just the 16 px grid", k: "tokens covered by the box" });
  s.addText("A patch is 16 px wide. The median lesion box measures:", {
    x: M, y: TY, w: 6.6, h: 0.36, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 14, color: INK });
  tab(s, [
    [th("PAI grade"), th("median box side"), th("patches per side"), th("tokens covered")],
    [td("3 — mild"), td("57 px", { align: "right" }), td("3.6", { align: "right" }), td("19", { align: "right" })],
    [td("4 — moderate"), td("80 px", { align: "right" }), td("5.0", { align: "right" }), td("34", { align: "right" })],
    [td("5 — severe", { bold: true }), td("127 px", { bold: true, color: RED, align: "right" }), td("7.9", { bold: true, color: RED, align: "right" }), td("77", { bold: true, color: RED, align: "right" })],
  ], { x: M, y: 2.45, w: 6.6, colW: [1.9, 1.65, 1.6, 1.45] });
  s.addText("The grade is written into how many cells the box covers.", {
    x: M, y: 4.35, w: 6.6, h: 0.4, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 14, bold: true, color: RED });
  nota(s, 7.4, TY, 5.32, 2.6, "Neither the head nor the encoder introduces it",
    "The shortcut is already present the moment the window is cut into a grid — before a single weight is applied. No architecture choice downstream can remove something that entered upstream of it.",
    RED, "FBF3F2");
  nota(s, 7.4, 4.95, 5.32, 1.7, "Which is why the fix had to be there too",
    "We did not change the encoder, the head, or the loss. We changed which tokens the box selects — the one stage where the leak occurs.",
    GRN, "F1F6F2");
  note(s, "40 s. This is the bridge: geometry enters at tiling time, so that is where we intervened.");
}

// ══════════════════════════════════════════════ ENCODER — il token
{
  const s = slide("What one latent vector actually contains", "the encoder  ·  representation", {
    m: "Target encoder (frozen)", c: "—",
    p: "3 blocks concatenated", k: "196 tokens × 1,152 d per lesion" });
  s.addText("Each 224 px window becomes a 14×14 grid of 196 tokens. We do not take the last block alone: we concatenate three depths, so every token is 1,152 numbers in three contiguous sections.", {
    x: M, y: TY, w: W - 2 * M, h: 0.48, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13.5, color: INK, lineSpacing: 20 });
  // il vettore, disegnato in scala
  const x0 = M, wtot = W - 2 * M, wsec = wtot / 3, y0 = 2.70;
  const sez = [["Block 2", "x₁ … x₃₈₄", DEEP], ["Block 7", "x₃₈₅ … x₇₆₈", TEAL], ["Block 11", "x₇₆₉ … x₁₁₅₂", GRN]];
  sez.forEach(([b, r, c], i) => {
    s.addShape(pres.ShapeType.rect, { x: x0 + i * wsec, y: y0, w: wsec - 0.04, h: 0.62,
      fill: { color: "FFFFFF" }, line: { color: c, width: 1.6 } });
    s.addText(r, { x: x0 + i * wsec, y: y0 + 0.16, w: wsec - 0.04, h: 0.3, isTextBox: true, margin: 0,
      fontFace: "Courier New", fontSize: 13, color: INK, align: "center" });
    s.addText(b, { x: x0 + i * wsec, y: y0 + 0.68, w: wsec - 0.04, h: 0.26, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 12, bold: true, color: c, align: "center" });
    s.addText("384 values", { x: x0 + i * wsec, y: y0 + 0.92, w: wsec - 0.04, h: 0.24, isTextBox: true,
      margin: 0, fontFace: F_B, fontSize: 10.5, color: MUT, align: "center" });
  });
  s.addText("tᵢ  ∈  ℝ¹¹⁵²        i ∈ [1, 196]", {
    x: M, y: 2.38, w: wtot, h: 0.26, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 13, bold: true, color: DEEP, align: "center" });
  nota(s, M, 4.15, 6.0, 2.3, "Why not the last block only",
    "The last block is the most COMPRESSED: it has discarded whatever the pre-training objective did not need. With a linear head — which is what the assignment's protocol uses — an intermediate block reads better.\n\nConcatenating three depths lets the trained pooling decide how much to weigh each, instead of us deciding in advance.",
    DEEP);
  nota(s, 6.85, 4.15, 5.87, 2.3, "The usual interpretation, and our caution",
    "Textbooks read depth as low-level texture → mid-level structure → high-level semantics. It is a reasonable motivation for taking three depths.\n\nWe do NOT present it as a measured fact about our encoder: we measured the depth profile instead, and it says something more specific — next slide.",
    AMB, "FDF7EC");
  note(s, "50 s. Show the vector, then say we measured what depth does rather than assuming it.");
}

// ══════════════════════════════════════════════ ENCODER — profilo di profondità
{
  const s = slide("What depth actually does, measured", "the encoder  ·  representation", {
    m: "Three encoders, 0 / 69 / 179 epochs", c: "One another, block by block",
    p: "k-NN probe, no trained parameters", k: "macro-F1, validation" });
  s.addText("A parameter-free k-NN probe read each block separately. The random encoder is the control: with untrained weights, depth should carry no structure — and it does not.", {
    x: M, y: TY, w: 7.4, h: 0.55, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13.5, color: INK, lineSpacing: 20 });
  tab(s, [
    [th("Encoder"), th("epochs"), th("Block 2"), th("Block 7"), th("Block 11"), th("spread")],
    [td("Random init", { bold: true }), td("0", { align: "right" }), td("0.7626", { align: "right" }), td("0.7639", { align: "right" }), td("0.7680", { align: "right" }), td("0.005", { color: MUT, align: "right" })],
    [td("I-JEPA finale"), td("69", { align: "right" }), td("0.6782", { bold: true, color: DEEP, align: "right" }), td("0.4043", { align: "right" }), td("0.3845", { color: RED, align: "right" }), td("0.294", { bold: true, color: DEEP, align: "right" })],
    [td("I-JEPA completa"), td("179", { align: "right" }), td("0.6400", { bold: true, color: DEEP, align: "right" }), td("0.4910", { align: "right" }), td("0.4892", { align: "right" }), td("0.151", { bold: true, color: DEEP, align: "right" })],
  ], { x: M, y: 2.6, w: 7.4, colW: [2.0, 0.85, 1.25, 1.25, 1.25, 0.8] });
  nota(s, M, 4.4, 7.4, 2.1, "Read the first row against the others",
    "On untrained weights the three blocks are indistinguishable — spread 0.005. On trained ones block 2 is worth nearly twice block 11.\n\nThis probe rewards geometry, so what the numbers say is precise: DEPTH PROGRESSIVELY DISCARDS THE SIZE INFORMATION. Block 2 still carries it; block 11 has largely shed it.",
    DEEP);
  nota(s, 8.25, TY, 4.47, 4.4, "Why this matters for our thesis",
    "It is the same story as the training trajectory, seen along a different axis.\n\nAcross EPOCHS the encoder unlearns the bounding-box shortcut. Across DEPTH it does the same thing within a single forward pass.\n\nThat is also the honest reason for concatenating three blocks rather than trusting the deepest one: they carry different information, and the trained pooling is better placed than we are to decide how much of each to use.\n\nAnd it is a caution about the textbook reading of depth — on this data the deepest block is not simply “the most useful one”.",
    GRN, "F1F6F2");
  note(s, "60 s. The random-encoder row is the control that makes the other two readable.");
}

// ══════════════════════════════════════════════ 8  LE TESTE
{
  const s = slide("Classification heads: three ways to read a bag", "method  ·  heads", {
    m: "Frozen encoder tokens", c: "Three head designs",
    p: "Both protocols", k: "PR-AUC PAI 5 · macro-F1" });
  tab(s, [
    [th("Head"), th("How it aggregates"), th("Trainable params"), th("Bag-size invariant?")],
    [td("Flat + attention pooling", { bold: true }), td("Learned weights over tokens, then one linear layer"), td("5.3 M"), td("no — pooled vector depends on the set", { color: RED })],
    [td("Gated attention pooling"), td("Same, with a gating branch (Ilse et al.)"), td("0.3 M"), td("no", { color: RED })],
    [td("Instance-level MIL", { bold: true }), td("Classify EVERY token, then average the probabilities", { bold: true }), td("0.6 M"), td("yes, by construction", { bold: true, color: GRN })],
  ], { x: M, y: TY, w: W - 2 * M, colW: [2.9, 4.6, 1.8, 2.8] });
  nota(s, M, 4.0, 6.0, 2.5, "Why instance-level MIL is the principled answer",
    "Embedding-level pooling builds one vector from N tokens, so N leaks in. Instance-level aggregation classifies each token and averages the decisions — and the mean of N probabilities does not depend on N.\n\nWe verified the invariance before measuring anything: varying the bag from 8 to 128 tokens moves the output by 6×10⁻⁸.",
    GRN, "F1F6F2");
  nota(s, 6.85, 4.0, 5.87, 2.5, "The hypothesis it falsified",
    "Under the prescribed protocol, MIL made the RANDOM encoder better, not worse: 0.7851 against 0.7705.\n\nThe proven invariance was real and insufficient. It removed the count but not the extent: tokens carry positional embeddings, so a large box includes peripheral positions. The mean changes not because the terms are more numerous, but because they are different ones.\n\nAn invariance proved on one channel does not protect against the others.",
    AMB, "FDF7EC");
  note(s, "55 s. Say the invariance was verified BEFORE measuring — it shows method, not luck.");
}

// ══════════════════════════════════════════════ 8b  LA TESTA E' MINUSCOLA
{
  const s = slide("The classifier is 3,459 parameters — and that is the point", "the head  ·  design choice", {
    m: "Linear 1,152 → 3", c: "Encoder: 21.6 M · pooling: 5.3 M",
    p: "Both protocols", k: "5 seeds per cell" });
  s.addText("The final stage takes the aggregated vector and maps it to three logits — one per PAI grade. A single fully-connected layer, nothing else.", {
    x: M, y: TY, w: 6.5, h: 0.55, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13.5, color: INK, lineSpacing: 20 });
  tab(s, [
    [th("Stage"), th("Parameters"), th("Share")],
    [td("Encoder (frozen)"), td("21,589,632", { align: "right" }), td("—", { color: MUT, align: "right" })],
    [td("Attention pooling"), td("5,316,480", { align: "right" }), td("99.9 %", { align: "right" })],
    [td("Classifier head", { bold: true }), td("3,459", { bold: true, color: GRN, align: "right" }), td("0.1 %", { bold: true, color: GRN, align: "right" })],
  ], { x: M, y: 2.6, w: 6.5, colW: [2.5, 2.2, 1.8] });
  s.addText("Of everything trained downstream, the classifier is one part in a thousand.", {
    x: M, y: 4.4, w: 6.5, h: 0.4, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 13, bold: true, color: DEEP });
  nota(s, 7.3, TY, 5.42, 2.5, "Small on purpose, not for lack of capacity",
    "A large head could reconstruct information the encoder had discarded, and the comparison would then measure the head, not the representation. Keeping it linear forces the answer to be already present in the frozen vector.",
    GRN, "F1F6F2");
  nota(s, 7.3, 4.85, 5.42, 1.8, "And what the 5 seeds change",
    "Head initialisation, batch order, and — for the sampling methods — which views are drawn. Not the encoder, not the split, not the test set.",
    DEEP);
  note(s, "40 s. The 0.1 % line is the one to say aloud: it is why the comparison is about the encoder.");
}

// ══════════════════════════════════════════════ 9  MIL RISULTATI
{
  const s = slide("Instance-level MIL isolates the encoder", "method  ·  heads  ·  result", {
    m: "I-JEPA (frozen, ep. 69)", c: "Random ViT (frozen)",
    p: "Instance-level MIL head", k: "macro-F1, test, 5 seeds" });
  s.addChart(pres.ChartType.bar, [
  /* Etichette su una riga: con "\n" pptxgenjs perde le categorie e stampa
   * 1, 2 sull'asse x. Qui il contrasto fra i due protocolli E' il messaggio,
   * quindi entrambi restano — ma l'asse si ferma a 0.85 invece che a 0.9. */
    { name: "Random ViT", labels: ["Prescribed protocol", "Fixed count K = 16"], values: [0.7851, 0.4014] },
    { name: "I-JEPA", labels: ["Prescribed protocol", "Fixed count K = 16"], values: [0.7570, 0.5171] },
  ], { x: M, y: TY, w: 6.9, h: 3.6, barDir: "col", barGrouping: "clustered",
    chartColors: [MUT, GRN], showTitle: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFormatCode: "0.000",
    showLegend: true, legendPos: "t", legendFontSize: 11, legendColor: INK,
    catAxisLabelColor: INK, catAxisLabelFontSize: 11,
    valAxisLabelColor: MUT, valAxisLabelFontSize: 10, valAxisLabelFormatCode: "0.0",
    valAxisMaxVal: 0.85, valAxisMinVal: 0, valAxisMajorUnit: 0.2,
    valGridLine: { color: "DDE3E6", size: 0.75 }, catGridLine: { style: "none" } });
  /* Una didascalia sola: la seconda riga che avevo aggiunto ripeteva lo stesso
   * concetto e le due caselle si toccavano. */
  s.addText("Same head, same encoders, same seeds — the only difference is whether bag size varies with the class.", {
    x: M, y: 5.6, w: 6.9, h: 0.4, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 12.5, italic: true, color: DEEP });
  nota(s, 7.75, TY, 4.97, 2.3, "With the bias present",
    "The random encoder wins by 0.0282 (z = −5.76). The head reads position, and position encodes size.",
    RED, "FBF3F2");
  nota(s, 7.75, 4.35, 4.97, 2.35, "With the bias removed",
    "The random encoder collapses to 0.4014 while I-JEPA holds at 0.5171: +0.1158, z = +6.49, a relative gain of +28.8 %.\n\nAsked “how much PAI-5 tissue is this single token?”, a random projection cannot answer and a pre-trained one can. This is the largest margin we measured.",
    GRN, "F1F6F2");
  note(s, "60 s. Left bars: bias present, random wins. Right bars: bias gone, gap opens.");
}

// ══════════════════════════════════════════════ 10  LA NOVITÀ
{
  const s = slide("Our novelty: balanced token sampling", "method  ·  novelty", {
    m: "Any frozen encoder", c: "Focal · class-weighted · oversampling · SMOTE",
    p: "Both protocols", k: "PR-AUC PAI 5" });
  s.addText("The assignment asks for a novelty acting in latent space. Ours rebalances by sampling different subsets of a lesion's own tokens — each subset becomes a training instance.", {
    x: M, y: TY, w: 7.2, h: 0.6, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 14, color: INK, lineSpacing: 20 });
  tab(s, [
    [th("Method"), th("What it multiplies"), th("Operates in latent space?"), th("Invents data?")],
    [td("Random oversampling"), td("identical copies of the same vector"), td("no", { color: RED }), td("no")],
    [td("Focal / class-weighted"), td("nothing — reweights the loss"), td("no", { color: RED }), td("no")],
    [td("SMOTE"), td("interpolated synthetic points"), td("yes"), td("yes", { color: RED })],
    [td("Balanced token sampling", { bold: true }), td("genuine distinct views of the same lesion", { bold: true }), td("yes", { bold: true, color: GRN }), td("no", { bold: true, color: GRN })],
  ], { x: M, y: 2.6, w: 7.2, colW: [2.2, 2.9, 1.3, 0.8] });
  s.addText([{ text: "α", options: { bold: true } }, { text: " sets views per class:  n", options: {} },
    { text: "c", options: { fontSize: 9 } }, { text: " = ⌈(max/n", options: {} }, { text: "c", options: { fontSize: 9 } },
    { text: ")", options: {} }, { text: "α", options: { fontSize: 9 } },
    { text: "⌉ — at α = 0.5, 6,894 instances against 4,719.", options: {} }],
    { x: M, y: 4.9, w: 7.2, h: 0.35, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 12.5, color: INK });
  nota(s, 8.05, TY, 4.67, 2.5, "Why the views are genuine",
    "Every view is built from real tokens of the same lesion, so nothing is invented and no vector is duplicated. The classifier sees one lesion described several genuinely different ways — which is augmentation, not replication.",
    GRN, "F1F6F2");
  nota(s, 8.05, 4.55, 4.67, 2.1, "The control we owed",
    "The method rebalances AND augments at once. A budget-matched uniform variant separates the two — without it we could not attribute the gain.",
    DEEP);
  note(s, "50 s. The table is the argument: only our method is both latent-space and non-inventing.");
}

// ══════════════════════════════════════════════ 11  NOVITÀ RISULTATI
{
  const s = slide("Novelty: ablation against every baseline", "results  ·  novelty", {
    m: "Random ViT (frozen)", c: "Four imbalance baselines",
    p: "As prescribed (bbox)", k: "PR-AUC PAI 5, test, 5 seeds" });
  tab(s, [
    [th("Method"), th("PR-AUC PAI 5"), th("Recall 5"), th("Precision 5")],
    [td("Balanced token sampling", { bold: true }), td("0.8826 ± 0.0050", { bold: true, color: GRN, align: "right" }), td("0.771", { align: "right" }), td("0.789", { bold: true, color: GRN, align: "right" })],
    [td("Focal loss"), td("0.8730 ± 0.0117", { align: "right" }), td("0.791", { align: "right" }), td("0.784", { align: "right" })],
    [td("Class-weighted CE"), td("0.8706 ± 0.0091", { align: "right" }), td("0.798", { align: "right" }), td("0.737", { align: "right" })],
    [td("Plain cross-entropy"), td("0.8676 ± 0.0065", { align: "right" }), td("0.755", { align: "right" }), td("0.774", { align: "right" })],
    [td("Random oversampling"), td("0.8658 ± 0.0142", { align: "right" }), td("0.798", { align: "right" }), td("0.728", { color: RED, align: "right" })],
  ], { x: M, y: TY, w: 6.9, colW: [2.4, 1.9, 1.3, 1.3] });
  s.addText("+0.0168 over oversampling (z = 2.50) and +0.0150 over plain CE (z = 4.11) — and it wins without the recall-for-precision trade the others make.", {
    x: M, y: 4.4, w: 6.9, h: 0.55, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 12.5, color: INK, lineSpacing: 18 });
  nota(s, M, 5.1, 6.9, 1.5, "Declared before being asked",
    "On F1 of the rare class focal loss edges ahead (0.7863 vs 0.7794). Our win is on the primary metric, not on all of them. And the α sweep ran on a fixed encoder, to isolate the parameter.",
    AMB, "FDF7EC");
  note(s, "40 s. Read the first row, then the precision column: that is where the novelty differs from the others.");
}

// ══════════════════════════════════════════════ 11b  LO SWEEP DI ALPHA
{
  const s = slide("The α sweep: more rebalancing is not better", "results  ·  objective 4", {
    m: "Random ViT (frozen)", c: "Four values of α",
    p: "As prescribed (bbox)", k: "PR-AUC PAI 5, test" });
  s.addText("α sets how many views each class receives. Screening on 3 seeds, then the two finalists re-measured on 5 seeds disjoint from the screening ones.", {
    x: M, y: TY, w: 6.4, h: 0.55, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13.5, color: INK, lineSpacing: 20 });
  tab(s, [
    [th("α"), th("views per class"), th("instances/epoch"), th("PR-AUC PAI 5")],
    [td("0.25"), td("1 / 1 / 2"), td("6,421", { align: "right" }), td("0.8793", { align: "right" })],
    [td("0.50", { bold: true }), td("1 / 2 / 3", { bold: true }), td("6,894", { align: "right" }), td("0.8814", { bold: true, color: GRN, align: "right" })],
    [td("0.75"), td("1 / 2 / 5"), td("7,840", { align: "right" }), td("0.8747", { align: "right" })],
    [td("1.00"), td("1 / 3 / 7"), td("10,015", { align: "right" }), td("0.8689", { color: RED, align: "right" })],
  ], { x: M, y: 2.6, w: 6.4, colW: [0.85, 1.85, 1.85, 1.85] });
  s.addText("α = 0.5 beats α = 1.0 by +0.0125, at 2.2 standard errors. Doubling the instances makes it worse.", {
    x: M, y: 4.8, w: 6.4, h: 0.55, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13,
    bold: true, color: DEEP, lineSpacing: 19 });
  nota(s, 7.55, TY, 5.17, 2.6, "Why the optimum is interior",
    "The views are subsets of the SAME lesion, so they are strongly correlated. With ICC ρ = 0.9864, seven views of a PAI 5 are worth 1.01 independent samples: they move the decision boundary without adding information.",
    DEEP);
  nota(s, 7.55, 5.0, 5.17, 1.6, "Reproducible three times",
    "α = 0.5 was measured independently three times — 0.8813 / 0.8814 / 0.8797 — with a maximum spread of 0.0017.",
    GRN, "F1F6F2");
  note(s, "40 s. The interior optimum is the interesting part: it is explained, not just observed.");
}

// ══════════════════════════════════════════════ 12  RISULTATO PRINCIPALE
{
  const s = slide("What the frozen encoder is worth, once the bias is gone", "results  ·  objective 2", {
    m: "I-JEPA (frozen, ep. 69)", c: "Random ViT (frozen)",
    p: "Fixed count K ∈ {16, 36, 64}", k: "PR-AUC PAI 5, test, 5 seeds" });
  s.addChart(pres.ChartType.bar, [
  /* IL PROTOCOLLO VIZIATO NON ENTRA IN QUESTO GRAFICO. Le sue barre stanno a
   * 0.87 e schiacciano in un terzo dell'altezza proprio l'intervallo dove sta
   * il risultato (0.36-0.51). Resta nella tabella a destra, dove serve come
   * controllo; qui toglierlo permette all'asse di aprirsi su 0-0.55.
   *
   * E le etichette NON hanno "\n": i ritorni a capo rompevano il parsing delle
   * categorie e pptxgenjs ripiegava sugli indici 1..5 sull'asse x. */
    { name: "Random ViT", labels: ["Fixed grid", "K = 16", "K = 36", "K = 64"], values: [0.3621, 0.3861, 0.3823, 0.3740] },
    { name: "I-JEPA", labels: ["Fixed grid", "K = 16", "K = 36", "K = 64"], values: [0.3792, 0.4713, 0.5031, 0.4553] },
  ], { x: M, y: TY, w: 7.3, h: 3.75, barDir: "col", barGrouping: "clustered", chartColors: [MUT, GRN],
    showTitle: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10,
    dataLabelColor: INK, dataLabelFormatCode: "0.000",
    showLegend: true, legendPos: "t", legendFontSize: 11, legendColor: INK,
    catAxisLabelColor: INK, catAxisLabelFontSize: 11,
    valAxisLabelColor: MUT, valAxisLabelFontSize: 10, valAxisLabelFormatCode: "0.0",
    valAxisMaxVal: 0.55, valAxisMinVal: 0, valAxisMajorUnit: 0.1,
    valGridLine: { color: "DDE3E6", size: 0.75 }, catGridLine: { style: "none" } });
  s.addText("The prescribed protocol is omitted here: at 0.87 its bars compress the range where the result lives. It is in the table on the right.", {
    x: M, y: 5.72, w: 7.3, h: 0.4, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11,
    italic: true, color: MUT });
  tab(s, [
    [th("Protocol"), th("Relative gain"), th("z")],
    [td("Prescribed"), td("+0 %", { color: MUT, align: "right" }), td("+0.84", { color: MUT, align: "right" })],
    [td("Fixed grid"), td("+5 %", { color: MUT, align: "right" }), td("+0.78", { color: MUT, align: "right" })],
    [td("K = 16"), td("+22 %", { bold: true, color: GRN, align: "right" }), td("+5.11", { bold: true, color: GRN, align: "right" })],
    [td("K = 36"), td("+32 %", { bold: true, color: GRN, align: "right" }), td("+8.72", { bold: true, color: GRN, align: "right" })],
    [td("K = 64"), td("+22 %", { bold: true, color: GRN, align: "right" }), td("+5.60", { bold: true, color: GRN, align: "right" })],
  ], { x: 8.15, y: TY, w: 4.57, colW: [1.87, 1.5, 1.2] });
  nota(s, 8.15, 4.1, 4.57, 2.5, "Not one lucky K",
    "Significant on all three values, so the result survives a fourfold change of window size. The two grey rows are the controls, where by construction nothing should appear — and nothing does.\n\nOn PR-AUC of the minority class the gain reaches +32 % at z = 8.72.",
    GRN, "F1F6F2");
  note(s, "60 s. Grey bars random, green I-JEPA. First two groups tie; the K groups do not.");
}

// ══════════════════════════════════════════════ 13  L'ENCODER IMPARA?
{
  const s = slide("Does the encoder actually learn? Measured during training", "results  ·  objective 1", {
    m: "I-JEPA, 279 epochs", c: "Itself, read two ways",
    p: "Both, same probe", k: "macro-F1, validation" });
  s.addText("The same probe read both protocols every ten epochs — 28 measurements, one encoder, identical moments. This is not a comparison between models; it is a comparison between two ways of reading one model.", {
    x: M, y: TY, w: 7.3, h: 0.65, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 13, color: INK, lineSpacing: 19 });
  tab(s, [
    [th("Series"), th("first 10 probes"), th("last 10"), th("Δ"), th("z")],
    [td("Quality — fixed count"), td("0.5406", { align: "right" }), td("0.5482", { align: "right" }), td("+0.0075", { align: "right" }), td("+1.15", { color: MUT, align: "right" })],
    [td("Shortcut — bag size", { bold: true }), td("0.7539", { align: "right" }), td("0.7271", { align: "right" }), td("−0.0268", { bold: true, color: RED, align: "right" }), td("−11.27", { bold: true, color: RED, align: "right" })],
    [td("Effective rank"), td("2.91", { align: "right" }), td("14.79", { align: "right" }), td("×5.1", { color: GRN, align: "right" }), td("—", { align: "right" })],
  ], { x: M, y: 2.75, w: 7.3, colW: [2.4, 1.45, 1.05, 1.2, 1.2] });
  nota(s, M, 4.6, 7.3, 1.85, "What the three rows say together",
    "The representation keeps changing — effective rank grows fivefold. Quality holds flat. And readability of the bag-size shortcut falls at eleven standard errors.\n\nI-JEPA reaches its quality by epoch 69 and spends the next two hundred epochs partly unlearning the shortcut.",
    GRN, "F1F6F2");
  tab(s, [
    [th("Fixed count K = 16"), th("epoch 69\nselected"), th("epoch 288\nnot selected")],
    [td("PR-AUC PAI 5"), td("0.4713", { bold: true, color: GRN, align: "right" }), td("0.4495", { bold: true, color: GRN, align: "right" })],
    [td("gain vs random"), td("+22 %", { color: GRN, align: "right" }), td("+16 %", { color: GRN, align: "right" })],
    [td("z"), td("+5.11", { align: "right" }), td("+3.15", { align: "right" })],
    [td("macro-F1 z"), td("+3.12", { align: "right" }), td("+2.14", { color: AMB, align: "right" })],
  ], { x: 8.15, y: TY, w: 4.57, colW: [1.87, 1.35, 1.35] });
  nota(s, 8.15, 4.25, 4.57, 2.35, "Not an artefact of checkpoint choice",
    "We also measured the last-epoch encoder, which no criterion selected. It still beats the random baseline on the primary metric.\n\nOn macro-F1 its z = 2.14 falls below our own 2.31 threshold — we say so rather than round it away.",
    DEEP);
  note(s, "60 s. Stress: one encoder, two readings. Then the unselected checkpoint.");
}

// ══════════════════════════════════════════════ 14  SETUP
{
  const s = slide("Experimental setup and statistical discipline", "how it was configured");
  tab(s, [
    [th("Stage"), th("Configuration")],
    [td("Pre-training", { bold: true }), td("ViT-S/16 · context + EMA target + shallow predictor (4 blocks, dim 96) · tile 224 px, batch 128, bf16\nlr 3e-5 · EMA τ 0.9996 → 1.0 · 289 of 300 scheduled epochs — best AND final encoder both extracted and evaluated · smooth-L1 on layer-normed target representations")],
    [td("Downstream", { bold: true }), td("Encoder FROZEN, as the assignment requires · attention pooling over selected tokens · layers 2, 7, 11 concatenated (1,152-d) · 5 seeds per cell")],
    [td("Metrics", { bold: true }), td("Primary: PR-AUC on PAI 5 — minority-specific and threshold-agnostic. Reported alongside: macro-F1, quadratic-weighted kappa. Global accuracy excluded by the assignment.")],
    [td("Significance", { bold: true }), td("|z| ≥ 2.31 — Student's t at 8 d.o.f., not 1.96: with five repetitions the normal underestimates the tail")],
  ], { x: M, y: 1.3, w: W - 2 * M, colW: [1.9, 10.2], fontSize: 11 });
  note(s, "35 s. Do not read the table — point at the three lines that matter: frozen, five seeds, PR-AUC on the minority class.");
}

// ══════════════════════════════════════════════ 14b  DISCIPLINA STATISTICA
{
  const s = slide("Statistical discipline", "how it was configured");
  s.addText("Three safeguards against reporting a number we could not defend.", {
    x: M, y: 1.3, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 14, color: INK });
  const g = [
    ["Selection never touches the test set",
     "Checkpoint chosen on validation. Learning rate chosen on validation — 3e-4 measured worse there: −0.0304, z = −2.98.", DEEP],
    ["Disjoint seeds for the α sweep",
     "The two finalists were re-measured on five seeds disjoint from the screening ones. Selecting and reporting on the same seeds inflates the winner by about one standard deviation.", DEEP],
    ["The ± is reproducibility, not generalisation",
     "It is the spread across five seeds. It does NOT capture test-set sampling error, which with 112 PAI-5 lesions is roughly 0.021 — ten times larger.", AMB],
  ];
  let y = 1.95;
  g.forEach((r, i) => {
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.06, w: 0.36, h: 0.36, fill: { color: r[2] } });
    s.addText(String(i + 1), { x: M, y: y + 0.06, w: 0.36, h: 0.36, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(r[0], { x: M + 0.55, y: y, w: 11.4, h: 0.4, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 15, bold: true, color: r[2] });
    s.addText(r[1], { x: M + 0.55, y: y + 0.42, w: 11.4, h: 0.7, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 12.5, color: MUT, lineSpacing: 17 });
    y += 1.42;
  });
  s.addText("Differences below 0.021 are reproducible — not proven to generalise. We say which is which.", {
    x: M, y: 6.15, w: W - 2 * M, h: 0.45, isTextBox: true, margin: 0,
    fontFace: F_H, fontSize: 15, italic: true, bold: true, color: DEEP });
  note(s, "35 s. Point 3 pre-empts the sharpest question about our error bars — say it before they ask.");
}

// ══════════════════════════════════════════════ 15  CONCLUSIONI
{
  const s = slide("Conclusions", "final considerations and future work");
  const p = [
    ["1", "The task is geometric, so the prescribed protocol carries a bag-size bias.",
     "The bounding-box mask alone gives macro-F1 0.7708; a random encoder with the full image gives 0.7705. Lesion area is the PAI criterion, so token count is nearly the label.", RED],
    ["2", "Removing the bias while keeping localisation reveals the encoder.",
     "Up to +32 % PR-AUC on the minority class (z = 8.72), on all three values of K, with the unselected checkpoint too, and +28.8 % with an instance-level MIL head.", GRN],
    ["3", "Our latent-space novelty beats every baseline on the primary metric.",
     "+0.0168 over oversampling (z = 2.50), with an interior optimum at α = 0.5 that the ICC of the views explains.", GRN],
  ];
  let y = 1.35;
  p.forEach(r => {
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.04, w: 0.34, h: 0.34, fill: { color: r[3] } });
    s.addText(r[0], { x: M, y: y + 0.04, w: 0.34, h: 0.34, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(r[1], { x: M + 0.5, y: y, w: 7.0, h: 0.42, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 14, bold: true, color: DEEP });
    s.addText(r[2], { x: M + 0.5, y: y + 0.42, w: 7.0, h: 0.75, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 11.5, color: MUT, lineSpacing: 16 });
    y += 1.35;
  });
  /* Il ciclo di pre-training e' un RISULTATO, non un limite: 289 epoche su
   * 300, e l'encoder finale non e' un'incognita — lo abbiamo estratto e
   * misurato. Elencarlo fra i limiti suggeriva una run monca. */
  s.addText([
    { text: "Training schedule completed.  ", options: { bold: true, color: DEEP } },
    { text: "289 of the 300 scheduled epochs, halted by our own power limiter. The last 11 change nothing measurable — quality had plateaued by epoch 69 (z = +1.15 across the whole run) — and we did not have to assume it: we extracted the final epoch-288 encoder and measured it. It still beats the random baseline by +16 % on the primary metric (z = 3.15).", options: { color: INK } },
  ], { x: M, y: 5.4, w: 7.4, h: 0.85, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11.5, lineSpacing: 16 });
  s.addText("There is no “best model” — only the best model for a given reading, and choosing that reading is the work.", {
    x: M, y: 6.35, w: 11.5, h: 0.5, isTextBox: true, margin: 0, fontFace: F_H, fontSize: 15,
    italic: true, bold: true, color: DEEP });
  note(s, "50 s. Three numbered claims, then the closing line. Do not add anything after it.");
}

// ══════════════════════════════════════════════ 15b  LIMITI E FUTURO
{
  const s = slide("Declared limitations, and what comes next", "final considerations");
  const col = (x, tit, voci, c, bg) => {
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.35, w: 5.9, h: 0.5, fill: { color: bg }, rectRadius: 0.06 });
    s.addText(tit, { x: x + 0.2, y: 1.45, w: 5.5, h: 0.32, isTextBox: true, margin: 0,
      fontFace: F_H, fontSize: 16, bold: true, color: c });
    let y = 2.05;
    voci.forEach(v => {
      s.addShape(pres.ShapeType.ellipse, { x: x + 0.2, y: y + 0.07, w: 0.15, h: 0.15, fill: { color: c } });
      s.addText(v[0], { x: x + 0.5, y: y, w: 5.2, h: 0.32, isTextBox: true, margin: 0,
        fontFace: F_B, fontSize: 13, bold: true, color: INK });
      s.addText(v[1], { x: x + 0.5, y: y + 0.32, w: 5.2, h: 0.6, isTextBox: true, margin: 0,
        fontFace: F_B, fontSize: 11.5, color: MUT, lineSpacing: 16 });
      y += 1.1;
    });
  };
  col(M, "Declared limitations", [
    ["Checkpoint selection", "Better on the criterion we selected it with, worse on size-blind readings — the same bias we denounce."],
    ["α sweep on a fixed encoder", "Deliberate, to isolate the parameter — but it means the sweep is not encoder-agnostic."],
    ["± is not generalisation", "Spread across seeds; test sampling error is ten times larger."],
    ["SMOTE not measured", "Not among the baselines of the final grid."],
  ], AMB, "FDF7EC");
  col(6.85, "Future work", [
    ["Remove the box from inference", "Detection and grading end to end. In real use nobody draws the bounding box — drawing it is already the diagnosis."],
    ["Per-token MIL as the default head", "It gives the largest margin we measured: +28.8 % on fixed count."],
    ["Explicit anti-collapse term", "Variance-covariance regularisation, instead of relying on the EMA alone."],
  ], GRN, "F1F6F2");
  note(s, "40 s. Read the left column without softening it. The right column is short on purpose.");
}

// ══════════════════════════════════════════════ 16  REFERENCES
{
  const s = slide("References", "where we drew inspiration from");
  const refs = [
    ["Assran, M. et al.", "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). CVPR 2023.", "The architecture the assignment prescribes"],
    ["Ilse, M., Tomczak, J., Welling, M.", "Attention-based Deep Multiple Instance Learning. ICML 2018.", "MIL framing, attention pooling, bag-size bias"],
    ["Do, H. V. et al.", "A Dataset of apical periodontitis lesions in panoramic radiographs. Data in Brief 54:110486, 2024.", "Dataset — Mendeley DOI 10.17632/kx52tk2ddj.3"],
    ["He, K. et al.", "Masked Autoencoders Are Scalable Vision Learners. CVPR 2022.", "Pixel-reconstruction alternative we argue against"],
    ["Grill, J.-B. et al.", "Bootstrap Your Own Latent (BYOL). NeurIPS 2020.", "EMA target networks and collapse avoidance"],
    ["Lin, T.-Y. et al.", "Focal Loss for Dense Object Detection. ICCV 2017.", "Imbalance baseline"],
    ["Chawla, N. V. et al.", "SMOTE: Synthetic Minority Over-sampling Technique. JAIR 16, 2002.", "Imbalance baseline in feature space"],
    ["Saito, T., Rehmsmeier, M.", "The precision-recall plot is more informative than the ROC plot on imbalanced datasets. PLoS ONE 10(3), 2015.", "Justifies PR-AUC as the primary metric"],
  ];
  let y = 1.3;
  refs.forEach(r => {
    s.addText([{ text: r[0] + "  ", options: { bold: true, color: DEEP } }, { text: r[1], options: { color: INK } }],
      { x: M, y: y, w: 8.3, h: 0.44, isTextBox: true, margin: 0, fontFace: F_B, fontSize: 11, lineSpacing: 14 });
    s.addText(r[2], { x: 9.1, y: y, w: 3.62, h: 0.44, isTextBox: true, margin: 0,
      fontFace: F_B, fontSize: 10, italic: true, color: MUT, lineSpacing: 13 });
    y += 0.63;
  });
  note(s, "15 s. Leave on screen for questions.");
}

pres.writeFile({ fileName: "Project8_CV_2025-2026.pptx" })
  .then(f => console.log("scritto:", f, "-", n, "slide"));
