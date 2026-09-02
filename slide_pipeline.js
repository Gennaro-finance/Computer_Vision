/*
 * Genera UNA SOLA slide: lo schema a blocchi della pipeline.
 *
 * File separato apposta. Il deck principale la numera automaticamente, quindi
 * inserirla a mano da PowerPoint spezzerebbe i numeri "corrente / totale" di
 * tutte le slide successive - che le Exam guidelines richiedono. Qui la slide
 * esce SENZA numero di pagina: si guarda, si decide dove va, e poi si sposta
 * il blocco dentro build_deck.js nel punto scelto, dove la numerazione si
 * rigenera da sola.
 *
 *   node slide_pipeline.js     ->  Pipeline_schema.pptx
 */
const P = require("pptxgenjs");

const INK = "111418", MUT = "58626C", LINE = "D6DBDF";
const DEEP = "0B3C49", TEAL = "1C7293";
const RED = "A62B1F", GRN = "1F6F43";
const F_H = "Cambria", F_B = "Calibri";

const pres = new P();
pres.layout = "LAYOUT_WIDE";
pres.title = "Pipeline — schema a blocchi";
const W = 13.333, H = 7.5, M = 0.62;

const s = pres.addSlide();
s.background = { color: "FFFFFF" };

s.addText("THE MAP BEFORE THE DETAILS", {
  x: M, y: 0.28, w: 10.5, h: 0.24, isTextBox: true, margin: 0,
  fontFace: F_B, fontSize: 10.5, bold: true, color: TEAL, charSpacing: 1.4 });
s.addText("The pipeline, end to end", {
  x: M, y: 0.54, w: W - 2 * M, h: 0.62, isTextBox: true, margin: 0,
  fontFace: F_H, fontSize: 27, bold: true, color: DEEP });

/* Due bande: sopra il pre-training che PRODUCE l'encoder, sotto il
 * downstream che lo USA. In mezzo la linea del congelamento. */
const bloc = (x, y, w, h, tit, sub, col, bg) => {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h,
    fill: { color: bg || "FFFFFF" }, line: { color: col, width: 1.6 }, rectRadius: 0.06 });
  s.addText(tit, { x: x + 0.06, y: y + 0.1, w: w - 0.12, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 11.5, bold: true, color: col, align: "center" });
  s.addText(sub, { x: x + 0.06, y: y + 0.38, w: w - 0.12, h: h - 0.46, isTextBox: true, margin: 0,
    fontFace: F_B, fontSize: 9.5, color: INK, align: "center", lineSpacing: 12 });
};
const fre = (x, y, w) => s.addShape(pres.ShapeType.line,
  { x, y, w, h: 0, line: { color: MUT, width: 1.4, endArrowType: "triangle" } });

// ---------------------------------------------- banda 1: pre-training
s.addText("1 — SELF-SUPERVISED PRE-TRAINING     ·     no labels used", {
  x: M, y: 1.28, w: 8, h: 0.24, isTextBox: true, margin: 0,
  fontFace: F_B, fontSize: 10, bold: true, color: MUT, charSpacing: 0.8 });
bloc(M, 1.58, 2.15, 0.95, "224 px tiles", "sampled anywhere\non the radiograph", DEEP);
fre(M + 2.15, 2.05, 0.32);
bloc(M + 2.47, 1.58, 2.15, 0.95, "Block masking", "1 context block\n+ 4 target blocks", DEEP);
fre(M + 4.62, 2.05, 0.32);
bloc(M + 4.94, 1.58, 2.15, 0.95, "Context encoder", "ViT-S/16 · 21.6 M\ntrained by gradient", DEEP);
fre(M + 7.09, 2.05, 0.32);
bloc(M + 7.41, 1.58, 2.15, 0.95, "Predictor", "4 blocks · dim 96\n2.4 % of an encoder", GRN);
fre(M + 9.56, 2.05, 0.32);
bloc(M + 9.88, 1.58, 2.22, 0.95, "smooth-L1 loss", "on LayerNormed target\nrepresentations, not pixels", GRN);

bloc(M + 4.94, 2.78, 2.15, 0.8, "Target encoder", "EMA copy · no gradient", TEAL, "F1F7F8");
s.addShape(pres.ShapeType.line, { x: M + 6.02, y: 2.53, w: 0, h: 0.24,
  line: { color: TEAL, width: 1.3, dashType: "dash", endArrowType: "triangle" } });
s.addText("EMA  τ = 0.9996", { x: M + 7.2, y: 3.0, w: 1.9, h: 0.24, isTextBox: true, margin: 0,
  fontFace: F_B, fontSize: 9.5, color: TEAL });

// ---------------------------------------------- la linea del congelamento
s.addShape(pres.ShapeType.line, { x: M, y: 3.82, w: W - 2 * M, h: 0,
  line: { color: TEAL, width: 2, dashType: "dash" } });
s.addText("the encoder is FROZEN from here down — no backbone weight moves again", {
  x: M, y: 3.86, w: W - 2 * M, h: 0.24, isTextBox: true, margin: 0,
  fontFace: F_B, fontSize: 10, bold: true, color: TEAL, align: "center" });

// ---------------------------------------------- banda 2: downstream
s.addText("2 — DOWNSTREAM CLASSIFICATION     ·     labels used here, and only here", {
  x: M, y: 4.28, w: 8, h: 0.24, isTextBox: true, margin: 0,
  fontFace: F_B, fontSize: 10, bold: true, color: MUT, charSpacing: 0.8 });
bloc(M, 4.58, 2.15, 0.95, "Lesion window", "224 px, centred\nscale preserved", DEEP);
fre(M + 2.15, 5.05, 0.32);
bloc(M + 2.47, 4.58, 2.15, 0.95, "Frozen encoder", "196 tokens × 1,152\nblocks 2, 7, 11", TEAL, "F1F7F8");
fre(M + 4.62, 5.05, 0.32);
bloc(M + 4.94, 4.58, 2.15, 0.95, "Token selection", "the bounding box\nchooses which tokens", RED, "FBF3F2");
fre(M + 7.09, 5.05, 0.32);
bloc(M + 7.41, 4.58, 2.15, 0.95, "Pooling + head", "5.3 M + 3,459 par.\nthe only trained part", DEEP);
fre(M + 9.56, 5.05, 0.32);
bloc(M + 9.88, 4.58, 2.22, 0.95, "PR-AUC on PAI 5", "5 seeds · test split\nnever seen before", DEEP);

// ---------------------------------------------- il punto che è nostro
s.addShape(pres.ShapeType.roundRect, { x: M, y: 5.85, w: W - 2 * M, h: 0.95,
  fill: { color: "FBF3F2" }, rectRadius: 0.07 });
s.addText("One stage is ours", { x: M + 0.2, y: 5.99, w: 4, h: 0.3, isTextBox: true, margin: 0,
  fontFace: F_H, fontSize: 14.5, bold: true, color: RED });
s.addText("Everything here is the assignment's pipeline except the red block. The bounding box enters exactly once, and that is where the bag-size bias lives — so that is the only stage we changed.", {
  x: M + 0.2, y: 6.33, w: W - 2 * M - 0.4, h: 0.4, isTextBox: true, margin: 0,
  fontFace: F_B, fontSize: 12, color: INK, lineSpacing: 18 });

s.addNotes("45 s. Trace it left to right with a finger, twice: the top band builds the encoder, the bottom band uses it. Then point at the red block.\n\nNO PAGE NUMBER: this slide is generated on its own. To put it in the deck, move this block into build_deck.js at the chosen position — the numbering regenerates there.");

pres.writeFile({ fileName: "Pipeline_schema.pptx" })
  .then(f => console.log("scritto:", f, "- 1 slide, senza numero di pagina"));
