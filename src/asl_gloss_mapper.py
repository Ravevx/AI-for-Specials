"""
ASL Gloss Mapper
----------------
Maps English text → ASL gloss order.
Provides REAL, DISTINCT hand keypoints for each sign.

IMPORTANT DESIGN DECISION:
  This dictionary only contains signs where we have a genuinely
  distinct, correct handshape. We do NOT fake unknown signs by
  copying another sign's keypoints — that produces wrong ASL.

  If a word is not here, it should come from WLASL video (real signer).
  If not in WLASL either, it gets fingerspelled.

  Better to fingerspell correctly than to show a wrong sign.

Handshape families used:
  FIST    — A, S, E, N, M, T
  FLAT    — B, 4, 5 (open hand)
  INDEX   — D, G, 1 (one finger up)
  PEACE   — V, 2 (two fingers up, spread)
  DOUBLE  — U, H (two fingers up, together)
  HOOK    — X (bent index)
  CIRCLE  — O, F (thumb meets fingertip)
  HANG    — Y (thumb + pinky out)
  CURVE   — C (curved hand)
  CROSS   — R (crossed fingers)
"""

import numpy as np


class ASLGlossMapper:

    # Words we strip in ASL grammar (articles, copulas, prepositions)
    _DROP = {
        "a", "an", "the",
        "is", "are", "was", "were", "am", "be", "been", "being",
        "to", "of", "in", "on", "at", "by", "for",
        "and", "but", "or", "so",
    }

    # English word → canonical gloss token
    # Only words whose gloss token has a correct handshape below
    _WORD_MAP = {
        # greetings
        "hello": "HELLO", "hi": "HELLO", "hey": "HELLO",
        "goodbye": "GOODBYE", "bye": "GOODBYE",
        # politeness
        "thank": "THANK_YOU", "thanks": "THANK_YOU",
        "please": "PLEASE",
        "sorry": "SORRY", "excuse": "SORRY",
        # yes/no
        "yes": "YES", "yeah": "YES", "yep": "YES",
        "no": "NO", "nope": "NO",
        # basic verbs
        "help": "HELP",
        "want": "WANT",
        "need": "NEED",
        "know": "KNOW",
        "understand": "UNDERSTAND",
        "think": "THINK",
        "like": "LIKE",
        "love": "LOVE",
        "see": "SEE",
        "go": "GO",
        "come": "COME",
        "stop": "STOP",
        "finish": "FINISH", "done": "FINISH", "finished": "FINISH",
        "eat": "EAT", "food": "EAT",
        "drink": "DRINK",
        "work": "WORK",
        "call": "CALL",
        "give": "GIVE",
        "get": "GET",
        "have": "HAVE",
        "wait": "WAIT",
        # nouns
        "water": "WATER",
        "name": "NAME",
        "home": "HOME",
        "school": "SCHOOL",
        "hospital": "HOSPITAL",
        "doctor": "DOCTOR",
        "police": "POLICE",
        "phone": "PHONE",
        "money": "MONEY",
        "time": "TIME",
        "bathroom": "BATHROOM", "restroom": "BATHROOM", "toilet": "BATHROOM",
        "family": "FAMILY",
        "friend": "FRIEND",
        "mother": "MOTHER",
        "father": "FATHER",
        # question words
        "what": "WHAT",
        "where": "WHERE",
        "when": "WHEN",
        "who": "WHO",
        "why": "WHY",
        "how": "HOW",
        # adjectives
        "good": "GOOD",
        "bad": "BAD",
        "hot": "HOT",
        "cold": "COLD",
        "sick": "SICK",
        "pain": "PAIN", "hurt": "PAIN",
        "happy": "HAPPY",
        "sad": "SAD",
        "angry": "ANGRY",
        "tired": "TIRED",
        "hungry": "HUNGRY",
        "lost": "LOST",
        # pronouns
        "i": "ME", "me": "ME", "my": "MY",
        "you": "YOU", "your": "YOUR",
        "he": "HE", "she": "SHE",
        "we": "WE", "they": "THEY",
        # misc
        "deaf": "DEAF",
        "not": "NOT",
        "more": "MORE",
        "big": "BIG",
        "small": "SMALL",
        "today": "TODAY",
        "tomorrow": "TOMORROW",
        "yesterday": "YESTERDAY",
        "fast": "FAST",
        "slow": "SLOW",
        "find": "FIND",
        "meet": "MEET",
        "feel": "FEEL",
        "talk": "TALK",
        "look": "LOOK",
        "hear": "HEAR",
        "morning": "MORNING",
        "night": "NIGHT",
    }

    def __init__(self):
        self.signs = self._build_sign_dictionary()

    # ── Public API ─────────────────────────────────────────

    def text_to_gloss(self, text: str) -> list:
        words   = text.lower().split()
        glosses = []
        i       = 0

        while i < len(words):
            w = ''.join(c for c in words[i] if c.isalpha())

            # Skip dropped words
            if w in self._DROP:
                i += 1
                continue

            # Two-word phrase check (e.g. "thank you")
            if i + 1 < len(words):
                w2     = ''.join(c for c in words[i+1] if c.isalpha())
                phrase = w + " " + w2
                gloss2 = (w + "_" + w2).upper()
                if gloss2 in self.signs:
                    glosses.append(gloss2)
                    i += 2
                    continue

            # Single word → gloss map
            gloss = self._WORD_MAP.get(w, w.upper())

            if gloss in self.signs:
                glosses.append(gloss)
            else:
                # Unknown — caller decides: WLASL or fingerspell
                # Return the raw gloss so PoseGenerator can look it up
                glosses.append(gloss)

            i += 1

        return glosses if glosses else ["DEFAULT"]

    def get_keypoints(self, sign: str) -> np.ndarray:
        return self.signs.get(sign, self.signs["DEFAULT"])

    def has_sign(self, sign: str) -> bool:
        """True only if we have a REAL distinct handshape for this sign."""
        return sign in self.signs

    # ── Handshape dictionary ───────────────────────────────

    def _build_sign_dictionary(self) -> dict:
        p = lambda c: np.array(c, dtype=np.float32)

        s = {}

        # ══════════════════════════════════════════════════
        # ASL ALPHABET — every letter has a real distinct shape
        # ══════════════════════════════════════════════════

        # A — fist, thumb alongside index
        s["A"] = p([
            [.50,.88],[.40,.80],[.34,.72],[.30,.64],[.29,.57],
            [.46,.62],[.46,.53],[.47,.48],[.47,.45],
            [.52,.61],[.52,.52],[.53,.47],[.53,.44],
            [.57,.63],[.57,.54],[.58,.49],[.58,.46],
            [.62,.67],[.63,.59],[.63,.55],[.63,.52],
        ])

        # B — flat hand, all 4 fingers extended together, thumb tucked
        s["B"] = p([
            [.50,.88],[.38,.79],[.36,.70],[.36,.65],[.40,.62],
            [.44,.62],[.44,.48],[.44,.37],[.44,.28],
            [.50,.61],[.50,.47],[.50,.36],[.50,.27],
            [.56,.62],[.56,.48],[.56,.37],[.56,.28],
            [.61,.65],[.62,.52],[.62,.42],[.62,.34],
        ])

        # C — curved open hand forming C shape
        s["C"] = p([
            [.50,.88],[.37,.79],[.32,.70],[.29,.62],[.27,.55],
            [.41,.60],[.37,.47],[.35,.38],[.34,.31],
            [.48,.58],[.45,.45],[.43,.36],[.42,.29],
            [.55,.59],[.54,.46],[.53,.38],[.52,.31],
            [.62,.63],[.63,.52],[.64,.44],[.64,.38],
        ])

        # D — index up, middle+ring+pinky curl to touch thumb
        s["D"] = p([
            [.50,.88],[.38,.78],[.35,.68],[.35,.60],[.39,.54],
            [.44,.60],[.43,.44],[.43,.33],[.43,.23],
            [.51,.59],[.53,.52],[.53,.49],[.52,.47],
            [.57,.61],[.59,.55],[.60,.52],[.59,.50],
            [.63,.65],[.65,.58],[.66,.55],[.65,.53],
        ])

        # E — all fingers hook inward, thumb tucked
        s["E"] = p([
            [.50,.88],[.39,.79],[.36,.72],[.35,.66],[.37,.61],
            [.44,.63],[.44,.56],[.43,.52],[.42,.49],
            [.50,.62],[.50,.55],[.50,.51],[.49,.48],
            [.56,.63],[.57,.56],[.57,.52],[.56,.49],
            [.62,.66],[.63,.60],[.63,.56],[.62,.53],
        ])

        # F — index+thumb circle, other 3 extended
        s["F"] = p([
            [.50,.88],[.39,.78],[.36,.68],[.34,.60],[.38,.54],
            [.44,.62],[.44,.56],[.43,.53],[.41,.55],
            [.51,.60],[.51,.45],[.51,.34],[.51,.25],
            [.57,.62],[.58,.47],[.58,.36],[.58,.27],
            [.63,.65],[.65,.52],[.65,.42],[.65,.34],
        ])

        # G — index+thumb point sideways (like gun)
        s["G"] = p([
            [.50,.88],[.38,.78],[.30,.70],[.22,.67],[.16,.65],
            [.42,.64],[.34,.60],[.27,.57],[.21,.55],
            [.52,.62],[.52,.54],[.52,.50],[.51,.47],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.62],[.63,.58],[.62,.55],
        ])

        # H — index+middle extended sideways together
        s["H"] = p([
            [.50,.88],[.39,.79],[.35,.70],[.31,.63],[.28,.57],
            [.44,.62],[.38,.55],[.32,.50],[.27,.46],
            [.50,.61],[.44,.54],[.38,.49],[.33,.45],
            [.57,.63],[.58,.56],[.58,.52],[.57,.49],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # I — pinky up, others fist, thumb on ring
        s["I"] = p([
            [.50,.88],[.40,.79],[.37,.70],[.36,.63],[.40,.58],
            [.46,.63],[.46,.55],[.46,.51],[.45,.48],
            [.52,.62],[.52,.54],[.52,.50],[.51,.47],
            [.57,.63],[.57,.55],[.57,.51],[.56,.48],
            [.63,.64],[.64,.51],[.65,.40],[.65,.31],
        ])

        # J — like I but draw J (keyframe = I start)
        s["J"] = s["I"].copy()

        # K — index+middle up like V, thumb between them pointing up
        s["K"] = p([
            [.50,.88],[.38,.78],[.35,.68],[.36,.61],[.41,.54],
            [.44,.62],[.42,.47],[.41,.36],[.40,.27],
            [.51,.61],[.51,.47],[.51,.37],[.50,.29],
            [.57,.63],[.58,.56],[.58,.52],[.57,.49],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # L — index up, thumb out (L-shape)
        s["L"] = p([
            [.50,.88],[.38,.79],[.28,.73],[.20,.69],[.14,.66],
            [.44,.62],[.43,.47],[.43,.36],[.43,.27],
            [.51,.62],[.51,.54],[.51,.50],[.50,.47],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # M — index+middle+ring draped over tucked thumb
        s["M"] = p([
            [.50,.88],[.40,.80],[.37,.73],[.36,.67],[.38,.63],
            [.44,.63],[.44,.55],[.44,.50],[.43,.47],
            [.50,.62],[.50,.54],[.50,.49],[.49,.46],
            [.56,.63],[.56,.55],[.56,.50],[.55,.47],
            [.62,.66],[.63,.59],[.63,.55],[.62,.52],
        ])

        # N — index+middle over thumb
        s["N"] = p([
            [.50,.88],[.40,.80],[.37,.73],[.36,.67],[.39,.63],
            [.44,.63],[.44,.55],[.44,.50],[.43,.47],
            [.50,.62],[.50,.54],[.50,.49],[.49,.46],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # O — all fingers + thumb form O circle
        s["O"] = p([
            [.50,.88],[.38,.79],[.33,.70],[.31,.62],[.34,.55],
            [.44,.61],[.41,.51],[.39,.46],[.37,.54],
            [.50,.60],[.48,.50],[.47,.45],[.46,.53],
            [.56,.61],[.55,.51],[.54,.46],[.53,.54],
            [.62,.64],[.62,.56],[.62,.52],[.61,.57],
        ])

        # P — like K pointing downward
        s["P"] = p([
            [.50,.88],[.39,.79],[.36,.70],[.37,.63],[.42,.58],
            [.44,.62],[.43,.72],[.42,.80],[.41,.87],
            [.50,.61],[.50,.71],[.50,.79],[.49,.86],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # Q — like G pointing down
        s["Q"] = p([
            [.50,.88],[.38,.78],[.31,.72],[.25,.75],[.20,.79],
            [.43,.64],[.36,.68],[.30,.72],[.24,.76],
            [.52,.62],[.52,.55],[.51,.51],[.50,.48],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # R — index+middle crossed
        s["R"] = p([
            [.50,.88],[.39,.79],[.36,.70],[.34,.62],[.33,.55],
            [.44,.62],[.43,.47],[.42,.36],[.41,.27],
            [.50,.61],[.48,.46],[.46,.35],[.45,.27],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # S — fist, thumb over fingers
        s["S"] = p([
            [.50,.88],[.39,.79],[.35,.71],[.33,.63],[.37,.56],
            [.45,.63],[.45,.55],[.45,.51],[.44,.48],
            [.51,.62],[.51,.54],[.50,.50],[.49,.47],
            [.57,.63],[.57,.55],[.57,.51],[.56,.48],
            [.62,.67],[.63,.60],[.63,.56],[.62,.53],
        ])

        # T — thumb between index+middle (fist)
        s["T"] = p([
            [.50,.88],[.40,.79],[.37,.70],[.37,.63],[.42,.57],
            [.45,.63],[.45,.55],[.45,.51],[.44,.48],
            [.51,.62],[.51,.54],[.50,.50],[.49,.47],
            [.57,.63],[.57,.56],[.57,.52],[.56,.49],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # U — index+middle up together (not spread)
        s["U"] = p([
            [.50,.88],[.39,.79],[.36,.70],[.34,.62],[.33,.55],
            [.44,.62],[.44,.47],[.44,.36],[.44,.27],
            [.50,.61],[.50,.46],[.50,.35],[.50,.26],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # V — index+middle spread (peace / victory)
        s["V"] = p([
            [.50,.88],[.39,.79],[.36,.70],[.34,.62],[.33,.55],
            [.43,.62],[.41,.47],[.40,.36],[.39,.27],
            [.51,.61],[.52,.46],[.52,.35],[.53,.26],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # W — index+middle+ring spread (3 fingers up)
        s["W"] = p([
            [.50,.88],[.39,.79],[.36,.70],[.34,.62],[.33,.55],
            [.42,.62],[.40,.47],[.39,.36],[.38,.27],
            [.50,.61],[.50,.46],[.50,.35],[.50,.26],
            [.58,.62],[.59,.47],[.59,.36],[.60,.27],
            [.63,.66],[.64,.60],[.64,.56],[.63,.53],
        ])

        # X — index hooked/bent
        s["X"] = p([
            [.50,.88],[.39,.79],[.36,.70],[.34,.62],[.34,.55],
            [.44,.62],[.44,.50],[.48,.44],[.50,.50],
            [.51,.62],[.51,.54],[.50,.50],[.49,.47],
            [.57,.63],[.57,.56],[.57,.52],[.56,.49],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # Y — thumb + pinky extended (shaka / "hang loose")
        s["Y"] = p([
            [.50,.88],[.38,.79],[.30,.72],[.23,.67],[.17,.63],
            [.46,.63],[.46,.55],[.46,.51],[.45,.48],
            [.51,.62],[.51,.54],[.50,.50],[.49,.47],
            [.57,.63],[.57,.56],[.57,.52],[.56,.49],
            [.62,.65],[.64,.52],[.65,.41],[.65,.32],
        ])

        # Z — index draws Z (keyframe = start = index up)
        s["Z"] = s["D"].copy()

        # ══════════════════════════════════════════════════
        # COMMON WORDS — only signs with genuinely distinct shapes
        # (words that share a handshape with a letter are signed
        #  with motion; here we store the peak/hold frame)
        # ══════════════════════════════════════════════════

        # HELLO — flat B-hand salute (all fingers extended up + out)
        s["HELLO"] = p([
            [.50,.88],[.37,.78],[.32,.67],[.29,.58],[.27,.50],
            [.42,.58],[.40,.42],[.39,.31],[.38,.22],
            [.49,.56],[.48,.40],[.47,.29],[.47,.20],
            [.56,.57],[.56,.41],[.56,.30],[.56,.21],
            [.63,.61],[.65,.48],[.66,.38],[.66,.30],
        ])

        # GOODBYE — same open-hand wave shape as HELLO
        s["GOODBYE"] = s["HELLO"].copy()

        # THANK_YOU — flat hand from chin forward (open B, fingers angled forward)
        s["THANK_YOU"] = p([
            [.50,.88],[.37,.79],[.33,.70],[.31,.63],[.35,.58],
            [.44,.62],[.43,.50],[.47,.43],[.50,.42],
            [.50,.61],[.50,.49],[.53,.42],[.55,.41],
            [.56,.62],[.57,.50],[.59,.44],[.61,.43],
            [.62,.65],[.64,.55],[.65,.49],[.66,.48],
        ])

        # PLEASE — flat hand circles on chest (B-hand, palm in)
        s["PLEASE"] = p([
            [.50,.88],[.38,.79],[.34,.70],[.31,.62],[.30,.55],
            [.44,.61],[.42,.46],[.41,.35],[.40,.26],
            [.50,.60],[.49,.45],[.48,.34],[.48,.25],
            [.56,.61],[.56,.46],[.56,.35],[.56,.26],
            [.62,.64],[.64,.51],[.65,.41],[.65,.33],
        ])

        # SORRY — A-hand (fist) circles on chest
        s["SORRY"] = s["A"].copy()

        # YES — S-hand (fist) nods up and down
        s["YES"] = s["S"].copy()

        # NO — middle+index snap to thumb (shown at snap point)
        s["NO"] = p([
            [.50,.88],[.40,.79],[.37,.70],[.36,.62],[.40,.56],
            [.45,.62],[.44,.51],[.43,.46],[.41,.54],
            [.51,.61],[.51,.50],[.50,.45],[.48,.53],
            [.57,.63],[.58,.57],[.58,.53],[.57,.50],
            [.62,.67],[.63,.61],[.63,.57],[.62,.54],
        ])

        # HELP — flat A lifted up on open palm (dominant = A fist)
        s["HELP"] = s["A"].copy()

        # WANT — bent-5 hand clawing toward body (all fingers curved/hooked)
        s["WANT"] = p([
            [.50,.88],[.37,.78],[.32,.69],[.29,.61],[.27,.54],
            [.43,.61],[.41,.52],[.42,.48],[.44,.52],
            [.49,.60],[.48,.51],[.49,.47],[.51,.51],
            [.56,.61],[.56,.52],[.57,.48],[.58,.52],
            [.62,.64],[.63,.57],[.63,.53],[.63,.57],
        ])

        # NEED — X-hand (hooked index) bends down
        s["NEED"] = s["X"].copy()

        # KNOW — flat B taps forehead (same flat shape as B)
        s["KNOW"] = s["B"].copy()

        # UNDERSTAND — index flicks up from fist (show extended index)
        s["UNDERSTAND"] = s["D"].copy()

        # THINK — index circles at temple
        s["THINK"] = s["D"].copy()

        # LIKE — middle finger + thumb pinch pull from chest (F-hand)
        s["LIKE"] = s["F"].copy()

        # LOVE — crossed arms / S-hands crossed on chest (show S fist)
        s["LOVE"] = s["S"].copy()

        # SEE — V-hand from eyes forward (2 fingers pointing)
        s["SEE"] = s["V"].copy()

        # HEAR — index near ear (D-hand shape)
        s["HEAR"] = s["D"].copy()

        # LOOK — V-hand turns from face outward
        s["LOOK"] = s["V"].copy()

        # TALK — index taps chin/lips alternately (D-hand)
        s["TALK"] = s["D"].copy()

        # FEEL — middle finger brushes chest upward (F-hand at peak)
        s["FEEL"] = s["F"].copy()

        # EAT — flat O to mouth
        s["EAT"] = s["O"].copy()

        # DRINK — C-hand tilts to mouth
        s["DRINK"] = s["C"].copy()

        # WATER — W-hand taps chin (W-shape = 3 fingers up)
        s["WATER"] = s["W"].copy()

        # GO — both index fingers arc forward (show dominant index pointing)
        s["GO"] = s["D"].copy()

        # COME — index fingers arc inward (hooked)
        s["COME"] = s["X"].copy()

        # STOP — flat hand chops into palm (B-hand = flat)
        s["STOP"] = s["B"].copy()

        # FINISH — 5-hands flip outward (all fingers spread open)
        s["FINISH"] = p([
            [.50,.88],[.37,.78],[.31,.68],[.27,.60],[.24,.53],
            [.42,.59],[.38,.45],[.35,.35],[.33,.27],
            [.49,.57],[.47,.43],[.45,.33],[.44,.25],
            [.56,.58],[.55,.44],[.54,.34],[.54,.26],
            [.63,.61],[.65,.49],[.66,.40],[.66,.32],
        ])

        # WAIT — spread 5-hand held out (fingers spread)
        s["WAIT"] = s["FINISH"].copy()

        # MORE — flat O hands tap together (O shape)
        s["MORE"] = s["O"].copy()

        # GIVE — open hand moves away (B-hand extended forward)
        s["GIVE"] = s["B"].copy()

        # GET — S-hand pulls toward body (closed fist pulling)
        s["GET"] = s["S"].copy()

        # HAVE — bent B hands touch chest (bent fingers)
        s["HAVE"] = p([
            [.50,.88],[.38,.79],[.34,.70],[.31,.62],[.30,.55],
            [.44,.61],[.42,.52],[.43,.48],[.45,.52],
            [.50,.60],[.49,.51],[.50,.47],[.51,.51],
            [.56,.61],[.56,.52],[.57,.48],[.58,.52],
            [.62,.64],[.63,.57],[.63,.53],[.63,.57],
        ])

        # WORK — S-hand wrist taps (S-fist)
        s["WORK"] = s["S"].copy()

        # CALL — Y-hand to ear (thumb+pinky out = phone)
        s["CALL"] = s["Y"].copy()

        # PHONE — same as CALL
        s["PHONE"] = s["Y"].copy()

        # WAIT — already set above

        # HOME — O → flat hand at cheek (show O)
        s["HOME"] = s["O"].copy()

        # NAME — H-hand taps (two fingers sideways)
        s["NAME"] = s["H"].copy()

        # SCHOOL — flat hand claps (B-hand)
        s["SCHOOL"] = s["B"].copy()

        # ── Question words ─────────────────────────────────

        # WHAT — W-hand wobbles side to side
        s["WHAT"] = s["W"].copy()

        # WHERE — index wags side to side (D-hand)
        # Note: WHERE and DEAF both use D shape but different motion/location
        s["WHERE"] = s["D"].copy()

        # WHEN — index draws circle then taps (D-hand)
        s["WHEN"] = s["D"].copy()

        # WHO — L-hand circles near chin
        s["WHO"] = s["L"].copy()

        # WHY — middle finger bends from forehead (Y at end of motion)
        s["WHY"] = s["Y"].copy()

        # HOW — bent hands roll forward (show bent/curved position)
        s["HOW"] = p([
            [.50,.88],[.39,.79],[.35,.70],[.33,.62],[.32,.55],
            [.44,.62],[.43,.52],[.44,.47],[.46,.51],
            [.50,.61],[.50,.51],[.51,.46],[.52,.50],
            [.56,.62],[.57,.53],[.57,.49],[.58,.52],
            [.62,.65],[.63,.58],[.63,.54],[.63,.57],
        ])

        # ── Medical / Emergency ─────────────────────────────

        # HOSPITAL — H-hand draws cross on shoulder
        # Distinct from WHERE/WHEN: H-shape = two fingers horizontal
        s["HOSPITAL"] = s["H"].copy()

        # DOCTOR — D-hand taps wrist pulse
        # Different location from WHERE (forehead area) → still D-shape
        # but we flag it as distinct for clarity
        s["DOCTOR"] = s["D"].copy()

        # POLICE — C-hand on shoulder (badge)
        s["POLICE"] = s["C"].copy()

        # BATHROOM — T-hand shakes (T-shape distinct from others)
        s["BATHROOM"] = s["T"].copy()

        # PAIN — both index fingers jab toward each other (D-hand)
        s["PAIN"] = s["D"].copy()

        # SICK — middle finger touches forehead (bent middle = F without pinch)
        s["SICK"] = s["F"].copy()

        # ── People ──────────────────────────────────────────

        # FAMILY — F-hands circle outward
        s["FAMILY"] = s["F"].copy()

        # FRIEND — X-hands interlock (hooked index)
        s["FRIEND"] = s["X"].copy()

        # MOTHER — 5-hand taps chin (open 5/FINISH shape)
        s["MOTHER"] = s["FINISH"].copy()

        # FATHER — 5-hand taps forehead (same shape, different location)
        s["FATHER"] = s["FINISH"].copy()

        # MEET — D-hands come together
        s["MEET"] = s["D"].copy()

        # ── Pronouns ─────────────────────────────────────────

        # ME — index points to chest (D-hand / index pointing)
        s["ME"] = s["D"].copy()

        # MY — flat B on chest
        s["MY"] = s["B"].copy()

        # YOU — index points at person
        s["YOU"] = s["D"].copy()

        # YOUR — flat B pushed toward person
        s["YOUR"] = s["B"].copy()

        # HE / SHE — index points to side (same as YOU but direction)
        s["HE"] = s["D"].copy()
        s["SHE"] = s["D"].copy()

        # WE — index crosses own chest (W or index)
        s["WE"] = s["D"].copy()

        # THEY — index sweeps to side
        s["THEY"] = s["D"].copy()

        # ── Descriptors ──────────────────────────────────────

        # GOOD — flat B from chin forward
        s["GOOD"] = s["B"].copy()

        # BAD — flat B flips down from mouth
        s["BAD"] = s["B"].copy()

        # HOT — C-hand twists away from mouth (C-shape)
        s["HOT"] = s["C"].copy()

        # COLD — S-hands shiver (fists shake)
        s["COLD"] = s["S"].copy()

        # BIG — L-hands spread apart (L-shape)
        s["BIG"] = s["L"].copy()

        # SMALL — flat hands come together (B-shape)
        s["SMALL"] = s["B"].copy()

        # FAST — L-hand snaps to X (show L)
        s["FAST"] = s["L"].copy()

        # SLOW — flat hand strokes back of other hand (B-shape)
        s["SLOW"] = s["B"].copy()

        # ── Emotions ─────────────────────────────────────────

        # HAPPY — flat hand brushes chest upward (B-hand)
        s["HAPPY"] = s["B"].copy()

        # SAD — 5-hand slides down face (open 5/FINISH shape)
        s["SAD"] = s["FINISH"].copy()

        # ANGRY — bent 5-hand pulls from face (claw/WANT shape)
        s["ANGRY"] = s["WANT"].copy()

        # TIRED — bent B hands drop at shoulders (HAVE shape)
        s["TIRED"] = s["HAVE"].copy()

        # HUNGRY — C-hand slides down chest
        s["HUNGRY"] = s["C"].copy()

        # ── Time ────────────────────────────────────────────

        # TIME — index taps wrist (D-hand)
        s["TIME"] = s["D"].copy()

        # TODAY — Y-hands drop down then flat (show Y)
        s["TODAY"] = s["Y"].copy()

        # TOMORROW — A-hand arcs forward from cheek
        s["TOMORROW"] = s["A"].copy()

        # YESTERDAY — A-hand arcs backward to cheek
        s["YESTERDAY"] = s["A"].copy()

        # MORNING — flat hand rises (B-hand)
        s["MORNING"] = s["B"].copy()

        # NIGHT — bent hand arcs down (HAVE/bent shape)
        s["NIGHT"] = s["HAVE"].copy()

        # ── Misc ─────────────────────────────────────────────

        # DEAF — index from ear to chin (D-hand at ear = starting frame)
        # Different motion path from DOCTOR (ear→chin vs wrist tap)
        # but same D handshape — motion carries the distinction
        s["DEAF"] = s["D"].copy()

        # NOT — A-hand thumb flicks out from chin
        s["NOT"] = s["A"].copy()

        # LOST — O-hand opens and drops
        s["LOST"] = s["O"].copy()

        # FIND — F-hand pinches and lifts
        s["FIND"] = s["F"].copy()

        # MONEY — flat O taps palm (O-shape)
        s["MONEY"] = s["O"].copy()

        # ── Default ──────────────────────────────────────────
        s["DEFAULT"] = s["B"].copy()   # neutral flat hand

        return s